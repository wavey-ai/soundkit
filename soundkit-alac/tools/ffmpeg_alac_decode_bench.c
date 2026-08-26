#include <errno.h>
#include <inttypes.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

typedef struct {
    AVPacket **items;
    size_t length;
    size_t capacity;
} packet_list;

static void fail_ffmpeg(const char *operation, int error) {
    char message[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(error, message, sizeof(message));
    fprintf(stderr, "%s: %s\n", operation, message);
    exit(1);
}

static uint64_t monotonic_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        perror("clock_gettime");
        exit(1);
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) +
           (uint64_t)value.tv_nsec;
}

static inline void black_box_pcm(const void *pcm) {
    __asm__ __volatile__("" : : "r"(pcm) : "memory");
}

static void append_packet(packet_list *packets, const AVPacket *source) {
    if (packets->length == packets->capacity) {
        size_t capacity = packets->capacity == 0 ? 64 : packets->capacity * 2;
        AVPacket **items = realloc(packets->items, capacity * sizeof(*items));
        if (items == NULL) {
            perror("realloc");
            exit(1);
        }
        packets->items = items;
        packets->capacity = capacity;
    }
    AVPacket *packet = av_packet_clone(source);
    if (packet == NULL) {
        fprintf(stderr, "av_packet_clone failed\n");
        exit(1);
    }
    packets->items[packets->length++] = packet;
}

static void free_packets(packet_list *packets) {
    for (size_t index = 0; index < packets->length; ++index) {
        av_packet_free(&packets->items[index]);
    }
    free(packets->items);
}

static AVCodecContext *load_input(const char *path, packet_list *packets,
                                  int64_t *input_bytes) {
    AVFormatContext *format = NULL;
    int result = avformat_open_input(&format, path, NULL, NULL);
    if (result < 0) fail_ffmpeg("avformat_open_input", result);
    result = avformat_find_stream_info(format, NULL);
    if (result < 0) fail_ffmpeg("avformat_find_stream_info", result);

    int stream_index = -1;
    for (unsigned int index = 0; index < format->nb_streams; ++index) {
        if (format->streams[index]->codecpar->codec_id == AV_CODEC_ID_ALAC) {
            stream_index = (int)index;
            break;
        }
    }
    if (stream_index < 0) {
        fprintf(stderr, "ALAC stream not found\n");
        exit(1);
    }

    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_ALAC);
    if (codec == NULL) {
        fprintf(stderr, "FFmpeg ALAC decoder not found\n");
        exit(1);
    }
    AVCodecContext *decoder = avcodec_alloc_context3(codec);
    if (decoder == NULL) {
        fprintf(stderr, "avcodec_alloc_context3 failed\n");
        exit(1);
    }
    result = avcodec_parameters_to_context(
        decoder, format->streams[stream_index]->codecpar);
    if (result < 0) fail_ffmpeg("avcodec_parameters_to_context", result);
    result = avcodec_open2(decoder, codec, NULL);
    if (result < 0) fail_ffmpeg("avcodec_open2", result);

    AVPacket *packet = av_packet_alloc();
    if (packet == NULL) {
        fprintf(stderr, "av_packet_alloc failed\n");
        exit(1);
    }
    while ((result = av_read_frame(format, packet)) >= 0) {
        if (packet->stream_index == stream_index) append_packet(packets, packet);
        av_packet_unref(packet);
    }
    if (result != AVERROR_EOF) fail_ffmpeg("av_read_frame", result);
    av_packet_free(&packet);

    struct stat metadata;
    if (stat(path, &metadata) != 0) {
        perror("stat");
        exit(1);
    }
    *input_bytes = metadata.st_size;
    avformat_close_input(&format);
    return decoder;
}

static int64_t checksum_frame(const AVFrame *frame, int bit_depth) {
    enum AVSampleFormat format = (enum AVSampleFormat)frame->format;
    int planar = av_sample_fmt_is_planar(format);
    int bytes = av_get_bytes_per_sample(format);
    int channels = frame->ch_layout.nb_channels;
    int64_t checksum = 0;
    for (int sample = 0; sample < frame->nb_samples; ++sample) {
        for (int channel = 0; channel < channels; ++channel) {
            const uint8_t *base = planar ? frame->extended_data[channel]
                                         : frame->extended_data[0];
            size_t index = planar ? (size_t)sample
                                  : (size_t)sample * channels + channel;
            const uint8_t *value = base + index * (size_t)bytes;
            if (bytes == 2) {
                int16_t decoded;
                memcpy(&decoded, value, sizeof(decoded));
                checksum += decoded;
            } else if (bytes == 4) {
                int32_t decoded;
                memcpy(&decoded, value, sizeof(decoded));
                if (bit_depth > 0 && bit_depth < 32) {
                    decoded >>= 32 - bit_depth;
                }
                checksum += decoded;
            } else {
                fprintf(stderr, "unsupported FFmpeg sample width %d\n", bytes);
                exit(1);
            }
        }
    }
    return checksum;
}

static void write_pcm_frame(const AVFrame *frame, int bit_depth, FILE *sink) {
    enum AVSampleFormat format = (enum AVSampleFormat)frame->format;
    int planar = av_sample_fmt_is_planar(format);
    int bytes = av_get_bytes_per_sample(format);
    int channels = frame->ch_layout.nb_channels;
    if (bit_depth != 16 && bit_depth != 24 && bit_depth != 32) {
        fprintf(stderr, "unsupported ALAC bit depth %d\n", bit_depth);
        exit(1);
    }
    if (bytes != 2 && bytes != 4) {
        fprintf(stderr, "unsupported FFmpeg sample width %d\n", bytes);
        exit(1);
    }

    for (int sample = 0; sample < frame->nb_samples; ++sample) {
        for (int channel = 0; channel < channels; ++channel) {
            const uint8_t *base = planar ? frame->extended_data[channel]
                                         : frame->extended_data[0];
            size_t index = planar ? (size_t)sample
                                  : (size_t)sample * channels + channel;
            const uint8_t *value = base + index * (size_t)bytes;
            int32_t decoded = 0;
            if (bytes == 2) {
                int16_t sample_value;
                memcpy(&sample_value, value, sizeof(sample_value));
                decoded = sample_value;
            } else {
                memcpy(&decoded, value, sizeof(decoded));
                if (bit_depth < 32) decoded >>= 32 - bit_depth;
            }

            if (bit_depth == 16) {
                int16_t sample_value = (int16_t)decoded;
                if (fwrite(&sample_value, sizeof(sample_value), 1, sink) != 1)
                    exit(1);
            } else if (bit_depth == 24) {
                uint8_t sample_value[3] = {
                    (uint8_t)decoded,
                    (uint8_t)(decoded >> 8),
                    (uint8_t)(decoded >> 16),
                };
                if (fwrite(sample_value, sizeof(sample_value), 1, sink) != 1)
                    exit(1);
            } else if (fwrite(&decoded, sizeof(decoded), 1, sink) != 1) {
                exit(1);
            }
        }
    }
}

static size_t decode_once(AVCodecContext *decoder, const packet_list *packets,
                          AVFrame *frame, FILE *sink, int64_t *checksum) {
    size_t samples = 0;
    int64_t sum = 0;
    for (size_t index = 0; index < packets->length; ++index) {
        int result = avcodec_send_packet(decoder, packets->items[index]);
        if (result < 0) fail_ffmpeg("avcodec_send_packet", result);
        for (;;) {
            result = avcodec_receive_frame(decoder, frame);
            if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) break;
            if (result < 0) fail_ffmpeg("avcodec_receive_frame", result);
            black_box_pcm(frame->extended_data[0]);
            if (sink != NULL)
                write_pcm_frame(frame, decoder->bits_per_raw_sample, sink);
            samples += (size_t)frame->nb_samples *
                       (size_t)frame->ch_layout.nb_channels;
            if (checksum != NULL)
                sum += checksum_frame(frame, decoder->bits_per_raw_sample);
            av_frame_unref(frame);
        }
    }
    if (checksum != NULL) *checksum = sum;
    return samples;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr,
                "usage: ffmpeg_alac_decode_bench <input.m4a> [iterations] "
                "[output.pcm]\n");
        return 2;
    }
    int iterations = argc >= 3 ? atoi(argv[2]) : 50;
    if (iterations <= 0) return 2;

    av_log_set_level(AV_LOG_ERROR);
    packet_list packets = {0};
    int64_t input_bytes = 0;
    AVCodecContext *decoder = load_input(argv[1], &packets, &input_bytes);
    AVFrame *frame = av_frame_alloc();
    if (frame == NULL) return 1;

    for (int index = 0; index < 3; ++index) {
        decode_once(decoder, &packets, frame, NULL, NULL);
    }

    uint64_t started = monotonic_ns();
    size_t total_samples = 0;
    for (int index = 0; index < iterations; ++index) {
        total_samples += decode_once(decoder, &packets, frame, NULL, NULL);
    }
    uint64_t elapsed = monotonic_ns() - started;
    int64_t checksum = 0;
    if (argc == 4) {
        FILE *output = fopen(argv[3], "wb");
        if (output == NULL) {
            perror("fopen");
            return 1;
        }
        decode_once(decoder, &packets, frame, output, &checksum);
        if (fclose(output) != 0) return 1;
    } else {
        decode_once(decoder, &packets, frame, NULL, &checksum);
    }

    printf("implementation=ffmpeg-c codec=alac operation=decode "
           "input_bytes=%" PRId64 " packets=%zu iterations=%d "
           "samples=%zu sample_rate=%d channels=%d bit_depth=%d "
           "elapsed_ns=%" PRIu64 " checksum=%" PRId64 "\n",
           input_bytes, packets.length, iterations, total_samples,
           decoder->sample_rate, decoder->ch_layout.nb_channels,
           decoder->bits_per_raw_sample, elapsed, checksum);

    av_frame_free(&frame);
    avcodec_free_context(&decoder);
    free_packets(&packets);
    return 0;
}
