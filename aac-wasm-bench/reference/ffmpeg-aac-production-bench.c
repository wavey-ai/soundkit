#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "libavcodec/avcodec.h"
#include "libavutil/channel_layout.h"
#include "libavutil/error.h"
#include "libavutil/mem.h"
#include "libavutil/samplefmt.h"
#include "libswresample/swresample.h"

#define MAX_ADTS_FRAME_BYTES 8191
#define MAX_PCM_SAMPLES 16384
#define FNV_OFFSET UINT64_C(0xcbf29ce484222325)
#define FNV_PRIME UINT64_C(0x100000001b3)

typedef struct StreamConfig {
    int object_type;
    int sample_rate_index;
    int sample_rate;
    int channels;
} StreamConfig;

typedef struct AdtsFrame {
    size_t raw_offset;
    size_t raw_size;
    size_t frame_size;
    StreamConfig config;
} AdtsFrame;

typedef struct DecodeReport {
    uint64_t decoded_frames;
    uint64_t samples;
    uint64_t checksum;
    int sample_rate;
    int channels;
} DecodeReport;

static const int SAMPLE_RATES[] = {
    96000, 88200, 64000, 48000, 44100, 32000, 24000,
    22050, 16000, 12000, 11025, 8000, 7350,
};

static volatile int16_t output_sink;

static double now_ms(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return -1.0;
    }
    return (double)now.tv_sec * 1000.0 + (double)now.tv_nsec / 1000000.0;
}

static int read_file(const char *path, uint8_t **data, size_t *size) {
    FILE *file = fopen(path, "rb");
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
        if (file != NULL) fclose(file);
        return AVERROR(errno == 0 ? EIO : errno);
    }
    const long length = ftell(file);
    if (length <= 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return AVERROR(EIO);
    }
    uint8_t *buffer = av_malloc((size_t)length);
    if (buffer == NULL) {
        fclose(file);
        return AVERROR(ENOMEM);
    }
    if (fread(buffer, 1, (size_t)length, file) != (size_t)length) {
        av_free(buffer);
        fclose(file);
        return AVERROR(EIO);
    }
    fclose(file);
    *data = buffer;
    *size = (size_t)length;
    return 0;
}

static int parse_adts_frame(const uint8_t *data, size_t size, size_t offset,
                            AdtsFrame *frame) {
    if (offset + 7 > size || data[offset] != 0xff ||
        (data[offset + 1] & 0xf6) != 0xf0) {
        return AVERROR_INVALIDDATA;
    }

    const int header_size = (data[offset + 1] & 1) != 0 ? 7 : 9;
    const size_t frame_size = (size_t)((data[offset + 3] & 3) << 11) |
                              (size_t)(data[offset + 4] << 3) |
                              (size_t)(data[offset + 5] >> 5);
    const int sample_rate_index = (data[offset + 2] >> 2) & 15;
    const int channels = ((data[offset + 2] & 1) << 2) |
                         ((data[offset + 3] >> 6) & 3);
    if (frame_size <= (size_t)header_size || frame_size > MAX_ADTS_FRAME_BYTES ||
        offset + frame_size > size || (data[offset + 6] & 3) != 0 ||
        sample_rate_index >=
            (int)(sizeof(SAMPLE_RATES) / sizeof(SAMPLE_RATES[0])) ||
        channels < 1 || channels > 2) {
        return AVERROR_INVALIDDATA;
    }

    frame->raw_offset = offset + (size_t)header_size;
    frame->raw_size = frame_size - (size_t)header_size;
    frame->frame_size = frame_size;
    frame->config = (StreamConfig){
        .object_type = ((data[offset + 2] >> 6) & 3) + 1,
        .sample_rate_index = sample_rate_index,
        .sample_rate = SAMPLE_RATES[sample_rate_index],
        .channels = channels,
    };
    return 0;
}

static int same_config(StreamConfig left, StreamConfig right) {
    return left.object_type == right.object_type &&
           left.sample_rate_index == right.sample_rate_index &&
           left.sample_rate == right.sample_rate &&
           left.channels == right.channels;
}

static uint64_t checksum_i16(uint64_t checksum, const int16_t *samples,
                             size_t count) {
    for (size_t index = 0; index < count; ++index) {
        const uint16_t sample = (uint16_t)samples[index];
        checksum = (checksum ^ (sample & 0xff)) * FNV_PRIME;
        checksum = (checksum ^ (sample >> 8)) * FNV_PRIME;
    }
    return checksum;
}

static int decode_once(const uint8_t *data, size_t size, int collect_checksum,
                       DecodeReport *report) {
    int result = 0;
    size_t offset = 0;
    AdtsFrame first;
    AVCodecContext *context = NULL;
    AVPacket *packet = NULL;
    AVFrame *frame = NULL;
    SwrContext *resampler = NULL;
    uint8_t *packet_data = NULL;
    int16_t *pcm = NULL;
    memset(report, 0, sizeof(*report));
    report->checksum = FNV_OFFSET;

    result = parse_adts_frame(data, size, 0, &first);
    if (result < 0) goto cleanup;

    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_AAC);
    if (codec == NULL) {
        result = AVERROR_DECODER_NOT_FOUND;
        goto cleanup;
    }
    context = avcodec_alloc_context3(codec);
    packet = av_packet_alloc();
    frame = av_frame_alloc();
    packet_data = av_mallocz(MAX_ADTS_FRAME_BYTES + AV_INPUT_BUFFER_PADDING_SIZE);
    pcm = av_malloc_array(MAX_PCM_SAMPLES, sizeof(*pcm));
    if (context == NULL || packet == NULL || frame == NULL || packet_data == NULL ||
        pcm == NULL) {
        result = AVERROR(ENOMEM);
        goto cleanup;
    }

    context->extradata = av_mallocz(2 + AV_INPUT_BUFFER_PADDING_SIZE);
    if (context->extradata == NULL) {
        result = AVERROR(ENOMEM);
        goto cleanup;
    }
    context->extradata_size = 2;
    context->extradata[0] = (first.config.object_type << 3) |
                            (first.config.sample_rate_index >> 1);
    context->extradata[1] = ((first.config.sample_rate_index & 1) << 7) |
                            (first.config.channels << 3);
    result = avcodec_open2(context, codec, NULL);
    if (result < 0) goto cleanup;

    AVChannelLayout layout = {0};
    av_channel_layout_default(&layout, first.config.channels);
    result = swr_alloc_set_opts2(
        &resampler, &layout, AV_SAMPLE_FMT_S16, first.config.sample_rate,
        &layout, context->sample_fmt, first.config.sample_rate, 0, NULL);
    av_channel_layout_uninit(&layout);
    if (result < 0) goto cleanup;
    result = swr_init(resampler);
    if (result < 0) goto cleanup;

    while (offset < size) {
        AdtsFrame input;
        result = parse_adts_frame(data, size, offset, &input);
        if (result < 0 || !same_config(first.config, input.config)) {
            result = AVERROR_INVALIDDATA;
            goto cleanup;
        }
        memcpy(packet_data, data + input.raw_offset, input.raw_size);
        memset(packet_data + input.raw_size, 0, AV_INPUT_BUFFER_PADDING_SIZE);
        packet->data = packet_data;
        packet->size = (int)input.raw_size;

        result = avcodec_send_packet(context, packet);
        if (result < 0) goto cleanup;
        while ((result = avcodec_receive_frame(context, frame)) >= 0) {
            uint8_t *output_planes[] = {(uint8_t *)pcm};
            const int output_frames = swr_convert(
                resampler, output_planes,
                MAX_PCM_SAMPLES / first.config.channels,
                (const uint8_t **)frame->extended_data, frame->nb_samples);
            if (output_frames < 0) {
                result = output_frames;
                goto cleanup;
            }
            const size_t output_samples =
                (size_t)output_frames * (size_t)first.config.channels;
            if (output_samples == 0 || output_samples > MAX_PCM_SAMPLES) {
                result = AVERROR_INVALIDDATA;
                goto cleanup;
            }
            if (collect_checksum) {
                report->checksum =
                    checksum_i16(report->checksum, pcm, output_samples);
            }
            output_sink ^= pcm[0];
            output_sink ^= pcm[output_samples - 1];
            ++report->decoded_frames;
            report->samples += output_samples;
            av_frame_unref(frame);
        }
        if (result != AVERROR(EAGAIN) && result != AVERROR_EOF) goto cleanup;
        offset += input.frame_size;
    }

    report->sample_rate = first.config.sample_rate;
    report->channels = first.config.channels;
    result = 0;

cleanup:
    swr_free(&resampler);
    av_free(pcm);
    av_free(packet_data);
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&context);
    return result;
}

static int compare_doubles(const void *left, const void *right) {
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: %s FIXTURE [ITERATIONS] [ROUNDS]\n", argv[0]);
        return 2;
    }
    const int iterations = argc >= 3 ? atoi(argv[2]) : 3;
    const int rounds = argc >= 4 ? atoi(argv[3]) : 11;
    if (iterations < 1 || rounds < 1) {
        fprintf(stderr, "iterations and rounds must be positive\n");
        return 2;
    }

    uint8_t *data = NULL;
    size_t size = 0;
    int result = read_file(argv[1], &data, &size);
    if (result < 0) {
        fprintf(stderr, "could not read fixture: %s\n", argv[1]);
        return 1;
    }
    double *elapsed = calloc((size_t)rounds, sizeof(*elapsed));
    if (elapsed == NULL) {
        av_free(data);
        return 1;
    }

    DecodeReport quality;
    result = decode_once(data, size, 1, &quality);
    if (result < 0) goto failed;

    for (int round = 0; round < rounds; ++round) {
        uint64_t total_frames = 0;
        uint64_t total_samples = 0;
        const double started = now_ms();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            DecodeReport decoded;
            result = decode_once(data, size, 0, &decoded);
            if (result < 0 || decoded.decoded_frames != quality.decoded_frames ||
                decoded.samples != quality.samples ||
                decoded.sample_rate != quality.sample_rate ||
                decoded.channels != quality.channels) {
                if (result >= 0) result = AVERROR_INVALIDDATA;
                goto failed;
            }
            total_frames += decoded.decoded_frames;
            total_samples += decoded.samples;
        }
        elapsed[round] = now_ms() - started;
        const double audio_seconds =
            (double)total_samples / quality.channels / quality.sample_rate;
        const double seconds = elapsed[round] / 1000.0;
        printf("ffmpeg-aac-production fixture=%s iterations=%d decoded_frames=%llu "
               "samples=%llu sample_rate=%d channels=%d elapsed_ms=%.3f "
               "rtf=%.6f x_realtime=%.1f frames_per_sec=%.1f checksum=%016llx\n",
               argv[1], iterations, (unsigned long long)total_frames,
               (unsigned long long)total_samples, quality.sample_rate,
               quality.channels, elapsed[round], seconds / audio_seconds,
               audio_seconds / seconds, (double)total_frames / seconds,
               (unsigned long long)quality.checksum);
    }

    qsort(elapsed, (size_t)rounds, sizeof(*elapsed), compare_doubles);
    printf("ffmpeg-aac-production median_elapsed_ms=%.3f best_elapsed_ms=%.3f\n",
           elapsed[rounds / 2], elapsed[0]);
    free(elapsed);
    av_free(data);
    return 0;

failed:
    {
        char message[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(result, message, sizeof(message));
        fprintf(stderr, "FFmpeg production decode failed: %s (%d)\n", message,
                result);
    }
    free(elapsed);
    av_free(data);
    return 1;
}
