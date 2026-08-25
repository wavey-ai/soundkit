#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define BENCH_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#include <time.h>
#define BENCH_EXPORT
#endif

#include "libavcodec/avcodec.h"
#include "libavutil/error.h"
#include "libavutil/mem.h"
#include "libavutil/samplefmt.h"

typedef struct AdtsFrame {
    uint8_t *raw;
    int raw_size;
} AdtsFrame;

typedef struct Fixture {
    AdtsFrame *frames;
    int frame_count;
    int audio_object_type;
    int sample_rate_index;
    int sample_rate;
    int channels;
} Fixture;

static uint32_t last_decoded_frames;
static double last_samples_per_channel;
static uint32_t last_checksum_high;
static uint32_t last_checksum_low;
static int last_error;

static const int sample_rates[] = {
    96000, 88200, 64000, 48000, 44100, 32000, 24000,
    22050, 16000, 12000, 11025, 8000, 7350,
};

static double bench_now_ms(void) {
#ifdef __EMSCRIPTEN__
    return emscripten_get_now();
#else
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return -1.0;
    }
    return (double)now.tv_sec * 1000.0 + (double)now.tv_nsec / 1000000.0;
#endif
}

static void free_fixture(Fixture *fixture) {
    if (fixture->frames != NULL) {
        for (int index = 0; index < fixture->frame_count; ++index) {
            av_free(fixture->frames[index].raw);
        }
        av_free(fixture->frames);
    }
    memset(fixture, 0, sizeof(*fixture));
}

static int parse_fixture(const uint8_t *data, size_t size, Fixture *fixture) {
    size_t offset = 0;
    int capacity = 0;
    memset(fixture, 0, sizeof(*fixture));

    while (offset + 7 <= size) {
        if (data[offset] != 0xff || (data[offset + 1] & 0xf6) != 0xf0) {
            return AVERROR_INVALIDDATA;
        }

        const int header_size = (data[offset + 1] & 1) != 0 ? 7 : 9;
        const int frame_size = ((data[offset + 3] & 3) << 11) |
                               (data[offset + 4] << 3) |
                               (data[offset + 5] >> 5);
        if (frame_size <= header_size || offset + (size_t)frame_size > size ||
            (data[offset + 6] & 3) != 0) {
            return AVERROR_INVALIDDATA;
        }

        if (fixture->frame_count == capacity) {
            const int next_capacity = capacity == 0 ? 1024 : capacity * 2;
            AdtsFrame *next = av_realloc_array(fixture->frames, next_capacity,
                                               sizeof(*fixture->frames));
            if (next == NULL) {
                return AVERROR(ENOMEM);
            }
            fixture->frames = next;
            capacity = next_capacity;
        }

        const int raw_size = frame_size - header_size;
        uint8_t *raw = av_mallocz((size_t)raw_size + AV_INPUT_BUFFER_PADDING_SIZE);
        if (raw == NULL) {
            return AVERROR(ENOMEM);
        }
        memcpy(raw, data + offset + header_size, raw_size);
        fixture->frames[fixture->frame_count++] = (AdtsFrame){raw, raw_size};

        if (fixture->frame_count == 1) {
            fixture->audio_object_type = ((data[offset + 2] >> 6) & 3) + 1;
            fixture->sample_rate_index = (data[offset + 2] >> 2) & 15;
            fixture->channels = ((data[offset + 2] & 1) << 2) |
                                ((data[offset + 3] >> 6) & 3);
            if (fixture->sample_rate_index >=
                (int)(sizeof(sample_rates) / sizeof(sample_rates[0]))) {
                return AVERROR_INVALIDDATA;
            }
            fixture->sample_rate = sample_rates[fixture->sample_rate_index];
        }

        offset += frame_size;
    }

    return offset == size && fixture->frame_count != 0 ? 0 : AVERROR_INVALIDDATA;
}

static uint64_t checksum_sample(uint64_t checksum, float sample) {
    uint32_t bits;
    memcpy(&bits, &sample, sizeof(bits));
    checksum ^= bits;
    return checksum * UINT64_C(0x100000001b3);
}

static int checksum_frame(const AVFrame *frame, uint64_t *checksum) {
    const int channels = frame->ch_layout.nb_channels;
    const int samples = frame->nb_samples;
    if (channels <= 0 || samples <= 0 || frame->format != AV_SAMPLE_FMT_FLTP) {
        return AVERROR_INVALIDDATA;
    }

    const int points[] = {0, samples / 2, samples - 1};
    for (int channel = 0; channel < channels; ++channel) {
        const float *plane = (const float *)frame->extended_data[channel];
        for (int point = 0; point < 3; ++point) {
            *checksum = checksum_sample(*checksum, plane[points[point]]);
        }
    }
    return 0;
}

static int decode_pass(AVCodecContext *context, AVPacket *packet, AVFrame *output,
                       const Fixture *fixture, int collect,
                       uint32_t *decoded_frames, uint64_t *samples_per_channel,
                       uint64_t *checksum) {
    for (int index = 0; index < fixture->frame_count; ++index) {
        packet->data = fixture->frames[index].raw;
        packet->size = fixture->frames[index].raw_size;

        int result = avcodec_send_packet(context, packet);
        if (result < 0) {
            return result;
        }

        while ((result = avcodec_receive_frame(context, output)) >= 0) {
            if (collect) {
                result = checksum_frame(output, checksum);
                if (result < 0) {
                    return result;
                }
                ++*decoded_frames;
                *samples_per_channel += output->nb_samples;
            }
            av_frame_unref(output);
        }
        if (result != AVERROR(EAGAIN) && result != AVERROR_EOF) {
            return result;
        }
    }
    return 0;
}

BENCH_EXPORT
double ffmpeg_aac_bench(const uint8_t *data, size_t size, int iterations) {
    Fixture fixture;
    AVCodecContext *context = NULL;
    AVPacket *packet = NULL;
    AVFrame *output = NULL;
    double elapsed = -1.0;
    uint32_t decoded_frames = 0;
    uint64_t samples_per_channel = 0;
    uint64_t checksum = UINT64_C(0xcbf29ce484222325);

    last_decoded_frames = 0;
    last_samples_per_channel = 0;
    last_checksum_high = 0;
    last_checksum_low = 0;
    last_error = 0;
    if (iterations < 1) {
        iterations = 1;
    }

    int result = parse_fixture(data, size, &fixture);
    if (result < 0) {
        last_error = result;
        goto cleanup;
    }

    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_AAC);
    if (codec == NULL) {
        last_error = AVERROR_DECODER_NOT_FOUND;
        goto cleanup;
    }
    context = avcodec_alloc_context3(codec);
    packet = av_packet_alloc();
    output = av_frame_alloc();
    if (context == NULL || packet == NULL || output == NULL) {
        last_error = AVERROR(ENOMEM);
        goto cleanup;
    }

    context->extradata = av_mallocz(2 + AV_INPUT_BUFFER_PADDING_SIZE);
    if (context->extradata == NULL) {
        last_error = AVERROR(ENOMEM);
        goto cleanup;
    }
    context->extradata_size = 2;
    context->extradata[0] = (fixture.audio_object_type << 3) |
                            (fixture.sample_rate_index >> 1);
    context->extradata[1] = ((fixture.sample_rate_index & 1) << 7) |
                            (fixture.channels << 3);

    result = avcodec_open2(context, codec, NULL);
    if (result < 0) {
        last_error = result;
        goto cleanup;
    }

    result = decode_pass(context, packet, output, &fixture, 0, &decoded_frames,
                         &samples_per_channel, &checksum);
    if (result < 0) {
        last_error = result;
        goto cleanup;
    }

    const double started = bench_now_ms();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        result = decode_pass(context, packet, output, &fixture, 1, &decoded_frames,
                             &samples_per_channel, &checksum);
        if (result < 0) {
            last_error = result;
            goto cleanup;
        }
    }
    elapsed = bench_now_ms() - started;

    last_decoded_frames = decoded_frames;
    last_samples_per_channel = (double)samples_per_channel;
    last_checksum_high = (uint32_t)(checksum >> 32);
    last_checksum_low = (uint32_t)checksum;

cleanup:
    av_frame_free(&output);
    av_packet_free(&packet);
    avcodec_free_context(&context);
    free_fixture(&fixture);
    return elapsed;
}

BENCH_EXPORT
uint32_t ffmpeg_aac_last_decoded_frames(void) {
    return last_decoded_frames;
}

BENCH_EXPORT
double ffmpeg_aac_last_samples_per_channel(void) {
    return last_samples_per_channel;
}

BENCH_EXPORT
uint32_t ffmpeg_aac_last_checksum_high(void) {
    return last_checksum_high;
}

BENCH_EXPORT
uint32_t ffmpeg_aac_last_checksum_low(void) {
    return last_checksum_low;
}

BENCH_EXPORT
int ffmpeg_aac_last_error(void) {
    return last_error;
}

#ifndef __EMSCRIPTEN__
static int compare_doubles(const void *left, const void *right) {
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

static int read_file(const char *path, uint8_t **data, size_t *size) {
    FILE *file = fopen(path, "rb");
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
        if (file != NULL) fclose(file);
        return -1;
    }
    const long length = ftell(file);
    if (length <= 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return -1;
    }
    uint8_t *buffer = malloc((size_t)length);
    if (buffer == NULL || fread(buffer, 1, (size_t)length, file) != (size_t)length) {
        free(buffer);
        fclose(file);
        return -1;
    }
    fclose(file);
    *data = buffer;
    *size = (size_t)length;
    return 0;
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
    if (read_file(argv[1], &data, &size) != 0) {
        fprintf(stderr, "could not read fixture: %s\n", argv[1]);
        return 1;
    }

    double *elapsed = calloc((size_t)rounds, sizeof(*elapsed));
    if (elapsed == NULL) {
        free(data);
        return 1;
    }

    for (int round = 0; round < rounds; ++round) {
        elapsed[round] = ffmpeg_aac_bench(data, size, iterations);
        if (elapsed[round] < 0.0 || last_error != 0) {
            char message[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(last_error, message, sizeof(message));
            fprintf(stderr, "FFmpeg decode failed: %s (%d)\n", message, last_error);
            free(elapsed);
            free(data);
            return 1;
        }
        const double audio_seconds = last_samples_per_channel / 48000.0;
        const uint64_t checksum = ((uint64_t)last_checksum_high << 32) | last_checksum_low;
        printf("ffmpeg-native decoded_frames=%u elapsed_ms=%.3f rtf=%.6f "
               "frames_per_sec=%.1f checksum=%016llx\n",
               last_decoded_frames, elapsed[round],
               elapsed[round] / 1000.0 / audio_seconds,
               (double)last_decoded_frames / (elapsed[round] / 1000.0),
               (unsigned long long)checksum);
    }

    qsort(elapsed, (size_t)rounds, sizeof(*elapsed), compare_doubles);
    printf("ffmpeg-native median_elapsed_ms=%.3f best_elapsed_ms=%.3f\n",
           elapsed[rounds / 2], elapsed[0]);
    free(elapsed);
    free(data);
    return 0;
}
#endif
