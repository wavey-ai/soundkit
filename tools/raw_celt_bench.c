#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <opus.h>

#define SAMPLE_RATE 48000
#define CHANNELS 2
#define MAX_PACKET_BYTES 1275
#define PI_F 3.14159265358979323846f

static const int FRAME_SIZES[] = {120, 240, 480, 960};
static const int BITRATES[] = {48000, 96000, 128000};

typedef enum {
    MODE_CBR = 0,
    MODE_VBR = 1,
    MODE_BOTH = 2
} BenchMode;

typedef struct {
    int repeats;
    int seconds;
    BenchMode mode;
} Options;

static void usage(void) {
    fprintf(stderr, "usage: raw_celt_bench_c [--repeats n] [--seconds n] [--mode cbr|vbr|both]\n");
    exit(2);
}

static int parse_positive_int(const char *value) {
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > 3600) {
        usage();
    }
    return (int)parsed;
}

static Options parse_options(int argc, char **argv) {
    Options options;
    options.repeats = 21;
    options.seconds = 4;
    options.mode = MODE_BOTH;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--repeats") == 0) {
            if (++i >= argc) usage();
            options.repeats = parse_positive_int(argv[i]);
        } else if (strcmp(argv[i], "--seconds") == 0) {
            if (++i >= argc) usage();
            options.seconds = parse_positive_int(argv[i]);
        } else if (strcmp(argv[i], "--mode") == 0) {
            if (++i >= argc) usage();
            if (strcmp(argv[i], "cbr") == 0) {
                options.mode = MODE_CBR;
            } else if (strcmp(argv[i], "vbr") == 0) {
                options.mode = MODE_VBR;
            } else if (strcmp(argv[i], "both") == 0) {
                options.mode = MODE_BOTH;
            } else {
                usage();
            }
        } else {
            usage();
        }
    }
    return options;
}

static double now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(1);
    }
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int cmp_double(const void *a, const void *b) {
    double left = *(const double *)a;
    double right = *(const double *)b;
    return (left > right) - (left < right);
}

static double median(double *samples, int count) {
    qsort(samples, (size_t)count, sizeof(double), cmp_double);
    return samples[count / 2];
}

static float *generate_fixture(int seconds, int *total_frames) {
    *total_frames = SAMPLE_RATE * seconds;
    float *pcm = (float *)malloc((size_t)(*total_frames) * CHANNELS * sizeof(float));
    if (!pcm) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }

    uint32_t noise = 0x12345678u;
    for (int i = 0; i < *total_frames; i++) {
        float t = (float)i / (float)SAMPLE_RATE;
        noise = noise * 1664525u + 1013904223u;
        float n = (float)(noise >> 9) / (float)(1u << 23) - 1.0f;
        float transient = 0.35f * expf(-900.0f * (t - 1.37f) * (t - 1.37f));
        float left =
            0.29f * sinf(2.0f * PI_F * 261.63f * t) +
            0.17f * sinf(2.0f * PI_F * 659.25f * t + 0.2f) +
            0.05f * sinf(2.0f * PI_F * 4210.0f * t) +
            0.015f * n +
            transient;
        float right =
            0.25f * sinf(2.0f * PI_F * 329.63f * t + 0.4f) -
            0.13f * sinf(2.0f * PI_F * 880.0f * t) +
            0.05f * sinf(2.0f * PI_F * 3910.0f * t + 0.7f) -
            0.012f * n -
            0.8f * transient;
        if (left > 1.0f) left = 1.0f;
        if (left < -1.0f) left = -1.0f;
        if (right > 1.0f) right = 1.0f;
        if (right < -1.0f) right = -1.0f;
        pcm[i * CHANNELS] = left;
        pcm[i * CHANNELS + 1] = right;
    }
    return pcm;
}

static const char *mode_label(BenchMode mode) {
    return mode == MODE_VBR ? "vbr" : "cbr";
}

static void configure_encoder(OpusEncoder *encoder, int bitrate, BenchMode mode) {
    int err = opus_encoder_ctl(encoder, OPUS_SET_BITRATE(bitrate));
    if (err != OPUS_OK) goto fail;
    err = opus_encoder_ctl(encoder, OPUS_SET_VBR(mode == MODE_VBR));
    if (err != OPUS_OK) goto fail;
    err = opus_encoder_ctl(encoder, OPUS_SET_VBR_CONSTRAINT(mode == MODE_VBR));
    if (err != OPUS_OK) goto fail;
    err = opus_encoder_ctl(encoder, OPUS_SET_BANDWIDTH(OPUS_BANDWIDTH_FULLBAND));
    if (err != OPUS_OK) goto fail;
    err = opus_encoder_ctl(encoder, OPUS_SET_SIGNAL(OPUS_SIGNAL_MUSIC));
    if (err != OPUS_OK) goto fail;
    return;
fail:
    fprintf(stderr, "opus_encoder_ctl failed: %s\n", opus_strerror(err));
    exit(1);
}

static uint64_t packet_checksum(const unsigned char *packet, int len) {
    uint64_t first = len > 0 ? packet[0] : 0;
    uint64_t last = len > 0 ? packet[len - 1] : 0;
    return (((uint64_t)len) << 16) ^ (first << 8) ^ last;
}

static float decoded_checksum(const float *decoded, int samples) {
    float first = samples > 0 ? decoded[0] : 0.0f;
    float middle = samples > 0 ? decoded[samples / 2] : 0.0f;
    float last = samples > 0 ? decoded[samples - 1] : 0.0f;
    return first + middle + last;
}

static int encode_packets_with_encoder(
    OpusEncoder *encoder,
    const float *pcm,
    int total_frames,
    int frame_size,
    unsigned char *packets,
    int *packet_lens,
    int *min_packet,
    int *max_packet,
    uint64_t *checksum
) {
    int packet_count = total_frames / frame_size;
    int bytes = 0;
    int min_len = MAX_PACKET_BYTES;
    int max_len = 0;
    uint64_t sum = 0;
    for (int frame = 0; frame < packet_count; frame++) {
        unsigned char *packet = packets + (size_t)frame * MAX_PACKET_BYTES;
        const float *input = pcm + (size_t)frame * frame_size * CHANNELS;
        int len = opus_encode_float(encoder, input, frame_size, packet, MAX_PACKET_BYTES);
        if (len < 0) {
            fprintf(stderr, "opus_encode_float failed: %s\n", opus_strerror(len));
            exit(1);
        }
        packet_lens[frame] = len;
        bytes += len;
        if (len < min_len) min_len = len;
        if (len > max_len) max_len = len;
        sum += packet_checksum(packet, len);
    }

    *min_packet = min_len;
    *max_packet = max_len;
    *checksum = sum;
    return bytes;
}

static int encode_packets(
    const float *pcm,
    int total_frames,
    int frame_size,
    int bitrate,
    BenchMode mode,
    unsigned char *packets,
    int *packet_lens,
    int *min_packet,
    int *max_packet,
    uint64_t *checksum
) {
    int err = OPUS_OK;
    OpusEncoder *encoder =
        opus_encoder_create(SAMPLE_RATE, CHANNELS, OPUS_APPLICATION_RESTRICTED_LOWDELAY, &err);
    if (err != OPUS_OK || !encoder) {
        fprintf(stderr, "opus_encoder_create failed: %s\n", opus_strerror(err));
        exit(1);
    }
    configure_encoder(encoder, bitrate, mode);
    int bytes = encode_packets_with_encoder(
        encoder, pcm, total_frames, frame_size, packets, packet_lens, min_packet, max_packet, checksum);
    opus_encoder_destroy(encoder);
    return bytes;
}

static double time_encode(
    const float *pcm,
    int total_frames,
    int frame_size,
    int bitrate,
    BenchMode mode,
    int repeats,
    unsigned char *packets,
    int *packet_lens,
    int *bytes_out,
    int *min_packet_out,
    int *max_packet_out,
    uint64_t *checksum_out
) {
    double *times = (double *)malloc((size_t)repeats * sizeof(double));
    if (!times) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    for (int repeat = 0; repeat < repeats; repeat++) {
        int err = OPUS_OK;
        OpusEncoder *encoder =
            opus_encoder_create(SAMPLE_RATE, CHANNELS, OPUS_APPLICATION_RESTRICTED_LOWDELAY, &err);
        if (err != OPUS_OK || !encoder) {
            fprintf(stderr, "opus_encoder_create failed: %s\n", opus_strerror(err));
            exit(1);
        }
        configure_encoder(encoder, bitrate, mode);
        double start = now_ms();
        *bytes_out = encode_packets_with_encoder(
            encoder,
            pcm,
            total_frames,
            frame_size,
            packets,
            packet_lens,
            min_packet_out,
            max_packet_out,
            checksum_out);
        times[repeat] = now_ms() - start;
        opus_encoder_destroy(encoder);
    }
    double value = median(times, repeats);
    free(times);
    return value;
}

static double time_decode(
    int frame_size,
    int packet_count,
    int repeats,
    const unsigned char *packets,
    const int *packet_lens,
    float *checksum_out
) {
    float *decoded = (float *)malloc((size_t)frame_size * CHANNELS * sizeof(float));
    double *times = (double *)malloc((size_t)repeats * sizeof(double));
    if (!decoded || !times) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }

    for (int repeat = 0; repeat < repeats; repeat++) {
        int err = OPUS_OK;
        OpusDecoder *decoder = opus_decoder_create(SAMPLE_RATE, CHANNELS, &err);
        if (err != OPUS_OK || !decoder) {
            fprintf(stderr, "opus_decoder_create failed: %s\n", opus_strerror(err));
            exit(1);
        }

        float checksum = 0.0f;
        double start = now_ms();
        for (int frame = 0; frame < packet_count; frame++) {
            const unsigned char *packet = packets + (size_t)frame * MAX_PACKET_BYTES;
            int decoded_frames =
                opus_decode_float(decoder, packet, packet_lens[frame], decoded, frame_size, 0);
            if (decoded_frames != frame_size) {
                fprintf(stderr, "opus_decode_float returned %d\n", decoded_frames);
                exit(1);
            }
            checksum += decoded_checksum(decoded, frame_size * CHANNELS);
        }
        times[repeat] = now_ms() - start;
        *checksum_out = checksum;
        opus_decoder_destroy(decoder);
    }

    double value = median(times, repeats);
    free(decoded);
    free(times);
    return value;
}

int main(int argc, char **argv) {
    Options options = parse_options(argc, argv);
    int total_frames = 0;
    float *pcm = generate_fixture(options.seconds, &total_frames);
    int max_packets = total_frames / FRAME_SIZES[0];
    unsigned char *packets =
        (unsigned char *)malloc((size_t)max_packets * MAX_PACKET_BYTES);
    int *packet_lens = (int *)malloc((size_t)max_packets * sizeof(int));
    if (!packets || !packet_lens) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }

    printf("impl\tmode\tframe_size\tframe_ms\tbitrate\tencode_ms\tdecode_ms\tbytes\tmin_packet\tmax_packet\tchecksum\n");
    for (BenchMode mode = MODE_CBR; mode <= MODE_VBR; mode++) {
        if (options.mode != MODE_BOTH && options.mode != mode) {
            continue;
        }
        for (size_t i = 0; i < sizeof(FRAME_SIZES) / sizeof(FRAME_SIZES[0]); i++) {
            int frame_size = FRAME_SIZES[i];
            int packet_count = total_frames / frame_size;
            for (size_t j = 0; j < sizeof(BITRATES) / sizeof(BITRATES[0]); j++) {
                int bitrate = BITRATES[j];
                int bytes = 0;
                int min_packet = 0;
                int max_packet = 0;
                uint64_t encode_sum = 0;
                double encode_ms = time_encode(
                    pcm,
                    total_frames,
                    frame_size,
                    bitrate,
                    mode,
                    options.repeats,
                    packets,
                    packet_lens,
                    &bytes,
                    &min_packet,
                    &max_packet,
                    &encode_sum);
                int packet_min_for_decode = 0;
                int packet_max_for_decode = 0;
                encode_packets(
                    pcm,
                    total_frames,
                    frame_size,
                    bitrate,
                    mode,
                    packets,
                    packet_lens,
                    &packet_min_for_decode,
                    &packet_max_for_decode,
                    &encode_sum);
                float decode_sum = 0.0f;
                double decode_ms =
                    time_decode(frame_size, packet_count, options.repeats, packets, packet_lens, &decode_sum);
                uint64_t checksum = encode_sum ^ (uint64_t)decode_sum;
                printf(
                    "c\t%s\t%d\t%.1f\t%d\t%.4f\t%.4f\t%d\t%d\t%d\t%llu\n",
                    mode_label(mode),
                    frame_size,
                    (double)frame_size * 1000.0 / (double)SAMPLE_RATE,
                    bitrate,
                    encode_ms,
                    decode_ms,
                    bytes,
                    min_packet,
                    max_packet,
                    (unsigned long long)checksum);
            }
        }
    }

    free(packet_lens);
    free(packets);
    free(pcm);
    return 0;
}
