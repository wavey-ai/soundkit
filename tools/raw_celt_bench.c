#include <float.h>
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
#define QUALITY_MAX_LAG_FRAMES (SAMPLE_RATE / 50)
static const int FRAME_SIZES[] = {120, 240, 480, 960};
static const int BITRATES[] = {48000, 96000, 128000, 160000, 192000, 256000, 320000, 384000, 512000};
static const int TARGET_BITRATES[] = {192000, 256000, 320000};

typedef enum {
    MODE_CBR = 0,
    MODE_VBR = 1,
    MODE_BOTH = 2
} BenchMode;

typedef enum {
    FIXTURE_MIXED = 0,
    FIXTURE_TONE = 1
} BenchFixture;

typedef struct {
    int repeats;
    int seconds;
    BenchMode mode;
    int dump_packets;
    int dump_decode_pitch;
    int frame_size;
    int bitrate;
    BenchFixture fixture;
    const char *input_s32le;
    int pcm_bits;
    int skip_quality;
    int quality_lag;
    int quality_lag_set;
    int application;
} Options;

typedef struct {
    int lag_frames;
    double snr_db;
} DecodeQuality;

static void usage(void) {
    fprintf(stderr, "usage: raw_celt_bench_c [--repeats n] [--seconds n] [--mode cbr|vbr|both] [--application audio|restricted-lowdelay] [--frame-size n] [--bitrate n] [--fixture mixed|tone] [--input-s32le path] [--pcm-bits 16|24] [--quality-lag frames] [--skip-quality] [--dump-packets n] [--dump-decode-pitch n]\n");
    exit(2);
}

static int parse_positive_int(const char *value, long maximum) {
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > maximum) {
        usage();
    }
    return (int)parsed;
}

static int contains_int(const int *values, size_t count, int value) {
    for (size_t i = 0; i < count; i++) {
        if (values[i] == value) return 1;
    }
    return 0;
}

static int bitrate_enabled(const Options *options, int bitrate) {
    if (options->bitrate != 0) return options->bitrate == bitrate;
    return contains_int(
        TARGET_BITRATES,
        sizeof(TARGET_BITRATES) / sizeof(TARGET_BITRATES[0]),
        bitrate);
}

static Options parse_options(int argc, char **argv) {
    Options options;
    options.repeats = 21;
    options.seconds = 4;
    options.mode = MODE_BOTH;
    options.dump_packets = 0;
    options.dump_decode_pitch = 0;
    options.frame_size = 0;
    options.bitrate = 0;
    options.fixture = FIXTURE_MIXED;
    options.input_s32le = NULL;
    options.pcm_bits = 0;
    options.skip_quality = 0;
    options.quality_lag = 0;
    options.quality_lag_set = 0;
    options.application = OPUS_APPLICATION_AUDIO;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--repeats") == 0) {
            if (++i >= argc) usage();
            options.repeats = parse_positive_int(argv[i], 3600);
        } else if (strcmp(argv[i], "--seconds") == 0) {
            if (++i >= argc) usage();
            options.seconds = parse_positive_int(argv[i], 3600);
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
        } else if (strcmp(argv[i], "--application") == 0) {
            if (++i >= argc) usage();
            if (strcmp(argv[i], "audio") == 0) {
                options.application = OPUS_APPLICATION_AUDIO;
            } else if (strcmp(argv[i], "restricted-lowdelay") == 0) {
                options.application = OPUS_APPLICATION_RESTRICTED_LOWDELAY;
            } else {
                usage();
            }
        } else if (strcmp(argv[i], "--frame-size") == 0) {
            if (++i >= argc) usage();
            options.frame_size = parse_positive_int(argv[i], SAMPLE_RATE);
            if (!contains_int(
                    FRAME_SIZES,
                    sizeof(FRAME_SIZES) / sizeof(FRAME_SIZES[0]),
                    options.frame_size)) {
                usage();
            }
        } else if (strcmp(argv[i], "--bitrate") == 0) {
            if (++i >= argc) usage();
            options.bitrate = parse_positive_int(argv[i], 1000000000L);
            if (!contains_int(
                    BITRATES,
                    sizeof(BITRATES) / sizeof(BITRATES[0]),
                    options.bitrate)) {
                usage();
            }
        } else if (strcmp(argv[i], "--fixture") == 0) {
            if (++i >= argc) usage();
            if (strcmp(argv[i], "mixed") == 0) {
                options.fixture = FIXTURE_MIXED;
            } else if (strcmp(argv[i], "tone") == 0) {
                options.fixture = FIXTURE_TONE;
            } else {
                usage();
            }
        } else if (strcmp(argv[i], "--input-s32le") == 0) {
            if (++i >= argc) usage();
            options.input_s32le = argv[i];
        } else if (strcmp(argv[i], "--pcm-bits") == 0) {
            if (++i >= argc) usage();
            options.pcm_bits = parse_positive_int(argv[i], 24);
            if (options.pcm_bits != 16 && options.pcm_bits != 24) usage();
        } else if (strcmp(argv[i], "--skip-quality") == 0) {
            options.skip_quality = 1;
        } else if (strcmp(argv[i], "--quality-lag") == 0) {
            if (++i >= argc) usage();
            char *end = NULL;
            long lag = strtol(argv[i], &end, 10);
            if (end == argv[i] || *end != '\0' ||
                lag < -QUALITY_MAX_LAG_FRAMES || lag > QUALITY_MAX_LAG_FRAMES) {
                usage();
            }
            options.quality_lag = (int)lag;
            options.quality_lag_set = 1;
        } else if (strcmp(argv[i], "--dump-packets") == 0) {
            if (++i >= argc) usage();
            options.dump_packets = parse_positive_int(argv[i], 3600);
        } else if (strcmp(argv[i], "--dump-decode-pitch") == 0) {
            if (++i >= argc) usage();
            options.dump_decode_pitch = parse_positive_int(argv[i], 3600);
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

static float centered_u16(uint32_t value) {
    return (float)((int)(value & 0xffffu) - 32768) * (1.0f / 32768.0f);
}

static float triangle_wave(uint32_t phase) {
    int p = (int)(phase & 0xffffu);
    int v = p < 32768 ? p - 16384 : 49152 - p;
    return (float)v * (1.0f / 16384.0f);
}

static float *generate_fixture(int seconds, int *total_frames, BenchFixture fixture) {
    *total_frames = SAMPLE_RATE * seconds;
    float *pcm = (float *)malloc((size_t)(*total_frames) * CHANNELS * sizeof(float));
    if (!pcm) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }

    if (fixture == FIXTURE_TONE) {
        for (int i = 0; i < *total_frames; i++) {
            double phase = 6.283185307179586 * (double)i / 48.0;
            pcm[i * CHANNELS] = (float)(0.25 * sin(phase));
            pcm[i * CHANNELS + 1] = (float)(0.22 * sin(phase + 0.2));
        }
        return pcm;
    }

    uint32_t noise = 0x12345678u;
    for (int i = 0; i < *total_frames; i++) {
        noise = noise * 1664525u + 1013904223u;
        float tri_a = triangle_wave((uint32_t)i * 713u);
        float tri_b = triangle_wave((uint32_t)i * 1451u + 0x4000u);
        float tri_c = triangle_wave((uint32_t)i * 977u + 0x2000u);
        float tri_d = triangle_wave((uint32_t)i * 3511u + 0x6000u);
        float n = centered_u16(noise) * (1.0f / 4096.0f);
        uint32_t pulse = (uint32_t)i & 8191u;
        float transient = pulse < 64u ? (float)(64u - pulse) * (1.0f / 512.0f) : 0.0f;
        float left = 0.25f * tri_a + 0.125f * tri_b + n + transient;
        float right = 0.21875f * tri_c - 0.09375f * tri_d - n - 0.5f * transient;
        if (left > 1.0f) left = 1.0f;
        if (left < -1.0f) left = -1.0f;
        if (right > 1.0f) right = 1.0f;
        if (right < -1.0f) right = -1.0f;
        pcm[i * CHANNELS] = left;
        pcm[i * CHANNELS + 1] = right;
    }
    return pcm;
}

static float *load_i24_s32le(
    const char *path,
    int seconds,
    int *total_frames,
    opus_int32 **pcm_i24_out
) {
    *total_frames = SAMPLE_RATE * seconds;
    size_t sample_count = (size_t)(*total_frames) * CHANNELS;
    opus_int32 *pcm_i24 = (opus_int32 *)malloc(sample_count * sizeof(*pcm_i24));
    float *pcm = (float *)malloc(sample_count * sizeof(*pcm));
    FILE *input = fopen(path, "rb");
    if (!input) {
        perror(path);
        exit(1);
    }
    if (!pcm_i24 || !pcm) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    size_t read = fread(pcm_i24, sizeof(*pcm_i24), sample_count, input);
    if (read != sample_count) {
        fprintf(
            stderr,
            "%s contains fewer than %d seconds of stereo 48 kHz S32LE samples\n",
            path,
            seconds);
        exit(1);
    }
    fclose(input);
    for (size_t i = 0; i < sample_count; i++) {
        if (pcm_i24[i] < -8388608 || pcm_i24[i] > 8388607) {
            fprintf(stderr, "%s contains a sample outside signed 24-bit range\n", path);
            exit(1);
        }
        pcm[i] = (float)pcm_i24[i] * (1.0f / 8388608.0f);
    }
    *pcm_i24_out = pcm_i24;
    return pcm;
}

static const char *mode_label(BenchMode mode) {
    return mode == MODE_VBR ? "vbr" : "cbr";
}

static void configure_encoder(OpusEncoder *encoder, int bitrate, BenchMode mode) {
    int err = opus_encoder_ctl(encoder, OPUS_SET_BITRATE(bitrate));
    if (err != OPUS_OK) goto fail;
    err = opus_encoder_ctl(encoder, OPUS_SET_COMPLEXITY(10));
    if (err != OPUS_OK) goto fail;
    err = opus_encoder_ctl(encoder, OPUS_SET_VBR(mode == MODE_VBR));
    if (err != OPUS_OK) goto fail;
    err = opus_encoder_ctl(encoder, OPUS_SET_VBR_CONSTRAINT(mode == MODE_VBR));
    if (err != OPUS_OK) goto fail;
    err = opus_encoder_ctl(encoder, OPUS_SET_BANDWIDTH(OPUS_BANDWIDTH_FULLBAND));
    if (err != OPUS_OK) goto fail;
    err = opus_encoder_ctl(encoder, OPUS_SET_SIGNAL(OPUS_SIGNAL_MUSIC));
    if (err != OPUS_OK) goto fail;
#ifdef DISABLE_PREDICTION
    err = opus_encoder_ctl(encoder, OPUS_SET_PREDICTION_DISABLED(1));
    if (err != OPUS_OK) goto fail;
#endif
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

static DecodeQuality quality_at_lag(
    const float *reference,
    const float *decoded,
    int total_frames,
    int lag,
    int requested_frames
) {
    int reference_start = lag >= 0 ? lag : 0;
    int decoded_start = lag >= 0 ? 0 : -lag;
    int available_reference = total_frames - reference_start;
    int available_decoded = total_frames - decoded_start;
    int compare_frames = requested_frames;
    if (compare_frames > available_reference) compare_frames = available_reference;
    if (compare_frames > available_decoded) compare_frames = available_decoded;

    double signal = 0.0;
    double error = 0.0;
    for (int frame = 0; frame < compare_frames; frame++) {
        int ref_base = (reference_start + frame) * CHANNELS;
        int dec_base = (decoded_start + frame) * CHANNELS;
        for (int channel = 0; channel < CHANNELS; channel++) {
            double expected = reference[ref_base + channel];
            double actual = decoded[dec_base + channel];
            double diff = expected - actual;
            signal += expected * expected;
            error += diff * diff;
        }
    }

    DecodeQuality quality;
    quality.lag_frames = lag;
    if (error <= DBL_EPSILON) {
        quality.snr_db = INFINITY;
    } else if (signal <= DBL_EPSILON) {
        quality.snr_db = -INFINITY;
    } else {
        quality.snr_db = 10.0 * log10(signal / error);
    }
    return quality;
}

static DecodeQuality aligned_quality(const float *reference, const float *decoded, int total_frames) {
    int max_lag = QUALITY_MAX_LAG_FRAMES;
    int max_available = total_frames > 16 ? total_frames - 16 : 0;
    if (max_lag > max_available) {
        max_lag = max_available;
    }
    int compare_frames = total_frames - max_lag;
    DecodeQuality best;
    best.lag_frames = 0;
    best.snr_db = -DBL_MAX;

    for (int lag = -max_lag; lag <= max_lag; lag++) {
        DecodeQuality candidate =
            quality_at_lag(reference, decoded, total_frames, lag, compare_frames);
        if (candidate.snr_db > best.snr_db) {
            best = candidate;
        }
    }

    return best;
}

static int encode_packets_with_encoder(
    OpusEncoder *encoder,
    const float *pcm,
    const opus_int16 *pcm_i16,
    const opus_int32 *pcm_i24,
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
        const opus_int16 *input_i16 =
            pcm_i16 ? pcm_i16 + (size_t)frame * frame_size * CHANNELS : NULL;
        const opus_int32 *input_i24 =
            pcm_i24 ? pcm_i24 + (size_t)frame * frame_size * CHANNELS : NULL;
        int len;
        if (pcm_i16) {
            len = opus_encode(encoder, input_i16, frame_size, packet, MAX_PACKET_BYTES);
        } else if (pcm_i24) {
            len = (int)opus_encode24(encoder, input_i24, frame_size, packet, MAX_PACKET_BYTES);
        } else {
            len = opus_encode_float(encoder, input, frame_size, packet, MAX_PACKET_BYTES);
        }
        if (len < 0) {
            fprintf(stderr, "opus encode failed: %s\n", opus_strerror(len));
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
    const opus_int16 *pcm_i16,
    const opus_int32 *pcm_i24,
    int total_frames,
    int frame_size,
    int bitrate,
    BenchMode mode,
    int application,
    unsigned char *packets,
    int *packet_lens,
    int *min_packet,
    int *max_packet,
    uint64_t *checksum
) {
    int err = OPUS_OK;
    OpusEncoder *encoder =
        opus_encoder_create(SAMPLE_RATE, CHANNELS, application, &err);
    if (err != OPUS_OK || !encoder) {
        fprintf(stderr, "opus_encoder_create failed: %s\n", opus_strerror(err));
        exit(1);
    }
    configure_encoder(encoder, bitrate, mode);
    int bytes = encode_packets_with_encoder(
        encoder,
        pcm,
        pcm_i16,
        pcm_i24,
        total_frames,
        frame_size,
        packets,
        packet_lens,
        min_packet,
        max_packet,
        checksum);
    opus_encoder_destroy(encoder);
    return bytes;
}

static double time_encode(
    const float *pcm,
    const opus_int16 *pcm_i16,
    const opus_int32 *pcm_i24,
    int total_frames,
    int frame_size,
    int bitrate,
    BenchMode mode,
    int application,
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
            opus_encoder_create(SAMPLE_RATE, CHANNELS, application, &err);
        if (err != OPUS_OK || !encoder) {
            fprintf(stderr, "opus_encoder_create failed: %s\n", opus_strerror(err));
            exit(1);
        }
        configure_encoder(encoder, bitrate, mode);
        double start = now_ms();
        *bytes_out = encode_packets_with_encoder(
            encoder,
            pcm,
            pcm_i16,
            pcm_i24,
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

static void decode_packets_to_buffer(
    int frame_size,
    int packet_count,
    const unsigned char *packets,
    const int *packet_lens,
    float *decoded,
    int pcm_bits
) {
    int err = OPUS_OK;
    OpusDecoder *decoder = opus_decoder_create(SAMPLE_RATE, CHANNELS, &err);
    if (err != OPUS_OK || !decoder) {
        fprintf(stderr, "opus_decoder_create failed: %s\n", opus_strerror(err));
        exit(1);
    }

    opus_int16 *decoded_i16 = pcm_bits == 16
        ? (opus_int16 *)malloc((size_t)frame_size * CHANNELS * sizeof(*decoded_i16))
        : NULL;
    opus_int32 *decoded_i24 = pcm_bits == 24
        ? (opus_int32 *)malloc((size_t)frame_size * CHANNELS * sizeof(*decoded_i24))
        : NULL;
    if ((pcm_bits == 16 && !decoded_i16) || (pcm_bits == 24 && !decoded_i24)) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    for (int frame = 0; frame < packet_count; frame++) {
        const unsigned char *packet = packets + (size_t)frame * MAX_PACKET_BYTES;
        float *output = decoded + (size_t)frame * frame_size * CHANNELS;
        int decoded_frames;
        if (pcm_bits == 16) {
            decoded_frames = opus_decode(
                decoder, packet, packet_lens[frame], decoded_i16, frame_size, 0);
        } else if (pcm_bits == 24) {
            decoded_frames = opus_decode24(
                decoder, packet, packet_lens[frame], decoded_i24, frame_size, 0);
        } else {
            decoded_frames = opus_decode_float(
                decoder, packet, packet_lens[frame], output, frame_size, 0);
        }
        if (decoded_frames != frame_size) {
            fprintf(stderr, "opus decode returned %d\n", decoded_frames);
            exit(1);
        }
        if (pcm_bits == 16) {
            for (int sample = 0; sample < frame_size * CHANNELS; sample++) {
                output[sample] = (float)decoded_i16[sample] * (1.0f / 32768.0f);
            }
        } else if (pcm_bits == 24) {
            for (int sample = 0; sample < frame_size * CHANNELS; sample++) {
                output[sample] = (float)decoded_i24[sample] * (1.0f / 8388608.0f);
            }
        }
    }
    free(decoded_i16);
    free(decoded_i24);
    opus_decoder_destroy(decoder);
}

static double time_decode(
    int frame_size,
    int packet_count,
    int repeats,
    const unsigned char *packets,
    const int *packet_lens,
    uint64_t *checksum_out,
    int pcm_bits
) {
    float *decoded = pcm_bits != 0
        ? NULL
        : (float *)malloc((size_t)frame_size * CHANNELS * sizeof(float));
    opus_int16 *decoded_i16 = pcm_bits == 16
        ? (opus_int16 *)malloc((size_t)frame_size * CHANNELS * sizeof(*decoded_i16))
        : NULL;
    opus_int32 *decoded_i24 = pcm_bits == 24
        ? (opus_int32 *)malloc((size_t)frame_size * CHANNELS * sizeof(*decoded_i24))
        : NULL;
    double *times = (double *)malloc((size_t)repeats * sizeof(double));
    if ((pcm_bits == 0 && !decoded) ||
        (pcm_bits == 16 && !decoded_i16) ||
        (pcm_bits == 24 && !decoded_i24) ||
        !times) {
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
        int64_t checksum_integer = 0;
        double start = now_ms();
        for (int frame = 0; frame < packet_count; frame++) {
            const unsigned char *packet = packets + (size_t)frame * MAX_PACKET_BYTES;
            int decoded_frames;
            if (pcm_bits == 16) {
                decoded_frames = opus_decode(
                    decoder, packet, packet_lens[frame], decoded_i16, frame_size, 0);
            } else if (pcm_bits == 24) {
                decoded_frames = opus_decode24(
                    decoder, packet, packet_lens[frame], decoded_i24, frame_size, 0);
            } else {
                decoded_frames = opus_decode_float(
                    decoder, packet, packet_lens[frame], decoded, frame_size, 0);
            }
            if (decoded_frames != frame_size) {
                fprintf(stderr, "opus decode returned %d\n", decoded_frames);
                exit(1);
            }
            if (pcm_bits == 16) {
                int samples = frame_size * CHANNELS;
                checksum_integer += decoded_i16[0];
                checksum_integer += decoded_i16[samples / 2];
                checksum_integer += decoded_i16[samples - 1];
            } else if (pcm_bits == 24) {
                int samples = frame_size * CHANNELS;
                checksum_integer += decoded_i24[0];
                checksum_integer += decoded_i24[samples / 2];
                checksum_integer += decoded_i24[samples - 1];
            } else {
                checksum += decoded_checksum(decoded, frame_size * CHANNELS);
            }
        }
        times[repeat] = now_ms() - start;
        if (pcm_bits != 0) {
            *checksum_out = (uint64_t)checksum_integer;
        } else {
            uint32_t checksum_bits;
            memcpy(&checksum_bits, &checksum, sizeof(checksum_bits));
            *checksum_out = checksum_bits;
        }
        opus_decoder_destroy(decoder);
    }

    double value = median(times, repeats);
    free(decoded);
    free(decoded_i16);
    free(decoded_i24);
    free(times);
    return value;
}

static void print_packet_hex(const unsigned char *packet, int len) {
    for (int i = 0; i < len; i++) {
        printf("%02x", packet[i]);
    }
    printf("\n");
}

static void dump_packets(
    const Options *options,
    const float *pcm,
    const opus_int16 *pcm_i16,
    const opus_int32 *pcm_i24,
    int total_frames,
    unsigned char *packets,
    int *packet_lens
) {
    printf("impl\tmode\tframe_size\tframe_ms\tbitrate\tframe\tlen\thex\n");
    for (BenchMode mode = MODE_CBR; mode <= MODE_VBR; mode++) {
        if (options->mode != MODE_BOTH && options->mode != mode) {
            continue;
        }
        for (size_t i = 0; i < sizeof(FRAME_SIZES) / sizeof(FRAME_SIZES[0]); i++) {
            int frame_size = FRAME_SIZES[i];
            if (options->frame_size != 0 && options->frame_size != frame_size) continue;
            int packet_count = total_frames / frame_size;
            for (size_t j = 0; j < sizeof(BITRATES) / sizeof(BITRATES[0]); j++) {
                int bitrate = BITRATES[j];
                if (!bitrate_enabled(options, bitrate)) continue;
                int min_packet = 0;
                int max_packet = 0;
                uint64_t checksum = 0;
                encode_packets(
                    pcm,
                    pcm_i16,
                    pcm_i24,
                    total_frames,
                    frame_size,
                    bitrate,
                    mode,
                    options->application,
                    packets,
                    packet_lens,
                    &min_packet,
                    &max_packet,
                    &checksum);
                int limit = options->dump_packets < packet_count ? options->dump_packets : packet_count;
                for (int frame = 0; frame < limit; frame++) {
                    const unsigned char *packet = packets + (size_t)frame * MAX_PACKET_BYTES;
                    printf(
                        "c\t%s\t%d\t%.1f\t%d\t%d\t%d\t",
                        mode_label(mode),
                        frame_size,
                        (double)frame_size * 1000.0 / (double)SAMPLE_RATE,
                        bitrate,
                        frame,
                        packet_lens[frame]);
                    print_packet_hex(packet, packet_lens[frame]);
                }
            }
        }
    }
}

static void dump_decode_pitch(
    const Options *options,
    const float *pcm,
    const opus_int16 *pcm_i16,
    const opus_int32 *pcm_i24,
    int total_frames,
    unsigned char *packets,
    int *packet_lens
) {
    float *decoded = (float *)malloc((size_t)FRAME_SIZES[3] * CHANNELS * sizeof(float));
    if (!decoded) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }

    printf("impl\tmode\tframe_size\tframe_ms\tbitrate\tframe\tlen\tpitch\n");
    for (BenchMode mode = MODE_CBR; mode <= MODE_VBR; mode++) {
        if (options->mode != MODE_BOTH && options->mode != mode) {
            continue;
        }
        for (size_t i = 0; i < sizeof(FRAME_SIZES) / sizeof(FRAME_SIZES[0]); i++) {
            int frame_size = FRAME_SIZES[i];
            if (options->frame_size != 0 && options->frame_size != frame_size) continue;
            int packet_count = total_frames / frame_size;
            for (size_t j = 0; j < sizeof(BITRATES) / sizeof(BITRATES[0]); j++) {
                int bitrate = BITRATES[j];
                if (!bitrate_enabled(options, bitrate)) continue;
                int min_packet = 0;
                int max_packet = 0;
                uint64_t checksum = 0;
                int err = OPUS_OK;
                OpusDecoder *decoder = NULL;
                int limit = options->dump_decode_pitch < packet_count ? options->dump_decode_pitch : packet_count;
                encode_packets(
                    pcm,
                    pcm_i16,
                    pcm_i24,
                    total_frames,
                    frame_size,
                    bitrate,
                    mode,
                    options->application,
                    packets,
                    packet_lens,
                    &min_packet,
                    &max_packet,
                    &checksum);
                decoder = opus_decoder_create(SAMPLE_RATE, CHANNELS, &err);
                if (err != OPUS_OK || !decoder) {
                    fprintf(stderr, "opus_decoder_create failed: %s\n", opus_strerror(err));
                    exit(1);
                }
                for (int frame = 0; frame < limit; frame++) {
                    const unsigned char *packet = packets + (size_t)frame * MAX_PACKET_BYTES;
                    int decoded_frames =
                        opus_decode_float(decoder, packet, packet_lens[frame], decoded, frame_size, 0);
                    int pitch = 0;
                    if (decoded_frames != frame_size) {
                        fprintf(stderr, "opus_decode_float returned %d\n", decoded_frames);
                        exit(1);
                    }
                    err = opus_decoder_ctl(decoder, OPUS_GET_PITCH(&pitch));
                    if (err != OPUS_OK) {
                        fprintf(stderr, "OPUS_GET_PITCH failed: %s\n", opus_strerror(err));
                        exit(1);
                    }
                    printf(
                        "c\t%s\t%d\t%.1f\t%d\t%d\t%d\t%d\n",
                        mode_label(mode),
                        frame_size,
                        (double)frame_size * 1000.0 / (double)SAMPLE_RATE,
                        bitrate,
                        frame,
                        packet_lens[frame],
                        pitch);
                }
                opus_decoder_destroy(decoder);
            }
        }
    }

    free(decoded);
}

int main(int argc, char **argv) {
    Options options = parse_options(argc, argv);
    int total_frames = 0;
    opus_int16 *pcm_i16 = NULL;
    opus_int32 *pcm_i24 = NULL;
    float *pcm;
    if (options.input_s32le) {
        pcm = load_i24_s32le(
            options.input_s32le, options.seconds, &total_frames, &pcm_i24);
        if (options.pcm_bits == 0) options.pcm_bits = 24;
        if (options.pcm_bits == 16) {
            size_t sample_count = (size_t)total_frames * CHANNELS;
            pcm_i16 = (opus_int16 *)malloc(sample_count * sizeof(*pcm_i16));
            if (!pcm_i16) {
                fprintf(stderr, "out of memory\n");
                exit(1);
            }
            for (size_t i = 0; i < sample_count; i++) {
                opus_int32 sample = pcm_i24[i];
                opus_int32 high_bits = sample >= 0
                    ? sample / 256
                    : -((-sample + 255) / 256);
                pcm_i16[i] = (opus_int16)high_bits;
                pcm[i] = (float)pcm_i16[i] * (1.0f / 32768.0f);
            }
            free(pcm_i24);
            pcm_i24 = NULL;
        }
    } else {
        pcm = generate_fixture(options.seconds, &total_frames, options.fixture);
        size_t sample_count = (size_t)total_frames * CHANNELS;
        if (options.pcm_bits == 16) {
            pcm_i16 = (opus_int16 *)malloc(sample_count * sizeof(*pcm_i16));
            if (!pcm_i16) {
                fprintf(stderr, "out of memory\n");
                exit(1);
            }
            for (size_t i = 0; i < sample_count; i++) {
                float value = roundf(pcm[i] * 32768.0f);
                if (value < -32768.0f) value = -32768.0f;
                if (value > 32767.0f) value = 32767.0f;
                pcm_i16[i] = (opus_int16)value;
                pcm[i] = (float)pcm_i16[i] * (1.0f / 32768.0f);
            }
        } else if (options.pcm_bits == 24) {
            pcm_i24 = (opus_int32 *)malloc(sample_count * sizeof(*pcm_i24));
            if (!pcm_i24) {
                fprintf(stderr, "out of memory\n");
                exit(1);
            }
            for (size_t i = 0; i < sample_count; i++) {
                float value = roundf(pcm[i] * 8388608.0f);
                if (value < -8388608.0f) value = -8388608.0f;
                if (value > 8388607.0f) value = 8388607.0f;
                pcm_i24[i] = (opus_int32)value;
                pcm[i] = (float)pcm_i24[i] * (1.0f / 8388608.0f);
            }
        }
    }
    int max_packets = total_frames / FRAME_SIZES[0];
    unsigned char *packets =
        (unsigned char *)malloc((size_t)max_packets * MAX_PACKET_BYTES);
    int *packet_lens = (int *)malloc((size_t)max_packets * sizeof(int));
    float *quality_decoded = options.skip_quality
        ? NULL
        : (float *)malloc((size_t)total_frames * CHANNELS * sizeof(float));
    if (!packets || !packet_lens || (!options.skip_quality && !quality_decoded)) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }

    if (options.dump_packets > 0) {
        dump_packets(&options, pcm, pcm_i16, pcm_i24, total_frames, packets, packet_lens);
        free(quality_decoded);
        free(packet_lens);
        free(packets);
        free(pcm_i16);
        free(pcm_i24);
        free(pcm);
        return 0;
    }
    if (options.dump_decode_pitch > 0) {
        dump_decode_pitch(
            &options, pcm, pcm_i16, pcm_i24, total_frames, packets, packet_lens);
        free(quality_decoded);
        free(packet_lens);
        free(packets);
        free(pcm_i16);
        free(pcm_i24);
        free(pcm);
        return 0;
    }

    printf("impl\tmode\tframe_size\tframe_ms\tbitrate\tencode_ms\tdecode_ms\tbytes\tmin_packet\tmax_packet\tchecksum\tquality_lag\tquality_snr_db\n");
    for (BenchMode mode = MODE_CBR; mode <= MODE_VBR; mode++) {
        if (options.mode != MODE_BOTH && options.mode != mode) {
            continue;
        }
        for (size_t i = 0; i < sizeof(FRAME_SIZES) / sizeof(FRAME_SIZES[0]); i++) {
            int frame_size = FRAME_SIZES[i];
            if (options.frame_size != 0 && options.frame_size != frame_size) continue;
            int packet_count = total_frames / frame_size;
            for (size_t j = 0; j < sizeof(BITRATES) / sizeof(BITRATES[0]); j++) {
                int bitrate = BITRATES[j];
                if (!bitrate_enabled(&options, bitrate)) continue;
                int bytes = 0;
                int min_packet = 0;
                int max_packet = 0;
                uint64_t encode_sum = 0;
                double encode_ms = time_encode(
                    pcm,
                    pcm_i16,
                    pcm_i24,
                    total_frames,
                    frame_size,
                    bitrate,
                    mode,
                    options.application,
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
                    pcm_i16,
                    pcm_i24,
                    total_frames,
                    frame_size,
                    bitrate,
                    mode,
                    options.application,
                    packets,
                    packet_lens,
                    &packet_min_for_decode,
                    &packet_max_for_decode,
                    &encode_sum);
                uint64_t decode_sum = 0;
                DecodeQuality quality;
                if (options.skip_quality) {
                    quality.lag_frames = 0;
                    quality.snr_db = NAN;
                } else {
                    decode_packets_to_buffer(
                        frame_size,
                        packet_count,
                        packets,
                        packet_lens,
                        quality_decoded,
                        options.pcm_bits);
                    int quality_frames = packet_count * frame_size;
                    if (options.quality_lag_set) {
                        int trim = QUALITY_MAX_LAG_FRAMES;
                        int max_available = quality_frames > 16 ? quality_frames - 16 : 0;
                        if (trim > max_available) trim = max_available;
                        quality = quality_at_lag(
                            pcm,
                            quality_decoded,
                            quality_frames,
                            options.quality_lag,
                            quality_frames - trim);
                    } else {
                        quality = aligned_quality(pcm, quality_decoded, quality_frames);
                    }
                }
                double decode_ms =
                    time_decode(
                        frame_size,
                        packet_count,
                        options.repeats,
                        packets,
                        packet_lens,
                        &decode_sum,
                        options.pcm_bits);
                uint64_t checksum = encode_sum ^ decode_sum;
                printf(
                    "c\t%s\t%d\t%.1f\t%d\t%.4f\t%.4f\t%d\t%d\t%d\t%llu\t%d\t%.2f\n",
                    mode_label(mode),
                    frame_size,
                    (double)frame_size * 1000.0 / (double)SAMPLE_RATE,
                    bitrate,
                    encode_ms,
                    decode_ms,
                    bytes,
                    min_packet,
                    max_packet,
                    (unsigned long long)checksum,
                    quality.lag_frames,
                    quality.snr_db);
            }
        }
    }

    free(quality_decoded);
    free(packet_lens);
    free(packets);
    free(pcm_i16);
    free(pcm_i24);
    free(pcm);
    return 0;
}
