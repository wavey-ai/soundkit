#include <float.h>
#include <limits.h>
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
#define LINE_CAPACITY 4096

typedef struct {
    int lag_frames;
    double snr_db;
} DecodeQuality;

typedef struct {
    int seconds;
    int repeats;
    int pcm_bits;
    const char *implementation;
    const char *mode;
    int frame_size;
    int bitrate;
} Options;

typedef struct {
    char implementation[16];
    char mode[16];
    int frame_size;
    double frame_ms;
    int bitrate;
} CaseKey;

static void usage(void) {
    fprintf(stderr, "usage: raw_celt_decode_dump_c [--seconds n] [--repeats n --pcm-bits 16|24] [--impl c|rust] [--mode cbr|vbr] [--frame-size n] [--bitrate bps] < packet-dump.tsv\n");
    exit(2);
}

static int parse_positive_int(const char *value) {
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > INT_MAX) {
        usage();
    }
    return (int)parsed;
}

static Options parse_options(int argc, char **argv) {
    Options options;
    options.seconds = 1;
    options.repeats = 0;
    options.pcm_bits = 24;
    options.implementation = NULL;
    options.mode = NULL;
    options.frame_size = 0;
    options.bitrate = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seconds") == 0) {
            if (++i >= argc) usage();
            options.seconds = parse_positive_int(argv[i]);
        } else if (strcmp(argv[i], "--repeats") == 0) {
            if (++i >= argc) usage();
            options.repeats = parse_positive_int(argv[i]);
        } else if (strcmp(argv[i], "--pcm-bits") == 0) {
            if (++i >= argc) usage();
            options.pcm_bits = parse_positive_int(argv[i]);
            if (options.pcm_bits != 16 && options.pcm_bits != 24) usage();
        } else if (strcmp(argv[i], "--impl") == 0) {
            if (++i >= argc) usage();
            if (strcmp(argv[i], "c") != 0 && strcmp(argv[i], "rust") != 0) usage();
            options.implementation = argv[i];
        } else if (strcmp(argv[i], "--mode") == 0) {
            if (++i >= argc) usage();
            if (strcmp(argv[i], "cbr") != 0 && strcmp(argv[i], "vbr") != 0) usage();
            options.mode = argv[i];
        } else if (strcmp(argv[i], "--frame-size") == 0) {
            if (++i >= argc) usage();
            options.frame_size = parse_positive_int(argv[i]);
        } else if (strcmp(argv[i], "--bitrate") == 0) {
            if (++i >= argc) usage();
            options.bitrate = parse_positive_int(argv[i]);
        } else {
            usage();
        }
    }
    return options;
}

static float centered_u16(uint32_t value) {
    return (float)((int)(value & 0xffffu) - 32768) * (1.0f / 32768.0f);
}

static float triangle_wave(uint32_t phase) {
    int p = (int)(phase & 0xffffu);
    int v = p < 32768 ? p - 16384 : 49152 - p;
    return (float)v * (1.0f / 16384.0f);
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
        int reference_start = lag >= 0 ? lag : 0;
        int decoded_start = lag >= 0 ? 0 : -lag;
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

        double snr_db;
        if (error <= DBL_EPSILON) {
            snr_db = INFINITY;
        } else if (signal <= DBL_EPSILON) {
            snr_db = -INFINITY;
        } else {
            snr_db = 10.0 * log10(signal / error);
        }
        if (snr_db > best.snr_db) {
            best.lag_frames = lag;
            best.snr_db = snr_db;
        }
    }

    return best;
}

static int hex_digit(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

static int decode_hex(const char *hex, unsigned char *packet, int expected_len) {
    size_t hex_len = strlen(hex);
    while (hex_len > 0 && (hex[hex_len - 1] == '\n' || hex[hex_len - 1] == '\r')) {
        hex_len--;
    }
    if ((hex_len & 1u) != 0 || (int)(hex_len / 2) != expected_len) {
        return 0;
    }
    for (size_t i = 0; i < hex_len; i += 2) {
        int hi = hex_digit(hex[i]);
        int lo = hex_digit(hex[i + 1]);
        if (hi < 0 || lo < 0) return 0;
        packet[i / 2] = (unsigned char)((hi << 4) | lo);
    }
    return 1;
}

static int case_matches(const Options *options, const CaseKey *key) {
    return (!options->implementation || strcmp(options->implementation, key->implementation) == 0)
        && (!options->mode || strcmp(options->mode, key->mode) == 0)
        && (!options->frame_size || options->frame_size == key->frame_size)
        && (!options->bitrate || options->bitrate == key->bitrate);
}

static int same_case(const CaseKey *left, const CaseKey *right) {
    return strcmp(left->implementation, right->implementation) == 0
        && strcmp(left->mode, right->mode) == 0
        && left->frame_size == right->frame_size
        && left->bitrate == right->bitrate;
}

static void decode_case(
    const CaseKey *key,
    int packet_count,
    const unsigned char *packets,
    const int *packet_lens,
    const float *reference
) {
    if (packet_count == 0) return;
    int err = OPUS_OK;
    OpusDecoder *decoder = opus_decoder_create(SAMPLE_RATE, CHANNELS, &err);
    if (err != OPUS_OK || !decoder) {
        fprintf(stderr, "opus_decoder_create failed: %s\n", opus_strerror(err));
        exit(1);
    }

    int decoded_frames_total = packet_count * key->frame_size;
    float *decoded = (float *)malloc((size_t)decoded_frames_total * CHANNELS * sizeof(float));
    if (!decoded) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    for (int frame = 0; frame < packet_count; frame++) {
        const unsigned char *packet = packets + (size_t)frame * MAX_PACKET_BYTES;
        float *output = decoded + (size_t)frame * key->frame_size * CHANNELS;
        int decoded_frames =
            opus_decode_float(decoder, packet, packet_lens[frame], output, key->frame_size, 0);
        if (decoded_frames != key->frame_size) {
            fprintf(stderr, "opus_decode_float returned %d for %s %s %d %d frame %d\n",
                    decoded_frames, key->implementation, key->mode, key->frame_size, key->bitrate, frame);
            exit(1);
        }
    }

    DecodeQuality quality = aligned_quality(reference, decoded, decoded_frames_total);
    printf(
        "%s\t%s\t%d\t%.1f\t%d\t%d\t%d\t%.2f\n",
        key->implementation,
        key->mode,
        key->frame_size,
        key->frame_ms,
        key->bitrate,
        packet_count,
        quality.lag_frames,
        quality.snr_db);

    free(decoded);
    opus_decoder_destroy(decoder);
}

static double now_ms(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec * 1000.0 + (double)value.tv_nsec / 1000000.0;
}

static int compare_double(const void *left, const void *right) {
    double a = *(const double *)left;
    double b = *(const double *)right;
    return (a > b) - (a < b);
}

static double median(double *values, int count) {
    qsort(values, (size_t)count, sizeof(*values), compare_double);
    return values[count / 2];
}

static void benchmark_decode_case(
    const Options *options,
    const CaseKey *key,
    int packet_count,
    const unsigned char *packets,
    const int *packet_lens
) {
    if (packet_count == 0) return;
    opus_int16 *decoded_i16 = options->pcm_bits == 16
        ? (opus_int16 *)malloc((size_t)key->frame_size * CHANNELS * sizeof(*decoded_i16))
        : NULL;
    opus_int32 *decoded_i24 = options->pcm_bits == 24
        ? (opus_int32 *)malloc((size_t)key->frame_size * CHANNELS * sizeof(*decoded_i24))
        : NULL;
    double *times = (double *)malloc((size_t)options->repeats * sizeof(*times));
    if ((options->pcm_bits == 16 && !decoded_i16)
        || (options->pcm_bits == 24 && !decoded_i24)
        || !times) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }

    int64_t checksum = 0;
    for (int repeat = 0; repeat < options->repeats; repeat++) {
        int err = OPUS_OK;
        OpusDecoder *decoder = opus_decoder_create(SAMPLE_RATE, CHANNELS, &err);
        if (err != OPUS_OK || !decoder) {
            fprintf(stderr, "opus_decoder_create failed: %s\n", opus_strerror(err));
            exit(1);
        }
        int64_t repeat_checksum = 0;
        double start = now_ms();
        for (int frame = 0; frame < packet_count; frame++) {
            const unsigned char *packet = packets + (size_t)frame * MAX_PACKET_BYTES;
            int decoded_frames;
            if (options->pcm_bits == 16) {
                decoded_frames = opus_decode(
                    decoder, packet, packet_lens[frame], decoded_i16, key->frame_size, 0);
                if (decoded_frames != key->frame_size) {
                    fprintf(stderr, "opus decode returned %d\n", decoded_frames);
                    exit(1);
                }
                int samples = key->frame_size * CHANNELS;
                repeat_checksum += decoded_i16[0];
                repeat_checksum += decoded_i16[samples / 2];
                repeat_checksum += decoded_i16[samples - 1];
            } else {
                decoded_frames = opus_decode24(
                    decoder, packet, packet_lens[frame], decoded_i24, key->frame_size, 0);
                if (decoded_frames != key->frame_size) {
                    fprintf(stderr, "opus decode returned %d\n", decoded_frames);
                    exit(1);
                }
                int samples = key->frame_size * CHANNELS;
                repeat_checksum += decoded_i24[0];
                repeat_checksum += decoded_i24[samples / 2];
                repeat_checksum += decoded_i24[samples - 1];
            }
        }
        times[repeat] = now_ms() - start;
        checksum = repeat_checksum;
        opus_decoder_destroy(decoder);
    }

    int packet_bytes = 0;
    for (int frame = 0; frame < packet_count; frame++) {
        packet_bytes += packet_lens[frame];
    }
    printf(
        "%s\tc\t%s\t%d\t%.1f\t%d\t%d\t%d\t%d\t%.4f\t%llu\n",
        key->implementation,
        key->mode,
        key->frame_size,
        key->frame_ms,
        key->bitrate,
        options->pcm_bits,
        packet_count,
        packet_bytes,
        median(times, options->repeats),
        (unsigned long long)checksum);

    free(times);
    free(decoded_i16);
    free(decoded_i24);
}

static void emit_case(
    const Options *options,
    const CaseKey *key,
    int packet_count,
    const unsigned char *packets,
    const int *packet_lens,
    const float *reference
) {
    if (options->repeats > 0) {
        benchmark_decode_case(options, key, packet_count, packets, packet_lens);
    } else {
        decode_case(key, packet_count, packets, packet_lens, reference);
    }
}

static int parse_line(char *line, CaseKey *key, int *frame, int *packet_len, char **hex) {
    char *cols[8];
    int count = 0;
    char *cursor = line;
    while (count < 8) {
        cols[count++] = cursor;
        char *tab = strchr(cursor, '\t');
        if (!tab) break;
        *tab = '\0';
        cursor = tab + 1;
    }
    if (count != 8 || strchr(cols[7], '\t')) {
        return 0;
    }

    snprintf(key->implementation, sizeof(key->implementation), "%s", cols[0]);
    snprintf(key->mode, sizeof(key->mode), "%s", cols[1]);
    key->frame_size = atoi(cols[2]);
    key->frame_ms = atof(cols[3]);
    key->bitrate = atoi(cols[4]);
    *frame = atoi(cols[5]);
    *packet_len = atoi(cols[6]);
    *hex = cols[7];
    return key->frame_size > 0 && key->bitrate > 0 && *frame >= 0
        && *packet_len >= 0 && *packet_len <= MAX_PACKET_BYTES;
}

int main(int argc, char **argv) {
    Options options = parse_options(argc, argv);
    int total_frames = SAMPLE_RATE * options.seconds;
    float *reference = options.repeats > 0
        ? NULL
        : generate_fixture(options.seconds, &total_frames);
    int max_packets = total_frames / 120;
    unsigned char *packets = (unsigned char *)malloc((size_t)max_packets * MAX_PACKET_BYTES);
    int *packet_lens = (int *)malloc((size_t)max_packets * sizeof(int));
    if (!packets || !packet_lens) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }

    if (options.repeats > 0) {
        printf("packet_impl\tdecoder_impl\tmode\tframe_size\tframe_ms\tbitrate\tpcm_bits\tpackets\tbytes\tdecode_ms\tchecksum\n");
    } else {
        printf("impl\tmode\tframe_size\tframe_ms\tbitrate\tframes\tquality_lag\tquality_snr_db\n");
    }

    char line[LINE_CAPACITY];
    CaseKey current;
    memset(&current, 0, sizeof(current));
    int have_current = 0;
    int packet_count = 0;
    int emitted = 0;
    while (fgets(line, sizeof(line), stdin)) {
        if (strncmp(line, "impl\t", 5) == 0 || line[0] == '\n' || line[0] == '\r') {
            continue;
        }

        CaseKey key;
        int frame = 0;
        int packet_len = 0;
        char *hex = NULL;
        if (!parse_line(line, &key, &frame, &packet_len, &hex)) {
            fprintf(stderr, "invalid packet dump row\n");
            exit(1);
        }
        if (!case_matches(&options, &key)) {
            continue;
        }
        if (have_current && !same_case(&current, &key)) {
            emit_case(&options, &current, packet_count, packets, packet_lens, reference);
            emitted++;
            packet_count = 0;
        }
        if (!have_current || !same_case(&current, &key)) {
            current = key;
            have_current = 1;
        }
        if (packet_count >= max_packets || frame != packet_count) {
            fprintf(stderr, "packet dump rows must be grouped by case and ordered by frame\n");
            exit(1);
        }
        if (!decode_hex(hex, packets + (size_t)packet_count * MAX_PACKET_BYTES, packet_len)) {
            fprintf(stderr, "invalid hex packet for %s %s %d %d frame %d\n",
                    key.implementation, key.mode, key.frame_size, key.bitrate, frame);
            exit(1);
        }
        packet_lens[packet_count++] = packet_len;
    }
    if (have_current) {
        emit_case(&options, &current, packet_count, packets, packet_lens, reference);
        emitted++;
    }
    if (emitted == 0) {
        fprintf(stderr, "no matching packets found\n");
        exit(1);
    }

    free(packet_lens);
    free(packets);
    free(reference);
    return 0;
}
