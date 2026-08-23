// In-process FLAC decode timing against the system libFLAC build.
//
// Mirrors examples/codec_bench.rs methodology: the compressed file is read
// into memory once, every run decodes through libFLAC's stream API, samples
// fold into a checksum sink so nothing is optimized away, and the report is
// a median over timed runs after one warm-up run.
//
// Usage:
//
//   libflac_bench FILE.flac [RUNS]
//   libflac_bench encode FILE.wav [RUNS] [LEVEL]

#include <FLAC/stream_decoder.h>
#include <FLAC/stream_encoder.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const uint8_t *data;
    size_t len;
    size_t pos;
    uint64_t sink;
} BenchState;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static FLAC__StreamDecoderReadStatus read_cb(
    const FLAC__StreamDecoder *decoder,
    FLAC__byte buffer[],
    size_t *bytes,
    void *client_data
) {
    (void)decoder;
    BenchState *state = client_data;
    size_t remaining = state->len - state->pos;
    if (*bytes == 0) {
        return FLAC__STREAM_DECODER_READ_STATUS_ABORT;
    }
    if (remaining == 0) {
        *bytes = 0;
        return FLAC__STREAM_DECODER_READ_STATUS_END_OF_STREAM;
    }
    size_t take = remaining < *bytes ? remaining : *bytes;
    memcpy(buffer, state->data + state->pos, take);
    state->pos += take;
    *bytes = take;
    return FLAC__STREAM_DECODER_READ_STATUS_CONTINUE;
}

static FLAC__StreamDecoderWriteStatus write_cb(
    const FLAC__StreamDecoder *decoder,
    const FLAC__Frame *frame,
    const FLAC__int32 *const buffer[],
    void *client_data
) {
    (void)decoder;
    BenchState *state = client_data;
    uint64_t sink = state->sink;
    for (unsigned ch = 0; ch < frame->header.channels; ch++) {
        const FLAC__int32 *samples = buffer[ch];
        for (unsigned i = 0; i < frame->header.blocksize; i++) {
            sink ^= (uint64_t)(uint32_t)samples[i];
        }
    }
    state->sink = sink;
    return FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
}

static void error_cb(
    const FLAC__StreamDecoder *decoder,
    FLAC__StreamDecoderErrorStatus status,
    void *client_data
) {
    (void)decoder;
    (void)client_data;
    fprintf(stderr, "libflac_bench: decoder error: %s\n",
            FLAC__StreamDecoderErrorStatusString[status]);
}

static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

/* --- encode mode ------------------------------------------------------- */

typedef struct {
    uint32_t sample_rate;
    uint16_t channels;
    uint16_t bits_per_sample;
    FLAC__int32 *samples; /* interleaved */
    size_t count;         /* total interleaved samples */
} PcmInput;

typedef struct {
    uint64_t sink;
} EncodeState;

static FLAC__StreamEncoderWriteStatus encode_write_cb(
    const FLAC__StreamEncoder *encoder,
    const FLAC__byte buffer[],
    size_t bytes,
    unsigned samples,
    unsigned current_frame,
    void *client_data
) {
    (void)encoder;
    (void)samples;
    (void)current_frame;
    EncodeState *state = client_data;
    for (size_t i = 0; i < bytes; i++) {
        state->sink ^= (uint64_t)buffer[i];
    }
    /* Value 0 is OK in libFLAC 1.5+ and CONTINUE before it. */
    return (FLAC__StreamEncoderWriteStatus)0;
}

static uint16_t read_le_u16(const uint8_t *p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

static uint32_t read_le_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int load_pcm_wave(const char *path, PcmInput *pcm) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        perror(path);
        return 1;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return 1;
    }
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    uint8_t *data = malloc((size_t)size);
    if (!data || fread(data, 1, (size_t)size, file) != (size_t)size ||
        size < 12 || memcmp(data, "RIFF", 4) != 0 ||
        memcmp(data + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "libflac_bench: %s is not a WAVE file\n", path);
        free(data);
        fclose(file);
        return 1;
    }
    fclose(file);

    const uint8_t *format = NULL;
    const uint8_t *payload = NULL;
    size_t payload_len = 0;
    size_t offset = 12;
    while (offset + 8 <= (size_t)size) {
        const uint8_t *id = data + offset;
        uint32_t declared = read_le_u32(data + offset + 4);
        size_t start = offset + 8;
        size_t end = start + declared;
        if (end > (size_t)size) {
            end = (size_t)size;
        }
        if (!memcmp(id, "fmt ", 4) && !format && start + 16 <= (size_t)size) {
            format = data + start;
        } else if (!memcmp(id, "data", 4)) {
            payload = data + start;
            payload_len = end - start;
        }
        offset = end + (declared & 1);
    }
    if (!format || !payload) {
        free(data);
        fprintf(stderr, "libflac_bench: missing fmt/data chunk\n");
        return 1;
    }
    pcm->channels = read_le_u16(format + 2);
    pcm->sample_rate = read_le_u32(format + 4);
    uint16_t container_bits = read_le_u16(format + 14);
    uint16_t format_tag = read_le_u16(format);
    const uint8_t *payload_base = payload;
    format = NULL;
    payload = NULL;
    free(data);
    if ((format_tag != 1 && format_tag != 0xfffe) ||
        (container_bits != 16 && container_bits != 24) ||
        pcm->channels == 0 || pcm->sample_rate == 0) {
        fprintf(stderr, "libflac_bench: unsupported WAVE geometry\n");
        return 1;
    }
    pcm->bits_per_sample = container_bits;
    size_t bytes_per_sample = container_bits / 8;
    if (payload_len % bytes_per_sample != 0) {
        fprintf(stderr, "libflac_bench: partial trailing sample\n");
        return 1;
    }
    pcm->count = payload_len / bytes_per_sample;
    pcm->samples = malloc(pcm->count * sizeof(FLAC__int32));
    if (!pcm->samples) {
        return 1;
    }
    for (size_t i = 0; i < pcm->count; i++) {
        const uint8_t *s = payload_base + i * bytes_per_sample;
        if (bytes_per_sample == 2) {
            pcm->samples[i] = (FLAC__int32)(int16_t)read_le_u16(s);
        } else {
            uint32_t u = (uint32_t)s[0] | ((uint32_t)s[1] << 8) |
                         ((uint32_t)s[2] << 16);
            pcm->samples[i] = (FLAC__int32)(u << 8) >> 8;
        }
    }
    return 0;
}

static int run_encode_once(const PcmInput *pcm, int level, double *elapsed) {
    FLAC__StreamEncoder *encoder = FLAC__stream_encoder_new();
    if (!encoder) {
        fprintf(stderr, "libflac_bench: cannot create encoder\n");
        return 1;
    }
    EncodeState state = {0};
    FLAC__stream_encoder_set_channels(encoder, pcm->channels);
    FLAC__stream_encoder_set_bits_per_sample(encoder, pcm->bits_per_sample);
    FLAC__stream_encoder_set_sample_rate(encoder, pcm->sample_rate);
    FLAC__stream_encoder_set_compression_level(encoder, (unsigned)level);
    FLAC__stream_encoder_init_stream(
        encoder, encode_write_cb, NULL, NULL, NULL, &state);

    double start = now_seconds();
    if (FLAC__stream_encoder_process_interleaved(
            encoder, pcm->samples, (unsigned)(pcm->count / pcm->channels)) ==
        0) {
        fprintf(stderr, "libflac_bench: encode failed\n");
        FLAC__stream_encoder_delete(encoder);
        return 1;
    }
    if (FLAC__stream_encoder_finish(encoder) == 0) {
        fprintf(stderr, "libflac_bench: encode finish failed\n");
        FLAC__stream_encoder_delete(encoder);
        return 1;
    }
    *elapsed = now_seconds() - start;

    FLAC__stream_encoder_delete(encoder);
    printf("checksum sink: %llu\n", (unsigned long long)state.sink);
    return 0;
}

static int run_once(BenchState *state, double *elapsed) {
    FLAC__StreamDecoder *decoder = FLAC__stream_decoder_new();
    if (!decoder) {
        fprintf(stderr, "libflac_bench: cannot create decoder\n");
        return 1;
    }
    state->pos = 0;
    state->sink = 0;
    FLAC__stream_decoder_init_stream(
        decoder, read_cb, NULL, NULL, NULL, NULL, write_cb, NULL, error_cb,
        state);

    double start = now_seconds();
    if (!FLAC__stream_decoder_process_until_end_of_stream(decoder)) {
        fprintf(stderr, "libflac_bench: decode failed\n");
        return 1;
    }
    *elapsed = now_seconds() - start;

    FLAC__stream_decoder_delete(decoder);
    return 0;
}

static int report(const char *label, int runs, double warmup, const double *timings) {
    double median = runs % 2 == 1
        ? timings[runs / 2]
        : (timings[runs / 2 - 1] + timings[runs / 2]) / 2.0;
    printf("%s: median %.4f s, min %.4f s over %d runs (warm-up %.4f s)\n",
           label, median, timings[0], runs, warmup);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "encode")) {
        if (argc < 3) {
            fprintf(stderr, "usage: libflac_bench encode FILE.wav [RUNS] [LEVEL]\n");
            return 2;
        }
        int runs = argc >= 4 ? atoi(argv[3]) : 5;
        int level = argc >= 5 ? atoi(argv[4]) : 5;
        if (runs < 1) {
            runs = 1;
        }
        if (level < 0 || level > 8) {
            level = 5;
        }
        PcmInput pcm;
        if (load_pcm_wave(argv[2], &pcm)) {
            return 1;
        }

        double warmup;
        if (run_encode_once(&pcm, level, &warmup)) {
            return 1;
        }
        double *timings = calloc((size_t)runs, sizeof(double));
        for (int i = 0; i < runs; i++) {
            if (run_encode_once(&pcm, level, &timings[i])) {
                return 1;
            }
        }
        qsort(timings, (size_t)runs, sizeof(double), compare_double);
        char label[64];
        snprintf(label, sizeof(label), "encode-libflac-l%d", level);
        report(label, runs, warmup, timings);
        free(timings);
        free(pcm.samples);
        return 0;
    }

    if (argc < 2) {
        fprintf(stderr, "usage: libflac_bench FILE.flac [RUNS]\n");
        return 2;
    }
    int runs = argc >= 3 ? atoi(argv[2]) : 5;
    if (runs < 1) {
        runs = 1;
    }

    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        perror(argv[1]);
        return 1;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    uint8_t *data = malloc((size_t)size);
    if (!data || fread(data, 1, (size_t)size, file) != (size_t)size) {
        fprintf(stderr, "libflac_bench: short read\n");
        return 1;
    }
    fclose(file);

    BenchState state = {data, (size_t)size, 0, 0};

    double warmup;
    if (run_once(&state, &warmup)) {
        return 1;
    }
    double *timings = calloc((size_t)runs, sizeof(double));
    for (int i = 0; i < runs; i++) {
        if (run_once(&state, &timings[i])) {
            return 1;
        }
    }
    qsort(timings, (size_t)runs, sizeof(double), compare_double);
    report("decode", runs, warmup, timings);
    printf("checksum sink: %llu\n", (unsigned long long)state.sink);

    free(timings);
    free(data);
    return 0;
}
