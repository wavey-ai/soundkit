// Persistent libFLAC reference over a sequence of 5 ms PCM frames.
//
// Build:
//   cc -O3 scripts/libflac_frame_bench.c $(pkg-config --cflags --libs flac) \
//      -lm -o target/release/libflac_frame_bench
//
// Usage:
//   libflac_frame_bench RATE LEVEL ITERATIONS RUNS [PCM_S32LE [PACKET_BUNDLE]]

#include <FLAC/stream_encoder.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "benchmark_clock.h"

// libFLAC 1.4 exports this setting but Debian 12's public header predates its
// declaration.
extern FLAC__bool FLAC__stream_encoder_set_do_md5(
    FLAC__StreamEncoder *encoder,
    FLAC__bool value
);

enum { CHANNELS = 2, BITS_PER_SAMPLE = 24, WARMUP = 1024 };

typedef struct {
    uint64_t bytes;
    FILE *bundle;
    int failed;
} Output;

typedef struct {
    FLAC__int32 *samples;
    size_t sample_count;
} Pcm;

static int compare_double(const void *left, const void *right) {
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return a < b ? -1 : a > b;
}

static double percentile(const double *sorted, size_t count, unsigned percent) {
    size_t rank = (count * percent + 99) / 100;
    if (rank == 0)
        rank = 1;
    if (rank > count)
        rank = count;
    return sorted[rank - 1];
}

static int write_u32le(FILE *file, uint32_t value) {
    const uint8_t bytes[4] = {
        (uint8_t)value,
        (uint8_t)(value >> 8),
        (uint8_t)(value >> 16),
        (uint8_t)(value >> 24),
    };
    return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes) ? 0 : 1;
}

static FLAC__StreamEncoderWriteStatus write_callback(
    const FLAC__StreamEncoder *encoder,
    const FLAC__byte buffer[],
    size_t bytes,
    unsigned samples,
    unsigned current_frame,
    void *client_data
) {
    (void)encoder;
    (void)current_frame;
    Output *output = client_data;
    if (samples == 0)
        return FLAC__STREAM_ENCODER_WRITE_STATUS_OK;
    output->bytes += bytes;
    if (output->bundle) {
        if (bytes > UINT32_MAX || write_u32le(output->bundle, (uint32_t)bytes) ||
            fwrite(buffer, 1, bytes, output->bundle) != bytes) {
            output->failed = 1;
            return FLAC__STREAM_ENCODER_WRITE_STATUS_FATAL_ERROR;
        }
    }
    return FLAC__STREAM_ENCODER_WRITE_STATUS_OK;
}

static FLAC__StreamEncoder *new_encoder(
    unsigned sample_rate,
    unsigned frame_length,
    unsigned level,
    Output *output
) {
    FLAC__StreamEncoder *encoder = FLAC__stream_encoder_new();
    if (!encoder)
        return NULL;
    const int ok =
        FLAC__stream_encoder_set_compression_level(encoder, level) &&
        FLAC__stream_encoder_set_channels(encoder, CHANNELS) &&
        FLAC__stream_encoder_set_bits_per_sample(encoder, BITS_PER_SAMPLE) &&
        FLAC__stream_encoder_set_sample_rate(encoder, sample_rate) &&
        FLAC__stream_encoder_set_blocksize(encoder, frame_length) &&
        FLAC__stream_encoder_set_do_md5(encoder, false);
    if (!ok || FLAC__stream_encoder_init_stream(
                   encoder, write_callback, NULL, NULL, NULL, output) !=
                   FLAC__STREAM_ENCODER_INIT_STATUS_OK) {
        FLAC__stream_encoder_delete(encoder);
        return NULL;
    }
    return encoder;
}

static Pcm synthetic_pcm(unsigned sample_rate, unsigned frame_length) {
    Pcm pcm = {0};
    pcm.sample_count = (size_t)frame_length * CHANNELS;
    pcm.samples = malloc(pcm.sample_count * sizeof(*pcm.samples));
    if (!pcm.samples)
        return pcm;
    const double tau = 6.283185307179586476925286766559;
    for (size_t index = 0; index < pcm.sample_count; index++) {
        const double phase = (double)index * 440.0 * tau / sample_rate;
        pcm.samples[index] = (FLAC__int32)(sin(phase) * 2000000.0);
    }
    return pcm;
}

static Pcm read_pcm(const char *path) {
    Pcm pcm = {0};
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) {
        if (file)
            fclose(file);
        return pcm;
    }
    const long length = ftell(file);
    if (length <= 0 || length % 4 != 0 || fseek(file, 0, SEEK_SET)) {
        fclose(file);
        return pcm;
    }
    const size_t bytes = (size_t)length;
    uint8_t *encoded = malloc(bytes);
    pcm.samples = malloc(bytes);
    if (!encoded || !pcm.samples || fread(encoded, 1, bytes, file) != bytes) {
        free(encoded);
        free(pcm.samples);
        pcm.samples = NULL;
        fclose(file);
        return pcm;
    }
    fclose(file);
    pcm.sample_count = bytes / 4;
    for (size_t index = 0; index < pcm.sample_count; index++) {
        const uint8_t *sample = &encoded[index * 4];
        pcm.samples[index] = (FLAC__int32)(
            (uint32_t)sample[0] |
            (uint32_t)sample[1] << 8 |
            (uint32_t)sample[2] << 16 |
            (uint32_t)sample[3] << 24
        );
    }
    free(encoded);
    return pcm;
}

static int generate_bundle(
    const char *path,
    const Pcm *pcm,
    size_t corpus_frames,
    unsigned sample_rate,
    unsigned frame_length,
    unsigned level
) {
    Output output = {0};
    output.bundle = fopen(path, "wb");
    if (!output.bundle)
        return 1;
    FLAC__StreamEncoder *encoder =
        new_encoder(sample_rate, frame_length, level, &output);
    if (!encoder) {
        fclose(output.bundle);
        return 1;
    }
    const size_t samples_per_frame = (size_t)frame_length * CHANNELS;
    for (size_t frame = 0; frame < corpus_frames; frame++) {
        if (!FLAC__stream_encoder_process_interleaved(
                encoder, &pcm->samples[frame * samples_per_frame], frame_length)) {
            output.failed = 1;
            break;
        }
    }
    const int failed = output.failed || !FLAC__stream_encoder_finish(encoder) ||
                       fclose(output.bundle) != 0;
    FLAC__stream_encoder_delete(encoder);
    return failed;
}

static int run_once(
    const Pcm *pcm,
    size_t corpus_frames,
    unsigned sample_rate,
    unsigned frame_length,
    unsigned level,
    unsigned iterations,
    double *timings,
    uint64_t *encoded_bytes
) {
    Output output = {0};
    FLAC__StreamEncoder *encoder =
        new_encoder(sample_rate, frame_length, level, &output);
    if (!encoder)
        return 1;
    const size_t samples_per_frame = (size_t)frame_length * CHANNELS;
    for (unsigned iteration = 0; iteration < WARMUP; iteration++) {
        const size_t frame = iteration % corpus_frames;
        if (!FLAC__stream_encoder_process_interleaved(
                encoder, &pcm->samples[frame * samples_per_frame], frame_length)) {
            FLAC__stream_encoder_delete(encoder);
            return 1;
        }
    }
    output.bytes = 0;
    for (unsigned iteration = 0; iteration < iterations; iteration++) {
        const size_t frame = iteration % corpus_frames;
        const uint64_t started = benchmark_now_ticks();
        const int ok = FLAC__stream_encoder_process_interleaved(
            encoder, &pcm->samples[frame * samples_per_frame], frame_length);
        timings[iteration] = benchmark_elapsed_us(started, benchmark_now_ticks());
        if (!ok) {
            FLAC__stream_encoder_delete(encoder);
            return 1;
        }
    }
    *encoded_bytes += output.bytes;
    const int failed = !FLAC__stream_encoder_finish(encoder);
    FLAC__stream_encoder_delete(encoder);
    return failed;
}

int main(int argc, char **argv) {
    if (argc < 5 || argc > 7) {
        fprintf(stderr,
                "usage: libflac_frame_bench RATE LEVEL ITERATIONS RUNS "
                "[PCM_S32LE [PACKET_BUNDLE]]\n");
        return 2;
    }
    const unsigned sample_rate = (unsigned)atoi(argv[1]);
    const unsigned level = (unsigned)atoi(argv[2]);
    const unsigned iterations = (unsigned)atoi(argv[3]);
    const unsigned runs = (unsigned)atoi(argv[4]);
    if ((sample_rate != 48000 && sample_rate != 96000) || level > 8 ||
        iterations == 0 || runs == 0)
        return 2;

    const unsigned frame_length = sample_rate / 200;
    const size_t samples_per_frame = (size_t)frame_length * CHANNELS;
    Pcm pcm = argc >= 6 ? read_pcm(argv[5])
                        : synthetic_pcm(sample_rate, frame_length);
    if (!pcm.samples || pcm.sample_count % samples_per_frame != 0) {
        fprintf(stderr, "PCM must contain complete stereo 5 ms S32LE frames\n");
        free(pcm.samples);
        return 1;
    }
    const size_t corpus_frames = pcm.sample_count / samples_per_frame;
    if (argc == 7 && generate_bundle(argv[6], &pcm, corpus_frames, sample_rate,
                                     frame_length, level)) {
        fprintf(stderr, "could not generate libFLAC packet bundle\n");
        free(pcm.samples);
        return 1;
    }

    const size_t timing_count = (size_t)iterations * runs;
    double *timings = malloc(timing_count * sizeof(*timings));
    if (!timings) {
        free(pcm.samples);
        return 1;
    }
    uint64_t encoded_bytes = 0;
    for (unsigned run = 0; run < runs; run++) {
        if (run_once(&pcm, corpus_frames, sample_rate, frame_length, level,
                     iterations, &timings[(size_t)run * iterations],
                     &encoded_bytes)) {
            fprintf(stderr, "libFLAC frame encode failed\n");
            free(timings);
            free(pcm.samples);
            return 1;
        }
    }
    qsort(timings, timing_count, sizeof(*timings), compare_double);
    const double pcm_bytes =
        (double)timing_count * frame_length * CHANNELS * (BITS_PER_SAMPLE / 8);
    printf(
        "libflac encode rate=%u frame=%u level=%u corpus_frames=%zu "
        "p50_us=%.3f p95_us=%.3f p99_us=%.3f min_us=%.3f "
        "encoded/pcm=%.4f calls=%zu runs=%u\n",
        sample_rate, frame_length, level, corpus_frames,
        percentile(timings, timing_count, 50),
        percentile(timings, timing_count, 95),
        percentile(timings, timing_count, 99), timings[0],
        (double)encoded_bytes / pcm_bytes, timing_count, runs
    );
    free(timings);
    free(pcm.samples);
    return 0;
}
