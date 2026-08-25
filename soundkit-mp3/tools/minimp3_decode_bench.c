#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MINIMP3_FLOAT_OUTPUT
#define MINIMP3_ONLY_MP3
#define MINIMP3_IMPLEMENTATION
#include "minimp3.h"

static uint64_t monotonic_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        perror("clock_gettime");
        exit(1);
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
}

static inline void black_box_pcm(const mp3d_sample_t *pcm) {
    __asm__ __volatile__("" : : "r"(pcm) : "memory");
}

static uint8_t *read_file(const char *path, size_t *size_out) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(file, 0, SEEK_END) != 0) exit(1);
    long size = ftell(file);
    if (size <= 0 || fseek(file, 0, SEEK_SET) != 0) exit(1);
    uint8_t *data = malloc((size_t)size);
    if (data == NULL || fread(data, 1, (size_t)size, file) != (size_t)size) exit(1);
    fclose(file);
    *size_out = (size_t)size;
    return data;
}

static size_t decode_once(const uint8_t *input, size_t input_size, FILE *sink,
                          int *sample_rate, int *channels, double *checksum) {
    mp3dec_t decoder;
    mp3dec_init(&decoder);
    mp3d_sample_t pcm[MINIMP3_MAX_SAMPLES_PER_FRAME];
    size_t offset = 0;
    size_t total_samples = 0;
    double sum = 0.0;

    while (offset < input_size) {
        mp3dec_frame_info_t info;
        int samples_per_channel = mp3dec_decode_frame(
            &decoder, input + offset, (int)(input_size - offset), pcm, &info);
        if (info.frame_bytes <= 0) break;
        offset += (size_t)info.frame_bytes;
        if (samples_per_channel <= 0) continue;

        size_t samples = (size_t)samples_per_channel * (size_t)info.channels;
        black_box_pcm(pcm);
        if (sink != NULL && fwrite(pcm, sizeof(*pcm), samples, sink) != samples) exit(1);
        if (checksum != NULL) {
            for (size_t index = 0; index < samples; ++index) sum += pcm[index];
        }
        total_samples += samples;
        *sample_rate = info.hz;
        *channels = info.channels;
    }

    if (checksum != NULL) *checksum = sum;
    return total_samples;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: minimp3_decode_bench <input.mp3> [iterations] [output.f32le]\n");
        return 2;
    }
    int iterations = argc >= 3 ? atoi(argv[2]) : 50;
    if (iterations <= 0) return 2;

    size_t input_size = 0;
    uint8_t *input = read_file(argv[1], &input_size);
    int sample_rate = 0;
    int channels = 0;
    double checksum = 0.0;

    for (int index = 0; index < 3; ++index) {
        decode_once(input, input_size, NULL, &sample_rate, &channels, NULL);
    }

    uint64_t started = monotonic_ns();
    size_t total_samples = 0;
    for (int index = 0; index < iterations; ++index) {
        total_samples +=
            decode_once(input, input_size, NULL, &sample_rate, &channels, NULL);
    }
    uint64_t elapsed = monotonic_ns() - started;

    decode_once(input, input_size, NULL, &sample_rate, &channels, &checksum);

    if (argc == 4) {
        FILE *output = fopen(argv[3], "wb");
        if (output == NULL) return 1;
        decode_once(input, input_size, output, &sample_rate, &channels, &checksum);
        fclose(output);
    }

    printf("implementation=minimp3-c codec=mp3 operation=decode input_bytes=%zu iterations=%d samples=%zu sample_rate=%d channels=%d elapsed_ns=%" PRIu64 " checksum=%.9f\n",
           input_size, iterations, total_samples, sample_rate, channels, elapsed, checksum);
    free(input);
    return 0;
}
