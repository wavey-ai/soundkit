#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CHUNK_SAMPLES = 16384 };

static FILE *open_input(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    return file;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: compare_f32le <reference.f32le> <candidate.f32le>\n");
        return 2;
    }

    FILE *reference = open_input(argv[1]);
    FILE *candidate = open_input(argv[2]);
    float reference_samples[CHUNK_SAMPLES];
    float candidate_samples[CHUNK_SAMPLES];
    uint64_t samples = 0;
    long double signal_squared = 0.0;
    long double error_squared = 0.0;
    double max_absolute_error = 0.0;

    for (;;) {
        size_t reference_count =
            fread(reference_samples, sizeof(*reference_samples), CHUNK_SAMPLES, reference);
        size_t candidate_count =
            fread(candidate_samples, sizeof(*candidate_samples), CHUNK_SAMPLES, candidate);
        if (reference_count != candidate_count) {
            fprintf(stderr, "sample-count mismatch: %zu versus %zu in final chunk\n",
                    reference_count, candidate_count);
            return 1;
        }
        if (reference_count == 0) break;

        for (size_t index = 0; index < reference_count; ++index) {
            double expected = reference_samples[index];
            double error = (double)candidate_samples[index] - expected;
            double absolute_error = fabs(error);
            signal_squared += (long double)expected * expected;
            error_squared += (long double)error * error;
            if (absolute_error > max_absolute_error) max_absolute_error = absolute_error;
        }
        samples += reference_count;
    }

    if (ferror(reference) || ferror(candidate)) {
        fprintf(stderr, "read failed\n");
        return 1;
    }
    fclose(reference);
    fclose(candidate);

    double snr_db = error_squared == 0.0
                        ? INFINITY
                        : 10.0 * log10((double)(signal_squared / error_squared));
    printf("samples=%" PRIu64
           " signal_squared=%.17Lg error_squared=%.17Lg max_abs=%.17g snr_db=%.9f\n",
           samples, signal_squared, error_squared, max_absolute_error, snr_db);
    return 0;
}
