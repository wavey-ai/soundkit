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
        fprintf(stderr,
                "usage: compare_s16le <reference.s16le> <candidate.s16le>\n");
        return 2;
    }
    FILE *reference = open_input(argv[1]);
    FILE *candidate = open_input(argv[2]);
    int16_t expected[CHUNK_SAMPLES];
    int16_t actual[CHUNK_SAMPLES];
    uint64_t samples = 0;
    uint64_t mismatches = 0;
    long double signal_squared = 0.0;
    long double error_squared = 0.0;
    int maximum_error = 0;

    for (;;) {
        size_t expected_count =
            fread(expected, sizeof(*expected), CHUNK_SAMPLES, reference);
        size_t actual_count =
            fread(actual, sizeof(*actual), CHUNK_SAMPLES, candidate);
        if (expected_count != actual_count) {
            fprintf(stderr, "sample-count mismatch: %zu versus %zu\n",
                    expected_count, actual_count);
            return 1;
        }
        if (expected_count == 0) break;
        for (size_t index = 0; index < expected_count; ++index) {
            int error = (int)actual[index] - (int)expected[index];
            int absolute_error = error < 0 ? -error : error;
            signal_squared +=
                (long double)expected[index] * (long double)expected[index];
            error_squared += (long double)error * (long double)error;
            mismatches += error != 0;
            if (absolute_error > maximum_error) maximum_error = absolute_error;
        }
        samples += expected_count;
    }
    if (ferror(reference) || ferror(candidate)) return 1;
    fclose(reference);
    fclose(candidate);

    double snr_db = error_squared == 0.0
                        ? INFINITY
                        : 10.0 * log10((double)(signal_squared / error_squared));
    double rms_error = sqrt((double)(error_squared / samples));
    printf("samples=%" PRIu64 " mismatches=%" PRIu64
           " signal_squared=%.17Lg error_squared=%.17Lg rms_error=%.9f "
           "max_error=%d snr_db=%.9f\n",
           samples, mismatches, signal_squared, error_squared, rms_error,
           maximum_error, snr_db);
    return 0;
}
