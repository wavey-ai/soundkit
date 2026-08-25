#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vorbis/vorbisfile.h>

typedef struct {
    const uint8_t *data;
    size_t size;
    size_t position;
} memory_source;

static uint64_t monotonic_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        perror("clock_gettime");
        exit(1);
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) +
           (uint64_t)value.tv_nsec;
}

static inline void black_box_pcm(const int16_t *pcm) {
    __asm__ __volatile__("" : : "r"(pcm) : "memory");
}

static size_t memory_read(void *ptr, size_t size, size_t count, void *opaque) {
    memory_source *source = opaque;
    if (size == 0) return 0;
    size_t available = (source->size - source->position) / size;
    size_t elements = count < available ? count : available;
    size_t bytes = elements * size;
    memcpy(ptr, source->data + source->position, bytes);
    source->position += bytes;
    return elements;
}

static int memory_seek(void *opaque, ogg_int64_t offset, int origin) {
    memory_source *source = opaque;
    ogg_int64_t base;
    if (origin == SEEK_SET) {
        base = 0;
    } else if (origin == SEEK_CUR) {
        base = (ogg_int64_t)source->position;
    } else if (origin == SEEK_END) {
        base = (ogg_int64_t)source->size;
    } else {
        return -1;
    }
    ogg_int64_t position = base + offset;
    if (position < 0 || (uint64_t)position > source->size) return -1;
    source->position = (size_t)position;
    return 0;
}

static int memory_close(void *opaque) {
    (void)opaque;
    return 0;
}

static long memory_tell(void *opaque) {
    memory_source *source = opaque;
    return (long)source->position;
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
    if (data == NULL || fread(data, 1, (size_t)size, file) != (size_t)size) {
        exit(1);
    }
    fclose(file);
    *size_out = (size_t)size;
    return data;
}

static size_t decode_once(const uint8_t *input, size_t input_size, FILE *sink,
                          long *sample_rate, int *channels,
                          int64_t *checksum) {
    memory_source source = {input, input_size, 0};
    ov_callbacks callbacks = {
        memory_read,
        memory_seek,
        memory_close,
        memory_tell,
    };
    OggVorbis_File decoder;
    if (ov_open_callbacks(&source, &decoder, NULL, 0, callbacks) != 0) {
        fprintf(stderr, "ov_open_callbacks failed\n");
        exit(1);
    }
    vorbis_info *info = ov_info(&decoder, -1);
    if (info == NULL) exit(1);
    *sample_rate = info->rate;
    *channels = info->channels;

    int16_t pcm[8192];
    size_t total_samples = 0;
    int64_t sum = 0;
    for (;;) {
        int section = 0;
        long bytes = ov_read(&decoder, (char *)pcm, sizeof(pcm), 0, 2, 1,
                             &section);
        if (bytes == 0) break;
        if (bytes < 0) {
            fprintf(stderr, "ov_read failed: %ld\n", bytes);
            exit(1);
        }
        size_t samples = (size_t)bytes / sizeof(*pcm);
        black_box_pcm(pcm);
        if (sink != NULL &&
            fwrite(pcm, sizeof(*pcm), samples, sink) != samples) {
            exit(1);
        }
        if (checksum != NULL) {
            for (size_t index = 0; index < samples; ++index) sum += pcm[index];
        }
        total_samples += samples;
    }
    ov_clear(&decoder);
    if (checksum != NULL) *checksum = sum;
    return total_samples;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr,
                "usage: libvorbis_decode_bench <input.ogg> [iterations] "
                "[output.s16le]\n");
        return 2;
    }
    int iterations = argc >= 3 ? atoi(argv[2]) : 50;
    if (iterations <= 0) return 2;

    size_t input_size = 0;
    uint8_t *input = read_file(argv[1], &input_size);
    long sample_rate = 0;
    int channels = 0;
    int64_t checksum = 0;

    for (int index = 0; index < 3; ++index) {
        decode_once(input, input_size, NULL, &sample_rate, &channels, NULL);
    }

    uint64_t started = monotonic_ns();
    size_t total_samples = 0;
    for (int index = 0; index < iterations; ++index) {
        total_samples += decode_once(input, input_size, NULL, &sample_rate,
                                     &channels, NULL);
    }
    uint64_t elapsed = monotonic_ns() - started;
    decode_once(input, input_size, NULL, &sample_rate, &channels, &checksum);

    if (argc == 4) {
        FILE *output = fopen(argv[3], "wb");
        if (output == NULL) return 1;
        decode_once(input, input_size, output, &sample_rate, &channels,
                    &checksum);
        fclose(output);
    }

    printf("implementation=libvorbis-c codec=vorbis operation=decode "
           "input_bytes=%zu iterations=%d samples=%zu sample_rate=%ld "
           "channels=%d elapsed_ns=%" PRIu64 " checksum=%" PRId64 "\n",
           input_size, iterations, total_samples, sample_rate, channels,
           elapsed, checksum);
    free(input);
    return 0;
}
