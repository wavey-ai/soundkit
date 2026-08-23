// Encode an S32LE frame sequence with libFLAC and write length-prefixed raw
// FLAC packets for differential testing.
//
// Build:
//   cc -O2 scripts/libflac_packet_fixture.c \
//      $(pkg-config --cflags --libs flac) -o libflac_packet_fixture
// Usage:
//   libflac_packet_fixture PCM_S32LE BUNDLE RATE CHANNELS 16|24 LEVEL

#include <FLAC/stream_encoder.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern FLAC__bool FLAC__stream_encoder_set_do_md5(
    FLAC__StreamEncoder *encoder,
    FLAC__bool value
);

typedef struct {
    FILE *file;
    size_t frames;
} Output;

static int write_u32le(FILE *file, uint32_t value) {
    const uint8_t bytes[4] = {
        (uint8_t)value,
        (uint8_t)(value >> 8),
        (uint8_t)(value >> 16),
        (uint8_t)(value >> 24),
    };
    return fwrite(bytes, 1, 4, file) == 4 ? 0 : 1;
}

static FLAC__StreamEncoderWriteStatus write_packet(
    const FLAC__StreamEncoder *encoder,
    const FLAC__byte buffer[],
    size_t bytes,
    unsigned samples,
    unsigned current_frame,
    void *client_data
) {
    (void)encoder;
    (void)current_frame;
    if (samples == 0)
        return FLAC__STREAM_ENCODER_WRITE_STATUS_OK;
    Output *output = client_data;
    if (bytes > UINT32_MAX || write_u32le(output->file, (uint32_t)bytes) ||
        fwrite(buffer, 1, bytes, output->file) != bytes)
        return FLAC__STREAM_ENCODER_WRITE_STATUS_FATAL_ERROR;
    output->frames++;
    return FLAC__STREAM_ENCODER_WRITE_STATUS_OK;
}

static FLAC__int32 *read_pcm(const char *path, size_t *sample_count) {
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) {
        if (file)
            fclose(file);
        return NULL;
    }
    const long length = ftell(file);
    if (length <= 0 || length % 4 || fseek(file, 0, SEEK_SET)) {
        fclose(file);
        return NULL;
    }
    uint8_t *bytes = malloc((size_t)length);
    FLAC__int32 *pcm = malloc((size_t)length);
    if (!bytes || !pcm || fread(bytes, 1, (size_t)length, file) != (size_t)length) {
        free(bytes);
        free(pcm);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *sample_count = (size_t)length / 4;
    for (size_t index = 0; index < *sample_count; index++) {
        const uint8_t *sample = &bytes[index * 4];
        pcm[index] = (FLAC__int32)(
            (uint32_t)sample[0] |
            (uint32_t)sample[1] << 8 |
            (uint32_t)sample[2] << 16 |
            (uint32_t)sample[3] << 24
        );
    }
    free(bytes);
    return pcm;
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr,
                "usage: libflac_packet_fixture PCM_S32LE BUNDLE RATE "
                "CHANNELS 16|24 LEVEL\n");
        return 2;
    }
    const unsigned sample_rate = (unsigned)atoi(argv[3]);
    const unsigned channels = (unsigned)atoi(argv[4]);
    const unsigned bits = (unsigned)atoi(argv[5]);
    const unsigned level = (unsigned)atoi(argv[6]);
    if ((sample_rate != 48000 && sample_rate != 96000) ||
        channels == 0 || channels > 8 || (bits != 16 && bits != 24) || level > 8)
        return 2;
    const unsigned frame_length = sample_rate / 200;
    const size_t samples_per_frame = (size_t)frame_length * channels;
    size_t sample_count = 0;
    FLAC__int32 *pcm = read_pcm(argv[1], &sample_count);
    if (!pcm || sample_count % samples_per_frame) {
        fprintf(stderr, "PCM does not contain complete 5 ms frames\n");
        free(pcm);
        return 1;
    }
    Output output = {0};
    output.file = fopen(argv[2], "wb");
    FLAC__StreamEncoder *encoder = FLAC__stream_encoder_new();
    const int configured = encoder && output.file &&
        FLAC__stream_encoder_set_compression_level(encoder, level) &&
        FLAC__stream_encoder_set_channels(encoder, channels) &&
        FLAC__stream_encoder_set_bits_per_sample(encoder, bits) &&
        FLAC__stream_encoder_set_sample_rate(encoder, sample_rate) &&
        FLAC__stream_encoder_set_blocksize(encoder, frame_length) &&
        FLAC__stream_encoder_set_do_md5(encoder, false) &&
        FLAC__stream_encoder_init_stream(
            encoder, write_packet, NULL, NULL, NULL, &output) ==
            FLAC__STREAM_ENCODER_INIT_STATUS_OK;
    if (!configured) {
        fprintf(stderr, "could not initialize libFLAC\n");
        if (output.file)
            fclose(output.file);
        if (encoder)
            FLAC__stream_encoder_delete(encoder);
        free(pcm);
        return 1;
    }
    const size_t frames = sample_count / samples_per_frame;
    int failed = 0;
    for (size_t frame = 0; frame < frames; frame++) {
        if (!FLAC__stream_encoder_process_interleaved(
                encoder, &pcm[frame * samples_per_frame], frame_length)) {
            failed = 1;
            break;
        }
    }
    failed |= !FLAC__stream_encoder_finish(encoder);
    failed |= fclose(output.file) != 0;
    FLAC__stream_encoder_delete(encoder);
    free(pcm);
    if (failed || output.frames != frames) {
        fprintf(stderr, "libFLAC packet generation failed\n");
        return 1;
    }
    printf("libflac fixture rate=%u channels=%u bits=%u level=%u frames=%zu\n",
           sample_rate, channels, bits, level, frames);
    return 0;
}
