#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <opus.h>

#define CHANNELS 2
#define SAMPLE_RATE 48000
#define MAX_PACKET_BYTES 1275

static void usage(void) {
    fprintf(stderr,
            "usage: raw_pcm_roundtrip_c <frame-size> <bitrate> <cbr|vbr> "
            "<input.f32le> <output.f32le>\n");
    exit(2);
}

static int parse_positive_int(const char *value) {
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > INT32_MAX) {
        usage();
    }
    return (int)parsed;
}

static int valid_frame_size(int frame_size) {
    return frame_size == 120 || frame_size == 240 || frame_size == 480 ||
           frame_size == 960;
}

static float *read_samples(const char *path, size_t *sample_count) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        perror("fseek");
        exit(1);
    }
    long byte_count = ftell(file);
    if (byte_count < 0 || byte_count % (long)sizeof(float) != 0) {
        fprintf(stderr, "invalid f32le input length\n");
        exit(1);
    }
    rewind(file);
    *sample_count = (size_t)byte_count / sizeof(float);
    float *samples = (float *)malloc(*sample_count * sizeof(float));
    if (!samples) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    if (fread(samples, sizeof(float), *sample_count, file) != *sample_count) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(1);
    }
    if (fclose(file) != 0) {
        perror("fclose");
        exit(1);
    }
    return samples;
}

static void write_samples(const char *path, const float *samples, size_t sample_count) {
    FILE *file = fopen(path, "wb");
    if (!file) {
        fprintf(stderr, "cannot create %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fwrite(samples, sizeof(float), sample_count, file) != sample_count) {
        fprintf(stderr, "cannot write %s\n", path);
        exit(1);
    }
    if (fclose(file) != 0) {
        perror("fclose");
        exit(1);
    }
}

static void check_opus(int result, const char *operation) {
    if (result != OPUS_OK) {
        fprintf(stderr, "%s failed: %s\n", operation, opus_strerror(result));
        exit(1);
    }
}

int main(int argc, char **argv) {
    if (argc != 6) {
        usage();
    }
    int frame_size = parse_positive_int(argv[1]);
    int bitrate = parse_positive_int(argv[2]);
    if (!valid_frame_size(frame_size)) {
        usage();
    }
    int vbr;
    if (strcmp(argv[3], "cbr") == 0) {
        vbr = 0;
    } else if (strcmp(argv[3], "vbr") == 0) {
        vbr = 1;
    } else {
        usage();
    }

    size_t sample_count = 0;
    float *input = read_samples(argv[4], &sample_count);
    if (sample_count % CHANNELS != 0) {
        fprintf(stderr, "input has a partial stereo frame\n");
        return 1;
    }
    float *output = (float *)calloc(sample_count + (size_t)frame_size * CHANNELS,
                                    sizeof(float));
    float *padded = (float *)calloc((size_t)frame_size * CHANNELS, sizeof(float));
    float *decoded = (float *)malloc((size_t)frame_size * CHANNELS * sizeof(float));
    unsigned char packet[MAX_PACKET_BYTES];
    if (!output || !padded || !decoded) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    int error = OPUS_OK;
    OpusEncoder *encoder = opus_encoder_create(
        SAMPLE_RATE, CHANNELS, OPUS_APPLICATION_AUDIO, &error);
    check_opus(error, "opus_encoder_create");
    OpusDecoder *decoder = opus_decoder_create(SAMPLE_RATE, CHANNELS, &error);
    check_opus(error, "opus_decoder_create");
    check_opus(opus_encoder_ctl(encoder, OPUS_SET_BITRATE(bitrate)),
               "OPUS_SET_BITRATE");
    check_opus(opus_encoder_ctl(encoder, OPUS_SET_VBR(vbr)), "OPUS_SET_VBR");
    check_opus(opus_encoder_ctl(encoder, OPUS_SET_VBR_CONSTRAINT(vbr)),
               "OPUS_SET_VBR_CONSTRAINT");
    check_opus(opus_encoder_ctl(encoder, OPUS_SET_BANDWIDTH(OPUS_BANDWIDTH_FULLBAND)),
               "OPUS_SET_BANDWIDTH");

    size_t frame_samples = (size_t)frame_size * CHANNELS;
    size_t output_offset = 0;
    size_t packet_bytes = 0;
    size_t packet_count = 0;
    int packet_min = MAX_PACKET_BYTES;
    int packet_max = 0;
    for (size_t offset = 0; offset < sample_count; offset += frame_samples) {
        size_t remaining = sample_count - offset;
        size_t copied = remaining < frame_samples ? remaining : frame_samples;
        memset(padded, 0, frame_samples * sizeof(float));
        memcpy(padded, input + offset, copied * sizeof(float));
        int packet_len = opus_encode_float(
            encoder, padded, frame_size, packet, MAX_PACKET_BYTES);
        if (packet_len < 0) {
            fprintf(stderr, "opus_encode_float failed: %s\n", opus_strerror(packet_len));
            return 1;
        }
        int decoded_frames = opus_decode_float(
            decoder, packet, packet_len, decoded, frame_size, 0);
        if (decoded_frames != frame_size) {
            fprintf(stderr, "opus_decode_float returned %d frames\n", decoded_frames);
            return 1;
        }
        memcpy(output + output_offset, decoded, frame_samples * sizeof(float));
        output_offset += frame_samples;
        packet_bytes += (size_t)packet_len;
        packet_count++;
        if (packet_len < packet_min) packet_min = packet_len;
        if (packet_len > packet_max) packet_max = packet_len;
    }
    write_samples(argv[5], output, sample_count);

    printf("{\"codec\":\"%s\",\"sample_rate\":%d,\"channels\":%d,"
           "\"frame_size\":%d,\"bitrate\":%d,\"mode\":\"%s\","
           "\"packets\":%zu,\"packet_bytes\":%zu,\"packet_min\":%d,"
           "\"packet_max\":%d}\n",
           opus_get_version_string(), SAMPLE_RATE, CHANNELS, frame_size, bitrate,
           vbr ? "vbr" : "cbr", packet_count, packet_bytes, packet_min, packet_max);

    opus_encoder_destroy(encoder);
    opus_decoder_destroy(decoder);
    free(input);
    free(output);
    free(padded);
    free(decoded);
    return 0;
}
