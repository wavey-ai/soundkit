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
            "usage: raw_pcm_roundtrip_c [--pcm-bits 16|24] <frame-size> "
            "<bitrate> <cbr|vbr> <input.raw> <output.f32le>\n");
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

static void *read_samples(
    const char *path,
    size_t sample_size,
    size_t *sample_count,
    const char *format
) {
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
    if (byte_count < 0 || byte_count % (long)sample_size != 0) {
        fprintf(stderr, "invalid %s input length\n", format);
        exit(1);
    }
    rewind(file);
    *sample_count = (size_t)byte_count / sample_size;
    void *samples = malloc(*sample_count * sample_size);
    if (!samples) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    if (fread(samples, sample_size, *sample_count, file) != *sample_count) {
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
    int pcm_bits = 0;
    int offset = 1;
    if (argc >= 3 && strcmp(argv[1], "--pcm-bits") == 0) {
        pcm_bits = parse_positive_int(argv[2]);
        if (pcm_bits != 16 && pcm_bits != 24) {
            usage();
        }
        offset = 3;
    }
    if (argc != offset + 5) {
        usage();
    }
    int frame_size = parse_positive_int(argv[offset]);
    int bitrate = parse_positive_int(argv[offset + 1]);
    if (!valid_frame_size(frame_size)) {
        usage();
    }
    int vbr;
    if (strcmp(argv[offset + 2], "cbr") == 0) {
        vbr = 0;
    } else if (strcmp(argv[offset + 2], "vbr") == 0) {
        vbr = 1;
    } else {
        usage();
    }

    size_t sample_count = 0;
    float *input = NULL;
    opus_int16 *input_i16 = NULL;
    opus_int32 *input_i24 = NULL;
    if (pcm_bits == 16) {
        input_i16 = (opus_int16 *)read_samples(
            argv[offset + 3], sizeof(*input_i16), &sample_count, "s16le");
    } else if (pcm_bits == 24) {
        input_i24 = (opus_int32 *)read_samples(
            argv[offset + 3], sizeof(*input_i24), &sample_count, "s32le");
        for (size_t sample = 0; sample < sample_count; sample++) {
            if (input_i24[sample] < -8388608 || input_i24[sample] > 8388607) {
                fprintf(stderr, "input contains a sample outside signed 24-bit range\n");
                return 1;
            }
        }
    } else {
        input = (float *)read_samples(
            argv[offset + 3], sizeof(*input), &sample_count, "f32le");
    }
    if (sample_count % CHANNELS != 0) {
        fprintf(stderr, "input has a partial stereo frame\n");
        return 1;
    }
    float *output = (float *)calloc(sample_count + (size_t)frame_size * CHANNELS,
                                    sizeof(float));
    float *padded = pcm_bits == 0
        ? (float *)calloc((size_t)frame_size * CHANNELS, sizeof(float))
        : NULL;
    float *decoded = pcm_bits == 0
        ? (float *)malloc((size_t)frame_size * CHANNELS * sizeof(float))
        : NULL;
    opus_int16 *padded_i16 = pcm_bits == 16
        ? (opus_int16 *)calloc((size_t)frame_size * CHANNELS, sizeof(*padded_i16))
        : NULL;
    opus_int16 *decoded_i16 = pcm_bits == 16
        ? (opus_int16 *)malloc((size_t)frame_size * CHANNELS * sizeof(*decoded_i16))
        : NULL;
    opus_int32 *padded_i24 = pcm_bits == 24
        ? (opus_int32 *)calloc((size_t)frame_size * CHANNELS, sizeof(*padded_i24))
        : NULL;
    opus_int32 *decoded_i24 = pcm_bits == 24
        ? (opus_int32 *)malloc((size_t)frame_size * CHANNELS * sizeof(*decoded_i24))
        : NULL;
    unsigned char packet[MAX_PACKET_BYTES];
    if (!output || (pcm_bits == 0 && (!padded || !decoded)) ||
        (pcm_bits == 16 && (!padded_i16 || !decoded_i16)) ||
        (pcm_bits == 24 && (!padded_i24 || !decoded_i24))) {
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
    check_opus(opus_encoder_ctl(encoder, OPUS_SET_COMPLEXITY(10)),
               "OPUS_SET_COMPLEXITY");
    check_opus(opus_encoder_ctl(encoder, OPUS_SET_BANDWIDTH(OPUS_BANDWIDTH_FULLBAND)),
               "OPUS_SET_BANDWIDTH");
    check_opus(opus_encoder_ctl(encoder, OPUS_SET_SIGNAL(OPUS_SIGNAL_MUSIC)),
               "OPUS_SET_SIGNAL");

    size_t frame_samples = (size_t)frame_size * CHANNELS;
    size_t output_offset = 0;
    size_t packet_bytes = 0;
    size_t packet_count = 0;
    int packet_min = MAX_PACKET_BYTES;
    int packet_max = 0;
    for (size_t offset = 0; offset < sample_count; offset += frame_samples) {
        size_t remaining = sample_count - offset;
        size_t copied = remaining < frame_samples ? remaining : frame_samples;
        int packet_len;
        if (pcm_bits == 16) {
            memset(padded_i16, 0, frame_samples * sizeof(*padded_i16));
            memcpy(padded_i16, input_i16 + offset, copied * sizeof(*padded_i16));
            packet_len = opus_encode(
                encoder, padded_i16, frame_size, packet, MAX_PACKET_BYTES);
        } else if (pcm_bits == 24) {
            memset(padded_i24, 0, frame_samples * sizeof(*padded_i24));
            memcpy(padded_i24, input_i24 + offset, copied * sizeof(*padded_i24));
            packet_len = (int)opus_encode24(
                encoder, padded_i24, frame_size, packet, MAX_PACKET_BYTES);
        } else {
            memset(padded, 0, frame_samples * sizeof(*padded));
            memcpy(padded, input + offset, copied * sizeof(*padded));
            packet_len = opus_encode_float(
                encoder, padded, frame_size, packet, MAX_PACKET_BYTES);
        }
        if (packet_len < 0) {
            fprintf(stderr, "opus_encode_float failed: %s\n", opus_strerror(packet_len));
            return 1;
        }
        int decoded_frames;
        if (pcm_bits == 16) {
            decoded_frames = opus_decode(
                decoder, packet, packet_len, decoded_i16, frame_size, 0);
        } else if (pcm_bits == 24) {
            decoded_frames = opus_decode24(
                decoder, packet, packet_len, decoded_i24, frame_size, 0);
        } else {
            decoded_frames = opus_decode_float(
                decoder, packet, packet_len, decoded, frame_size, 0);
        }
        if (decoded_frames != frame_size) {
            fprintf(stderr, "opus_decode_float returned %d frames\n", decoded_frames);
            return 1;
        }
        if (pcm_bits == 16) {
            for (size_t sample = 0; sample < frame_samples; sample++) {
                output[output_offset + sample] =
                    (float)decoded_i16[sample] * (1.0f / 32768.0f);
            }
        } else if (pcm_bits == 24) {
            for (size_t sample = 0; sample < frame_samples; sample++) {
                output[output_offset + sample] =
                    (float)decoded_i24[sample] * (1.0f / 8388608.0f);
            }
        } else {
            memcpy(output + output_offset, decoded, frame_samples * sizeof(float));
        }
        output_offset += frame_samples;
        packet_bytes += (size_t)packet_len;
        packet_count++;
        if (packet_len < packet_min) packet_min = packet_len;
        if (packet_len > packet_max) packet_max = packet_len;
    }
    write_samples(argv[offset + 4], output, sample_count);

    const char *pcm_label = pcm_bits == 16 ? "16" : pcm_bits == 24 ? "24" : "float";
    printf("{\"codec\":\"%s\",\"sample_rate\":%d,\"channels\":%d,\"pcm_bits\":\"%s\","
           "\"frame_size\":%d,\"bitrate\":%d,\"mode\":\"%s\","
           "\"packets\":%zu,\"packet_bytes\":%zu,\"packet_min\":%d,"
           "\"packet_max\":%d}\n",
           opus_get_version_string(), SAMPLE_RATE, CHANNELS, pcm_label, frame_size, bitrate,
           vbr ? "vbr" : "cbr", packet_count, packet_bytes, packet_min, packet_max);

    opus_encoder_destroy(encoder);
    opus_decoder_destroy(decoder);
    free(input);
    free(input_i16);
    free(input_i24);
    free(output);
    free(padded);
    free(decoded);
    free(padded_i16);
    free(decoded_i16);
    free(padded_i24);
    free(decoded_i24);
    return 0;
}
