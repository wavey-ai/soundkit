#include <emscripten.h>
#include <opus.h>
#include <stdint.h>
#include <stdlib.h>

#define MAX_PACKET_BYTES 1275

typedef struct {
    OpusEncoder *codec;
    int channels;
    int frame_size;
    opus_int16 *input_i16;
    opus_int32 *input_i24;
    unsigned char packet[MAX_PACKET_BYTES];
    int packet_len;
} ReferenceEncoder;

typedef struct {
    OpusDecoder *codec;
    int channels;
    int frame_size;
    unsigned char packet[MAX_PACKET_BYTES];
    opus_int16 *output_i16;
    opus_int32 *output_i24;
} ReferenceDecoder;

static int configure_encoder(OpusEncoder *codec, int bitrate, int vbr) {
    int error = opus_encoder_ctl(codec, OPUS_SET_BITRATE(bitrate));
    if (error != OPUS_OK) return error;
    error = opus_encoder_ctl(codec, OPUS_SET_VBR(vbr != 0));
    if (error != OPUS_OK) return error;
    error = opus_encoder_ctl(codec, OPUS_SET_VBR_CONSTRAINT(vbr != 0));
    if (error != OPUS_OK) return error;
    error = opus_encoder_ctl(codec, OPUS_SET_BANDWIDTH(OPUS_BANDWIDTH_FULLBAND));
    if (error != OPUS_OK) return error;
    return opus_encoder_ctl(codec, OPUS_SET_SIGNAL(OPUS_SIGNAL_MUSIC));
}

EMSCRIPTEN_KEEPALIVE ReferenceEncoder *reference_encoder_new(
    int channels,
    int sample_rate,
    int bitrate,
    int frame_size) {
    int error = OPUS_OK;
    if (channels <= 0 || frame_size <= 0) return NULL;
    ReferenceEncoder *encoder = calloc(1, sizeof(*encoder));
    if (encoder == NULL) return NULL;
    encoder->channels = channels;
    encoder->frame_size = frame_size;
    size_t samples = (size_t)channels * (size_t)frame_size;
    encoder->input_i16 = malloc(samples * sizeof(*encoder->input_i16));
    encoder->input_i24 = malloc(samples * sizeof(*encoder->input_i24));
    encoder->codec = opus_encoder_create(
        sample_rate,
        channels,
        OPUS_APPLICATION_AUDIO,
        &error);
    if (encoder->input_i16 == NULL || encoder->input_i24 == NULL ||
        encoder->codec == NULL || error != OPUS_OK ||
        configure_encoder(encoder->codec, bitrate, 0) != OPUS_OK) {
        if (encoder->codec != NULL) opus_encoder_destroy(encoder->codec);
        free(encoder->input_i16);
        free(encoder->input_i24);
        free(encoder);
        return NULL;
    }
    return encoder;
}

EMSCRIPTEN_KEEPALIVE void reference_encoder_destroy(ReferenceEncoder *encoder) {
    if (encoder == NULL) return;
    opus_encoder_destroy(encoder->codec);
    free(encoder->input_i16);
    free(encoder->input_i24);
    free(encoder);
}

EMSCRIPTEN_KEEPALIVE int reference_encoder_set_vbr(ReferenceEncoder *encoder, int vbr) {
    if (encoder == NULL) return OPUS_BAD_ARG;
    int error = opus_encoder_ctl(encoder->codec, OPUS_SET_VBR(vbr != 0));
    if (error != OPUS_OK) return error;
    return opus_encoder_ctl(encoder->codec, OPUS_SET_VBR_CONSTRAINT(vbr != 0));
}

EMSCRIPTEN_KEEPALIVE uintptr_t reference_encoder_input_i16_ptr(ReferenceEncoder *encoder) {
    return encoder == NULL ? 0 : (uintptr_t)encoder->input_i16;
}

EMSCRIPTEN_KEEPALIVE uintptr_t reference_encoder_input_i24_ptr(ReferenceEncoder *encoder) {
    return encoder == NULL ? 0 : (uintptr_t)encoder->input_i24;
}

EMSCRIPTEN_KEEPALIVE int reference_encoder_input_len(ReferenceEncoder *encoder) {
    return encoder == NULL ? 0 : encoder->channels * encoder->frame_size;
}

EMSCRIPTEN_KEEPALIVE uintptr_t reference_encoder_packet_ptr(ReferenceEncoder *encoder) {
    return encoder == NULL ? 0 : (uintptr_t)encoder->packet;
}

EMSCRIPTEN_KEEPALIVE int reference_encoder_packet_len(ReferenceEncoder *encoder) {
    return encoder == NULL ? 0 : encoder->packet_len;
}

EMSCRIPTEN_KEEPALIVE int reference_encoder_encode_i16(ReferenceEncoder *encoder) {
    if (encoder == NULL) return OPUS_BAD_ARG;
    encoder->packet_len = opus_encode(
        encoder->codec,
        encoder->input_i16,
        encoder->frame_size,
        encoder->packet,
        MAX_PACKET_BYTES);
    return encoder->packet_len;
}

EMSCRIPTEN_KEEPALIVE int reference_encoder_encode_i24(ReferenceEncoder *encoder) {
    if (encoder == NULL) return OPUS_BAD_ARG;
    encoder->packet_len = opus_encode24(
        encoder->codec,
        encoder->input_i24,
        encoder->frame_size,
        encoder->packet,
        MAX_PACKET_BYTES);
    return encoder->packet_len;
}

EMSCRIPTEN_KEEPALIVE ReferenceDecoder *reference_decoder_new(
    int channels,
    int sample_rate,
    int frame_size) {
    int error = OPUS_OK;
    if (channels <= 0 || frame_size <= 0) return NULL;
    ReferenceDecoder *decoder = calloc(1, sizeof(*decoder));
    if (decoder == NULL) return NULL;
    decoder->channels = channels;
    decoder->frame_size = frame_size;
    size_t samples = (size_t)channels * (size_t)frame_size;
    decoder->output_i16 = malloc(samples * sizeof(*decoder->output_i16));
    decoder->output_i24 = malloc(samples * sizeof(*decoder->output_i24));
    decoder->codec = opus_decoder_create(sample_rate, channels, &error);
    if (decoder->output_i16 == NULL || decoder->output_i24 == NULL ||
        decoder->codec == NULL || error != OPUS_OK) {
        if (decoder->codec != NULL) opus_decoder_destroy(decoder->codec);
        free(decoder->output_i16);
        free(decoder->output_i24);
        free(decoder);
        return NULL;
    }
    return decoder;
}

EMSCRIPTEN_KEEPALIVE void reference_decoder_destroy(ReferenceDecoder *decoder) {
    if (decoder == NULL) return;
    opus_decoder_destroy(decoder->codec);
    free(decoder->output_i16);
    free(decoder->output_i24);
    free(decoder);
}

EMSCRIPTEN_KEEPALIVE uintptr_t reference_decoder_packet_ptr(ReferenceDecoder *decoder) {
    return decoder == NULL ? 0 : (uintptr_t)decoder->packet;
}

EMSCRIPTEN_KEEPALIVE int reference_decoder_packet_capacity(ReferenceDecoder *decoder) {
    return decoder == NULL ? 0 : MAX_PACKET_BYTES;
}

EMSCRIPTEN_KEEPALIVE uintptr_t reference_decoder_output_i16_ptr(ReferenceDecoder *decoder) {
    return decoder == NULL ? 0 : (uintptr_t)decoder->output_i16;
}

EMSCRIPTEN_KEEPALIVE uintptr_t reference_decoder_output_i24_ptr(ReferenceDecoder *decoder) {
    return decoder == NULL ? 0 : (uintptr_t)decoder->output_i24;
}

EMSCRIPTEN_KEEPALIVE int reference_decoder_output_len(ReferenceDecoder *decoder) {
    return decoder == NULL ? 0 : decoder->channels * decoder->frame_size;
}

EMSCRIPTEN_KEEPALIVE int reference_decoder_decode_i16(
    ReferenceDecoder *decoder,
    int packet_len) {
    if (decoder == NULL || packet_len < 0 || packet_len > MAX_PACKET_BYTES) {
        return OPUS_BAD_ARG;
    }
    return opus_decode(
        decoder->codec,
        decoder->packet,
        packet_len,
        decoder->output_i16,
        decoder->frame_size,
        0);
}

EMSCRIPTEN_KEEPALIVE int reference_decoder_decode_i24(
    ReferenceDecoder *decoder,
    int packet_len) {
    if (decoder == NULL || packet_len < 0 || packet_len > MAX_PACKET_BYTES) {
        return OPUS_BAD_ARG;
    }
    return opus_decode24(
        decoder->codec,
        decoder->packet,
        packet_len,
        decoder->output_i24,
        decoder->frame_size,
        0);
}
