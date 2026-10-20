/* libopus side of the whole-buffer wasm A/B harness.
 * The ABI matches examples/wasm_ab.rs. */
#include <emscripten.h>
#include <opus.h>
#include <stdint.h>
#include <stdlib.h>

#define CHANNELS 2
#define MAX_PACKET 1275

EMSCRIPTEN_KEEPALIVE void *ab_alloc(uint32_t bytes) { return calloc(bytes, 1); }

EMSCRIPTEN_KEEPALIVE void ab_free(void *ptr, uint32_t bytes) {
    (void)bytes;
    free(ptr);
}

EMSCRIPTEN_KEEPALIVE int32_t ab_encode(const int32_t *pcm, uint32_t samples,
                                       uint32_t frame_size, uint32_t bitrate,
                                       uint32_t vbr, uint32_t pcm_bits,
                                       uint8_t *out, uint32_t cap) {
    int error = OPUS_OK;
    OpusEncoder *enc =
        opus_encoder_create(48000, CHANNELS, OPUS_APPLICATION_AUDIO, &error);
    if (enc == NULL || error != OPUS_OK) return -1;
    opus_encoder_ctl(enc, OPUS_SET_BITRATE((opus_int32)bitrate));
    opus_encoder_ctl(enc, OPUS_SET_VBR(vbr != 0));
    opus_encoder_ctl(enc, OPUS_SET_VBR_CONSTRAINT(vbr == 1));
    opus_encoder_ctl(enc, OPUS_SET_BANDWIDTH(OPUS_BANDWIDTH_FULLBAND));
    opus_encoder_ctl(enc, OPUS_SET_SIGNAL(OPUS_SIGNAL_MUSIC));
    opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY(10));
    opus_int16 *pcm16 = malloc(sizeof(opus_int16) * frame_size * CHANNELS);
    uint32_t frames = samples / frame_size;
    uint32_t pos = 0;
    int32_t result = 0;
    for (uint32_t f = 0; f < frames; f++) {
        const int32_t *in = pcm + (size_t)f * frame_size * CHANNELS;
        int n;
        if (pos + 2 + MAX_PACKET > cap) {
            result = -3;
            break;
        }
        if (pcm_bits == 16) {
            for (uint32_t i = 0; i < frame_size * CHANNELS; i++) {
                pcm16[i] = (opus_int16)(in[i] >> 8);
            }
            n = opus_encode(enc, pcm16, (int)frame_size, out + pos + 2, MAX_PACKET);
        } else {
            n = opus_encode24(enc, in, (int)frame_size, out + pos + 2, MAX_PACKET);
        }
        if (n < 0) {
            result = -2;
            break;
        }
        out[pos] = (uint8_t)(n & 0xff);
        out[pos + 1] = (uint8_t)(n >> 8);
        pos += 2 + (uint32_t)n;
    }
    free(pcm16);
    opus_encoder_destroy(enc);
    return result < 0 ? result : (int32_t)pos;
}

EMSCRIPTEN_KEEPALIVE int32_t ab_decode(const uint8_t *packets, uint32_t len,
                                       uint32_t frame_size, uint32_t pcm_bits,
                                       int32_t *out, uint32_t out_samples) {
    int error = OPUS_OK;
    OpusDecoder *dec = opus_decoder_create(48000, CHANNELS, &error);
    if (dec == NULL || error != OPUS_OK) return -1;
    opus_int16 *pcm16 = malloc(sizeof(opus_int16) * frame_size * CHANNELS);
    uint32_t pos = 0;
    uint32_t total = 0;
    int32_t result = 0;
    while (pos + 2 <= len) {
        uint32_t n = (uint32_t)packets[pos] | ((uint32_t)packets[pos + 1] << 8);
        int r;
        pos += 2;
        if (pos + n > len || total + frame_size > out_samples) {
            result = -3;
            break;
        }
        if (pcm_bits == 16) {
            r = opus_decode(dec, packets + pos, (opus_int32)n, pcm16, (int)frame_size, 0);
            if (r > 0) {
                int32_t *dst = out + (size_t)total * CHANNELS;
                for (int i = 0; i < r * CHANNELS; i++) dst[i] = (int32_t)pcm16[i] * 256;
            }
        } else {
            r = opus_decode24(dec, packets + pos, (opus_int32)n,
                              out + (size_t)total * CHANNELS, (int)frame_size, 0);
        }
        if (r < 0) {
            result = -4;
            break;
        }
        pos += n;
        total += (uint32_t)r;
    }
    free(pcm16);
    opus_decoder_destroy(dec);
    return result < 0 ? result : (int32_t)total;
}
