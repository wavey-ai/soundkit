/*
 * libopus Opus Custom reference tool for the soundkit-opus interop harness.
 *
 * Build against a libopus tree configured with custom modes, for example:
 *   cmake -DOPUS_CUSTOM_MODES=ON ... && ninja opus
 *   cc -O2 -I$OPUS_SRC/include tools/opus_custom_interop.c $OPUS_BUILD/libopus.a -lm
 * Add -DOPUS_CUSTOM_INTERNALS -I$OPUS_SRC/celt -I$OPUS_SRC to enable the
 * `modeinfo` command, which prints the mode tables.
 *
 * Commands (the Rust example `opus_custom_interop` takes the same arguments):
 *   encode   --frame N --channels C [--bitrate B|max] [--vbr 0|1] [--cvbr 0|1]
 *            [--signalling 0|1] [--complexity N] [--max-bytes N]
 *            --pcm s16|s24|f32 IN.raw OUT.pkt
 *   decode   --frame N --channels C [--signalling 0|1] [--errors stop|skip]
 *            --pcm s16|s24|f32 IN.pkt OUT.raw
 *   modeinfo --frame N
 *
 * Raw PCM is interleaved little-endian: s16 is int16, s24 is int32 holding a
 * sign-extended 24-bit value, f32 is float. A trailing partial frame is
 * ignored. A packet file is a sequence of records: a little-endian uint16
 * length followed by that many bytes. A zero length marks a lost packet,
 * which the decoder conceals. With --errors skip, a packet the decoder rejects
 * writes no output, its index is printed to stderr, and decoding continues.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "opus_custom.h"
#include "opus_defines.h"

#ifdef OPUS_CUSTOM_INTERNALS
#include "modes.h"
#endif

#define CELT_SET_SIGNALLING_REQUEST 10016
#define MAX_PACKET 1275

typedef struct {
    int frame;
    int channels;
    int bitrate;
    int vbr;
    int cvbr;
    int signalling;
    int complexity;
    int max_bytes;
    int skip_errors;
    const char *schedule;
    const char *pcm;
    const char *in;
    const char *out;
} Args;

static void die(const char *message) {
    fprintf(stderr, "opus_custom_interop: %s\n", message);
    exit(1);
}

static void usage(void) {
    fprintf(stderr,
            "usage: opus_custom_interop encode|decode|modeinfo --frame N [--channels C] "
            "[--bitrate B|max] [--vbr 0|1] [--cvbr 0|1] [--signalling 0|1] "
            "[--complexity N] [--max-bytes N] [--errors stop|skip] [--pcm s16|s24|f32] "
            "[--rate-schedule FRAME:BITRATE,...] [IN OUT]\n");
    exit(2);
}

static Args parse(int argc, char **argv, int first) {
    Args a;
    int positional = 0;
    memset(&a, 0, sizeof(a));
    a.frame = 0;
    a.channels = 2;
    a.bitrate = OPUS_BITRATE_MAX;
    a.vbr = 0;
    a.cvbr = 1;
    a.signalling = 1;
    a.complexity = 10;
    a.max_bytes = MAX_PACKET;
    a.pcm = "s16";
    for (int i = first; i < argc; i++) {
        const char *k = argv[i];
        if (k[0] == '-' && k[1] == '-') {
            if (i + 1 >= argc) usage();
            const char *v = argv[++i];
            if (!strcmp(k, "--frame")) a.frame = atoi(v);
            else if (!strcmp(k, "--channels")) a.channels = atoi(v);
            else if (!strcmp(k, "--bitrate")) a.bitrate = strcmp(v, "max") ? atoi(v) : OPUS_BITRATE_MAX;
            else if (!strcmp(k, "--vbr")) a.vbr = atoi(v);
            else if (!strcmp(k, "--cvbr")) a.cvbr = atoi(v);
            else if (!strcmp(k, "--signalling")) a.signalling = atoi(v);
            else if (!strcmp(k, "--complexity")) a.complexity = atoi(v);
            else if (!strcmp(k, "--max-bytes")) a.max_bytes = atoi(v);
            else if (!strcmp(k, "--pcm")) a.pcm = v;
            else if (!strcmp(k, "--errors")) a.skip_errors = !strcmp(v, "skip");
            else if (!strcmp(k, "--rate-schedule")) a.schedule = v;
            else usage();
        } else if (positional == 0) {
            a.in = k;
            positional++;
        } else if (positional == 1) {
            a.out = k;
            positional++;
        } else {
            usage();
        }
    }
    if (a.frame <= 0) usage();
    if (strcmp(a.pcm, "s16") && strcmp(a.pcm, "s24") && strcmp(a.pcm, "f32")) usage();
    return a;
}

static unsigned char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open input");
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *data = malloc(size > 0 ? (size_t)size : 1);
    if (size > 0 && fread(data, 1, (size_t)size, f) != (size_t)size) die("short read");
    fclose(f);
    *len = (size_t)size;
    return data;
}

static OpusCustomMode *make_mode(int frame) {
    int err = 0;
    OpusCustomMode *mode = opus_custom_mode_create(48000, frame, &err);
    if (!mode || err != OPUS_OK) die("opus_custom_mode_create failed");
    return mode;
}

static int sample_bytes(const char *pcm) {
    return strcmp(pcm, "s16") == 0 ? 2 : 4;
}

static int encode(Args a) {
    size_t len = 0;
    unsigned char *raw = read_file(a.in, &len);
    int err = 0;
    OpusCustomMode *mode = make_mode(a.frame);
    OpusCustomEncoder *enc = opus_custom_encoder_create(mode, a.channels, &err);
    if (!enc || err != OPUS_OK) die("opus_custom_encoder_create failed");
    if (opus_custom_encoder_ctl(enc, OPUS_SET_COMPLEXITY(a.complexity)) != OPUS_OK) die("complexity");
    if (opus_custom_encoder_ctl(enc, OPUS_SET_BITRATE(a.bitrate)) != OPUS_OK) die("bitrate");
    if (opus_custom_encoder_ctl(enc, OPUS_SET_VBR(a.vbr)) != OPUS_OK) die("vbr");
    if (opus_custom_encoder_ctl(enc, OPUS_SET_VBR_CONSTRAINT(a.cvbr)) != OPUS_OK) die("cvbr");
    if (opus_custom_encoder_ctl(enc, CELT_SET_SIGNALLING_REQUEST, (opus_int32)a.signalling) != OPUS_OK)
        die("signalling");
    FILE *out = fopen(a.out, "wb");
    if (!out) die("cannot open output");
    size_t frame_bytes = (size_t)a.frame * a.channels * sample_bytes(a.pcm);
    size_t frames = len / frame_bytes;
    unsigned char packet[MAX_PACKET];
    const char *next = a.schedule;
    for (size_t f = 0; f < frames; f++) {
        const unsigned char *p = raw + f * frame_bytes;
        int n;
        /* OPUS_SET_BITRATE before the scheduled frame, as a live encoder
           would change rate mid-stream. */
        while (next && *next) {
            char *end;
            long at = strtol(next, &end, 10);
            if (*end != ':') die("bad --rate-schedule");
            if ((size_t)at != f) break;
            long rate = strtol(end + 1, &end, 10);
            if (opus_custom_encoder_ctl(enc, OPUS_SET_BITRATE((opus_int32)rate)) != OPUS_OK)
                die("scheduled bitrate");
            next = *end == ',' ? end + 1 : end;
        }
        if (!strcmp(a.pcm, "s16")) {
            n = opus_custom_encode(enc, (const opus_int16 *)p, a.frame, packet, a.max_bytes);
        } else if (!strcmp(a.pcm, "s24")) {
            n = opus_custom_encode24(enc, (const opus_int32 *)p, a.frame, packet, a.max_bytes);
        } else {
            n = opus_custom_encode_float(enc, (const float *)p, a.frame, packet, a.max_bytes);
        }
        if (n <= 0) {
            fprintf(stderr, "frame %zu: encode error %d\n", f, n);
            return 1;
        }
        unsigned char hdr[2] = {(unsigned char)(n & 0xff), (unsigned char)(n >> 8)};
        fwrite(hdr, 1, 2, out);
        fwrite(packet, 1, (size_t)n, out);
    }
    fclose(out);
    opus_custom_encoder_destroy(enc);
    opus_custom_mode_destroy(mode);
    free(raw);
    return 0;
}

static int decode(Args a) {
    size_t len = 0;
    unsigned char *data = read_file(a.in, &len);
    int err = 0;
    OpusCustomMode *mode = make_mode(a.frame);
    OpusCustomDecoder *dec = opus_custom_decoder_create(mode, a.channels, &err);
    if (!dec || err != OPUS_OK) die("opus_custom_decoder_create failed");
    if (opus_custom_decoder_ctl(dec, CELT_SET_SIGNALLING_REQUEST, (opus_int32)a.signalling) != OPUS_OK)
        die("signalling");
    FILE *out = fopen(a.out, "wb");
    if (!out) die("cannot open output");
    size_t pos = 0;
    size_t index = 0;
    int16_t *s16 = malloc(sizeof(int16_t) * a.frame * a.channels);
    int32_t *s32 = malloc(sizeof(int32_t) * a.frame * a.channels);
    float *f32 = malloc(sizeof(float) * a.frame * a.channels);
    while (pos + 2 <= len) {
        int n = data[pos] | (data[pos + 1] << 8);
        pos += 2;
        if (pos + (size_t)n > len) die("truncated packet file");
        const unsigned char *packet = n ? data + pos : NULL;
        int ret;
        if (!strcmp(a.pcm, "s16")) {
            ret = opus_custom_decode(dec, packet, n, s16, a.frame);
            if (ret > 0) fwrite(s16, sizeof(int16_t), (size_t)ret * a.channels, out);
        } else if (!strcmp(a.pcm, "s24")) {
            ret = opus_custom_decode24(dec, packet, n, s32, a.frame);
            if (ret > 0) fwrite(s32, sizeof(int32_t), (size_t)ret * a.channels, out);
        } else {
            ret = opus_custom_decode_float(dec, packet, n, f32, a.frame);
            if (ret > 0) fwrite(f32, sizeof(float), (size_t)ret * a.channels, out);
        }
        if (ret <= 0) {
            fprintf(stderr, "packet %zu: decode error %d\n", index, ret);
            if (!a.skip_errors) return 1;
        }
        pos += (size_t)n;
        index++;
    }
    fclose(out);
    free(s16);
    free(s32);
    free(f32);
    opus_custom_decoder_destroy(dec);
    opus_custom_mode_destroy(mode);
    free(data);
    return 0;
}

static int modeinfo(Args a) {
#ifdef OPUS_CUSTOM_INTERNALS
    OpusCustomMode *m = make_mode(a.frame);
    int i;
    printf("fs %d\nframe %d\nmax_lm %d\nshort_mdct_size %d\nnb_short_mdcts %d\noverlap %d\n",
           (int)m->Fs, a.frame, m->maxLM, m->shortMdctSize, m->nbShortMdcts, m->overlap);
    printf("nb_ebands %d\neff_ebands %d\nnb_alloc_vectors %d\n", m->nbEBands, m->effEBands,
           m->nbAllocVectors);
    printf("ebands");
    for (i = 0; i <= m->nbEBands; i++) printf(" %d", m->eBands[i]);
    printf("\nalloc_vectors");
    for (i = 0; i < m->nbAllocVectors * m->nbEBands; i++) printf(" %d", m->allocVectors[i]);
    printf("\nlog_n");
    for (i = 0; i < m->nbEBands; i++) printf(" %d", m->logN[i]);
    printf("\nwindow_bits");
    for (i = 0; i < m->overlap; i++) {
        union { float f; uint32_t u; } w;
        w.f = m->window[i];
        printf(" %08x", w.u);
    }
    printf("\ncache_size %d\ncache_index", m->cache.size);
    for (i = 0; i < m->nbEBands * (m->maxLM + 2); i++) printf(" %d", m->cache.index[i]);
    printf("\ncache_bits");
    for (i = 0; i < m->cache.size; i++) printf(" %d", m->cache.bits[i]);
    printf("\ncache_caps");
    for (i = 0; i < (m->maxLM + 1) * 2 * m->nbEBands; i++) printf(" %d", m->cache.caps[i]);
    printf("\npreemph %.10f %.10f %.10f %.10f\n", m->preemph[0], m->preemph[1], m->preemph[2],
           m->preemph[3]);
    {
        int err = 0, lookahead = 0;
        OpusCustomDecoder *e = opus_custom_decoder_create(m, 1, &err);
        opus_custom_decoder_ctl(e, OPUS_GET_LOOKAHEAD(&lookahead));
        printf("lookahead %d\n", lookahead);
        opus_custom_decoder_destroy(e);
    }
    opus_custom_mode_destroy(m);
    return 0;
#else
    (void)a;
    die("built without OPUS_CUSTOM_INTERNALS");
    return 1;
#endif
}

int main(int argc, char **argv) {
    if (argc < 2) usage();
    Args a = parse(argc, argv, 2);
    if (!strcmp(argv[1], "encode")) {
        if (!a.in || !a.out) usage();
        return encode(a);
    }
    if (!strcmp(argv[1], "decode")) {
        if (!a.in || !a.out) usage();
        return decode(a);
    }
    if (!strcmp(argv[1], "modeinfo")) return modeinfo(a);
    usage();
    return 2;
}
