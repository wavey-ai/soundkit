// Decode a length-prefixed raw FLAC packet bundle with FFmpeg and compare it
// bit-exactly with an interleaved S32LE corpus.
//
// Build:
//   cc -O2 scripts/ffmpeg_flac_packet_verify.c \
//      $(pkg-config --cflags --libs libavcodec libavutil) -o ffmpeg_flac_packet_verify
// Usage:
//   ffmpeg_flac_packet_verify BUNDLE PCM_S32LE RATE CHANNELS 16|24

#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_PACKET_BYTES = 16 * 1024 * 1024 };

typedef struct {
    uint8_t *data;
    size_t size;
} Bytes;

typedef struct {
    AVPacket **items;
    size_t count;
} Packets;

static uint32_t read_u32le(const uint8_t *bytes) {
    return (uint32_t)bytes[0] |
           (uint32_t)bytes[1] << 8 |
           (uint32_t)bytes[2] << 16 |
           (uint32_t)bytes[3] << 24;
}

static Bytes read_file(const char *path) {
    Bytes bytes = {0};
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) {
        if (file)
            fclose(file);
        return bytes;
    }
    const long length = ftell(file);
    if (length <= 0 || fseek(file, 0, SEEK_SET)) {
        fclose(file);
        return bytes;
    }
    bytes.size = (size_t)length;
    bytes.data = malloc(bytes.size);
    if (!bytes.data || fread(bytes.data, 1, bytes.size, file) != bytes.size) {
        free(bytes.data);
        bytes.data = NULL;
        bytes.size = 0;
    }
    fclose(file);
    return bytes;
}

static void free_packets(Packets *packets) {
    for (size_t index = 0; index < packets->count; index++)
        av_packet_free(&packets->items[index]);
    free(packets->items);
    packets->items = NULL;
    packets->count = 0;
}

static Packets read_packets(const char *path) {
    Packets packets = {0};
    Bytes bytes = read_file(path);
    size_t capacity = 0;
    size_t offset = 0;
    while (bytes.data && offset < bytes.size) {
        if (bytes.size - offset < 4)
            goto fail;
        const uint32_t length = read_u32le(&bytes.data[offset]);
        offset += 4;
        if (length == 0 || length > MAX_PACKET_BYTES || bytes.size - offset < length)
            goto fail;
        if (packets.count == capacity) {
            const size_t next = capacity ? capacity * 2 : 64;
            AVPacket **items = realloc(packets.items, next * sizeof(*items));
            if (!items)
                goto fail;
            packets.items = items;
            capacity = next;
        }
        AVPacket *packet = av_packet_alloc();
        if (!packet || av_new_packet(packet, (int)length) < 0) {
            av_packet_free(&packet);
            goto fail;
        }
        memcpy(packet->data, &bytes.data[offset], length);
        packets.items[packets.count++] = packet;
        offset += length;
    }
    free(bytes.data);
    return packets;

fail:
    free(bytes.data);
    free_packets(&packets);
    return packets;
}

static int decode_one(AVCodecContext *context, AVPacket *packet, AVFrame *frame) {
    const int sent = avcodec_send_packet(context, packet);
    if (sent < 0)
        return sent;
    return avcodec_receive_frame(context, frame);
}

static int sample_i32(
    const AVFrame *frame,
    unsigned channels,
    unsigned sample,
    unsigned channel,
    int32_t *output
) {
    const enum AVSampleFormat format = (enum AVSampleFormat)frame->format;
    const int planar = av_sample_fmt_is_planar(format);
    const size_t index = planar ? sample : (size_t)sample * channels + channel;
    const uint8_t *data = frame->extended_data[planar ? channel : 0];
    switch (format) {
        case AV_SAMPLE_FMT_S16:
        case AV_SAMPLE_FMT_S16P:
            *output = ((const int16_t *)data)[index];
            return 0;
        case AV_SAMPLE_FMT_S32:
        case AV_SAMPLE_FMT_S32P:
            *output = ((const int32_t *)data)[index];
            return 0;
        default:
            return 1;
    }
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr,
                "usage: ffmpeg_flac_packet_verify BUNDLE PCM_S32LE RATE "
                "CHANNELS 16|24\n");
        return 2;
    }
    const unsigned sample_rate = (unsigned)atoi(argv[3]);
    const unsigned channels = (unsigned)atoi(argv[4]);
    const unsigned bits = (unsigned)atoi(argv[5]);
    if ((sample_rate != 48000 && sample_rate != 96000) ||
        channels == 0 || channels > 8 || (bits != 16 && bits != 24))
        return 2;
    const unsigned frame_length = sample_rate / 200;
    const size_t samples_per_frame = (size_t)frame_length * channels;
    Packets packets = read_packets(argv[1]);
    Bytes pcm_bytes = read_file(argv[2]);
    if (!packets.items || !pcm_bytes.data || pcm_bytes.size % 4 ||
        pcm_bytes.size / 4 != packets.count * samples_per_frame) {
        fprintf(stderr, "bundle and PCM corpus geometry do not match\n");
        free_packets(&packets);
        free(pcm_bytes.data);
        return 1;
    }
    int32_t *pcm = malloc(pcm_bytes.size);
    if (!pcm) {
        free_packets(&packets);
        free(pcm_bytes.data);
        return 1;
    }
    for (size_t index = 0; index < pcm_bytes.size / 4; index++)
        pcm[index] = (int32_t)read_u32le(&pcm_bytes.data[index * 4]);
    free(pcm_bytes.data);

    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_FLAC);
    AVCodecContext *context = codec ? avcodec_alloc_context3(codec) : NULL;
    AVFrame *frame = av_frame_alloc();
    if (!context || !frame)
        goto fail;
    context->sample_rate = (int)sample_rate;
    context->bits_per_raw_sample = (int)bits;
    context->thread_count = 1;
#if LIBAVUTIL_VERSION_MAJOR >= 57
    av_channel_layout_default(&context->ch_layout, (int)channels);
#else
    context->channels = (int)channels;
    context->channel_layout = av_get_default_channel_layout((int)channels);
#endif
    if (avcodec_open2(context, codec, NULL) < 0)
        goto fail;

    int shift = -1;
    enum AVSampleFormat decoded_format = AV_SAMPLE_FMT_NONE;
    for (size_t packet_index = 0; packet_index < packets.count; packet_index++) {
        if (decode_one(context, packets.items[packet_index], frame) < 0 ||
            frame->nb_samples != (int)frame_length)
            goto mismatch;
        decoded_format = (enum AVSampleFormat)frame->format;
#if LIBAVUTIL_VERSION_MAJOR >= 57
        if (frame->ch_layout.nb_channels != (int)channels)
            goto mismatch;
#else
        if (frame->channels != (int)channels)
            goto mismatch;
#endif
        const int32_t *expected = &pcm[packet_index * samples_per_frame];
        if (shift < 0) {
            for (size_t index = 0; index < samples_per_frame && shift < 0; index++) {
                if (expected[index] == 0)
                    continue;
                int32_t got = 0;
                if (sample_i32(frame, channels, (unsigned)(index / channels),
                               (unsigned)(index % channels), &got))
                    goto mismatch;
                if (got == expected[index])
                    shift = 0;
                else if (got == (int32_t)((uint32_t)expected[index] << 8))
                    shift = 8;
                else if (got == (int32_t)((uint32_t)expected[index] << 16))
                    shift = 16;
                else
                    goto mismatch;
            }
        }
        const int check_shift = shift < 0 ? 0 : shift;
        for (size_t index = 0; index < samples_per_frame; index++) {
            int32_t got = 0;
            if (sample_i32(frame, channels, (unsigned)(index / channels),
                           (unsigned)(index % channels), &got) ||
                got != (int32_t)((uint32_t)expected[index] << check_shift))
                goto mismatch;
        }
        av_frame_unref(frame);
    }
    printf("ffmpeg verified rate=%u channels=%u bits=%u frames=%zu "
           "sample_fmt=%s sample_shift=%d\n",
           sample_rate, channels, bits, packets.count,
           av_get_sample_fmt_name(decoded_format),
           shift < 0 ? 0 : shift);
    av_frame_free(&frame);
    avcodec_free_context(&context);
    free(pcm);
    free_packets(&packets);
    return 0;

mismatch:
    fprintf(stderr, "FFmpeg PCM mismatch in packet sequence\n");
fail:
    av_frame_free(&frame);
    avcodec_free_context(&context);
    free(pcm);
    free_packets(&packets);
    return 1;
}
