#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>

static uint64_t monotonic_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        perror("clock_gettime");
        exit(1);
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
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
    uint8_t *data = malloc((size_t)size + AV_INPUT_BUFFER_PADDING_SIZE);
    if (data == NULL) exit(1);
    if (fread(data, 1, (size_t)size, file) != (size_t)size) exit(1);
    fclose(file);
    memset(data + size, 0, AV_INPUT_BUFFER_PADDING_SIZE);
    *size_out = (size_t)size;
    return data;
}

static int decode_once(AVCodecContext *context, AVFrame *frame, uint8_t *data, size_t size) {
    AVPacket packet = {0};
    packet.data = data;
    packet.size = (int)size;
    int result = avcodec_send_packet(context, &packet);
    if (result < 0) return result;

    int samples = 0;
    for (;;) {
        result = avcodec_receive_frame(context, frame);
        if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) break;
        if (result < 0) return result;
        samples += frame->nb_samples * frame->ch_layout.nb_channels;
        av_frame_unref(frame);
    }
    return samples;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: ffmpeg_g722_decode_bench <input.g722> [iterations]\n");
        return 2;
    }
    int iterations = argc == 3 ? atoi(argv[2]) : 100;
    if (iterations <= 0) return 2;

    size_t input_size = 0;
    uint8_t *input = read_file(argv[1], &input_size);
    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_ADPCM_G722);
    if (codec == NULL) {
        fprintf(stderr, "FFmpeg G.722 decoder is unavailable\n");
        return 1;
    }
    AVCodecContext *context = avcodec_alloc_context3(codec);
    AVFrame *frame = av_frame_alloc();
    if (context == NULL || frame == NULL || avcodec_open2(context, codec, NULL) < 0) return 1;

    for (int index = 0; index < 5; ++index) {
        avcodec_flush_buffers(context);
        if (decode_once(context, frame, input, input_size) < 0) return 1;
    }

    uint64_t started = monotonic_ns();
    uint64_t samples = 0;
    for (int index = 0; index < iterations; ++index) {
        avcodec_flush_buffers(context);
        int decoded = decode_once(context, frame, input, input_size);
        if (decoded < 0) return 1;
        samples += (uint64_t)decoded;
    }
    uint64_t elapsed = monotonic_ns() - started;

    printf("implementation=ffmpeg codec=g722 operation=decode input_bytes=%zu iterations=%d samples=%" PRIu64 " elapsed_ns=%" PRIu64 "\n",
           input_size, iterations, samples, elapsed);

    av_frame_free(&frame);
    avcodec_free_context(&context);
    free(input);
    return 0;
}
