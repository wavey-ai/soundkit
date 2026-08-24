// Persistent FFmpeg reference over a bundle of raw 5 ms FLAC frames.
//
// Build:
//   cc -O3 scripts/ffmpeg_flac_packet_bench.c \
//      $(pkg-config --cflags --libs libavcodec libavutil) \
//      -o target/release/ffmpeg_flac_packet_bench
//
// Usage:
//   ffmpeg_flac_packet_bench PACKET_BUNDLE PCM_S32LE RATE ITERATIONS RUNS

#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "benchmark_clock.h"

enum {
    CHANNELS = 2,
    BITS_PER_SAMPLE = 24,
    WARMUP = 1024,
    MAX_PACKET_BYTES = 16 * 1024 * 1024
};

typedef struct {
    uint8_t *data;
    size_t size;
} Bytes;

typedef struct {
    AVPacket **packets;
    size_t count;
    uint64_t total_bytes;
} Bundle;

static int compare_double(const void *left, const void *right) {
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return a < b ? -1 : a > b;
}

static double percentile(const double *sorted, size_t count, unsigned percent) {
    size_t rank = (count * percent + 99) / 100;
    if (rank == 0)
        rank = 1;
    if (rank > count)
        rank = count;
    return sorted[rank - 1];
}

static uint32_t read_u32le(const uint8_t *bytes) {
    return (uint32_t)bytes[0] |
           (uint32_t)bytes[1] << 8 |
           (uint32_t)bytes[2] << 16 |
           (uint32_t)bytes[3] << 24;
}

static Bytes read_file(const char *path) {
    Bytes result = {0};
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) {
        if (file)
            fclose(file);
        return result;
    }
    const long length = ftell(file);
    if (length <= 0 || fseek(file, 0, SEEK_SET)) {
        fclose(file);
        return result;
    }
    result.size = (size_t)length;
    result.data = malloc(result.size);
    if (!result.data || fread(result.data, 1, result.size, file) != result.size) {
        free(result.data);
        result.data = NULL;
        result.size = 0;
    }
    fclose(file);
    return result;
}

static void free_bundle(Bundle *bundle) {
    if (!bundle->packets)
        return;
    for (size_t index = 0; index < bundle->count; index++)
        av_packet_free(&bundle->packets[index]);
    free(bundle->packets);
    bundle->packets = NULL;
    bundle->count = 0;
}

static Bundle read_bundle(const char *path) {
    Bundle bundle = {0};
    Bytes bytes = read_file(path);
    if (!bytes.data)
        return bundle;
    size_t offset = 0;
    size_t capacity = 0;
    while (offset < bytes.size) {
        if (bytes.size - offset < 4)
            goto fail;
        const uint32_t length = read_u32le(&bytes.data[offset]);
        offset += 4;
        if (length == 0 || length > MAX_PACKET_BYTES ||
            bytes.size - offset < length)
            goto fail;
        if (bundle.count == capacity) {
            const size_t next = capacity ? capacity * 2 : 64;
            AVPacket **packets = realloc(bundle.packets, next * sizeof(*packets));
            if (!packets)
                goto fail;
            bundle.packets = packets;
            capacity = next;
        }
        AVPacket *packet = av_packet_alloc();
        if (!packet || av_new_packet(packet, (int)length) < 0) {
            av_packet_free(&packet);
            goto fail;
        }
        memcpy(packet->data, &bytes.data[offset], length);
        bundle.packets[bundle.count++] = packet;
        bundle.total_bytes += length;
        offset += length;
    }
    free(bytes.data);
    return bundle;

fail:
    free(bytes.data);
    free_bundle(&bundle);
    return bundle;
}

static int decode_one(AVCodecContext *context, AVPacket *packet, AVFrame *frame) {
    int status = avcodec_send_packet(context, packet);
    if (status < 0)
        return status;
    status = avcodec_receive_frame(context, frame);
    return status < 0 ? status : 0;
}

static int32_t decoded_sample(
    const AVFrame *frame,
    unsigned sample,
    unsigned channel
) {
    const int planar = av_sample_fmt_is_planar((enum AVSampleFormat)frame->format);
    const int32_t *samples =
        (const int32_t *)frame->extended_data[planar ? channel : 0];
    return samples[planar ? sample : sample * CHANNELS + channel];
}

static int verify_pcm(
    const AVFrame *frame,
    const int32_t *expected,
    unsigned frame_length,
    int *sample_shift
) {
    if (frame->nb_samples != (int)frame_length || frame->format == AV_SAMPLE_FMT_NONE)
        return 1;
#if LIBAVUTIL_VERSION_MAJOR >= 57
    if (frame->ch_layout.nb_channels != CHANNELS)
        return 1;
#else
    if (frame->channels != CHANNELS)
        return 1;
#endif
    int shift = *sample_shift;
    if (shift < 0) {
        for (unsigned sample = 0; sample < frame_length && shift < 0; sample++) {
            for (unsigned channel = 0; channel < CHANNELS; channel++) {
                const int32_t want = expected[sample * CHANNELS + channel];
                if (want == 0)
                    continue;
                const int32_t got = decoded_sample(frame, sample, channel);
                if (got == want)
                    shift = 0;
                else if (got == (int32_t)((uint32_t)want << 8))
                    shift = 8;
                else
                    return 1;
            }
        }
        if (shift >= 0)
            *sample_shift = shift;
    }
    const int verification_shift = shift < 0 ? 0 : shift;
    for (unsigned sample = 0; sample < frame_length; sample++) {
        for (unsigned channel = 0; channel < CHANNELS; channel++) {
            const int32_t got = decoded_sample(frame, sample, channel);
            const int32_t want = expected[sample * CHANNELS + channel];
            if (got != (int32_t)((uint32_t)want << verification_shift))
                return 1;
        }
    }
    return 0;
}

static AVCodecContext *new_decoder(unsigned sample_rate) {
    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_FLAC);
    AVCodecContext *context = codec ? avcodec_alloc_context3(codec) : NULL;
    if (!context)
        return NULL;
    context->sample_rate = (int)sample_rate;
    context->bits_per_raw_sample = BITS_PER_SAMPLE;
    context->thread_count = 1;
#if LIBAVUTIL_VERSION_MAJOR >= 57
    av_channel_layout_default(&context->ch_layout, CHANNELS);
#else
    context->channels = CHANNELS;
    context->channel_layout = AV_CH_LAYOUT_STEREO;
#endif
    if (avcodec_open2(context, codec, NULL) < 0) {
        avcodec_free_context(&context);
        return NULL;
    }
    return context;
}

static int run_once(
    const Bundle *bundle,
    unsigned sample_rate,
    unsigned frame_length,
    unsigned iterations,
    double *timings,
    uint64_t *checksum,
    const char **sample_format
) {
    AVCodecContext *context = new_decoder(sample_rate);
    AVFrame *frame = av_frame_alloc();
    if (!context || !frame) {
        avcodec_free_context(&context);
        av_frame_free(&frame);
        return 1;
    }
    for (unsigned iteration = 0; iteration < WARMUP; iteration++) {
        AVPacket *packet = bundle->packets[iteration % bundle->count];
        if (decode_one(context, packet, frame) < 0)
            goto fail;
        av_frame_unref(frame);
    }
    uint64_t sum = 0;
    for (unsigned iteration = 0; iteration < iterations; iteration++) {
        AVPacket *packet = bundle->packets[iteration % bundle->count];
        const uint64_t started = benchmark_now_ticks();
        const int status = decode_one(context, packet, frame);
        timings[iteration] = benchmark_elapsed_us(started, benchmark_now_ticks());
        if (status < 0)
            goto fail;
        sum += (uint32_t)decoded_sample(
            frame, iteration % frame_length, iteration & 1);
        *sample_format = av_get_sample_fmt_name((enum AVSampleFormat)frame->format);
        av_frame_unref(frame);
    }
    *checksum += sum;
    av_frame_free(&frame);
    avcodec_free_context(&context);
    return 0;

fail:
    av_frame_free(&frame);
    avcodec_free_context(&context);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr,
                "usage: ffmpeg_flac_packet_bench PACKET_BUNDLE PCM_S32LE "
                "RATE ITERATIONS RUNS\n");
        return 2;
    }
    const unsigned sample_rate = (unsigned)atoi(argv[3]);
    const unsigned iterations = (unsigned)atoi(argv[4]);
    const unsigned runs = (unsigned)atoi(argv[5]);
    if ((sample_rate != 48000 && sample_rate != 96000) ||
        iterations == 0 || runs == 0)
        return 2;
    const unsigned frame_length = sample_rate / 200;
    const size_t samples_per_frame = (size_t)frame_length * CHANNELS;
    Bundle bundle = read_bundle(argv[1]);
    Bytes encoded_pcm = read_file(argv[2]);
    if (!bundle.packets || !encoded_pcm.data ||
        encoded_pcm.size != bundle.count * samples_per_frame * sizeof(int32_t)) {
        fprintf(stderr, "bundle and S32LE corpus geometry do not match\n");
        free_bundle(&bundle);
        free(encoded_pcm.data);
        return 1;
    }
    int32_t *expected = malloc(encoded_pcm.size);
    if (!expected) {
        free_bundle(&bundle);
        free(encoded_pcm.data);
        return 1;
    }
    for (size_t index = 0; index < encoded_pcm.size / 4; index++)
        expected[index] = (int32_t)read_u32le(&encoded_pcm.data[index * 4]);
    free(encoded_pcm.data);

    AVCodecContext *verify_context = new_decoder(sample_rate);
    AVFrame *verify_frame = av_frame_alloc();
    int sample_shift = -1;
    if (!verify_context || !verify_frame)
        goto fail;
    for (size_t index = 0; index < bundle.count; index++) {
        if (decode_one(verify_context, bundle.packets[index], verify_frame) < 0 ||
            verify_pcm(verify_frame, &expected[index * samples_per_frame],
                       frame_length, &sample_shift)) {
            fprintf(stderr, "FFmpeg decoded PCM mismatch in corpus frame %zu\n", index);
            goto fail;
        }
        av_frame_unref(verify_frame);
    }
    av_frame_free(&verify_frame);
    avcodec_free_context(&verify_context);

    const size_t timing_count = (size_t)iterations * runs;
    double *timings = malloc(timing_count * sizeof(*timings));
    if (!timings)
        goto fail;
    uint64_t checksum = 0;
    const char *sample_format = NULL;
    for (unsigned run = 0; run < runs; run++) {
        if (run_once(&bundle, sample_rate, frame_length, iterations,
                     &timings[(size_t)run * iterations], &checksum,
                     &sample_format)) {
            free(timings);
            goto fail;
        }
    }
    qsort(timings, timing_count, sizeof(*timings), compare_double);
    const double pcm_bytes =
        (double)bundle.count * frame_length * CHANNELS * (BITS_PER_SAMPLE / 8);
    printf(
        "ffmpeg decode rate=%u frame=%u corpus_frames=%zu "
        "p50_us=%.3f p95_us=%.3f p99_us=%.3f min_us=%.3f "
        "encoded/pcm=%.4f sample_fmt=%s sample_shift=%d checksum=%llu "
        "calls=%zu runs=%u\n",
        sample_rate, frame_length, bundle.count,
        percentile(timings, timing_count, 50),
        percentile(timings, timing_count, 95),
        percentile(timings, timing_count, 99), timings[0],
        (double)bundle.total_bytes / pcm_bytes,
        sample_format ? sample_format : "unknown", sample_shift < 0 ? 0 : sample_shift,
        (unsigned long long)checksum, timing_count, runs
    );
    free(timings);
    free(expected);
    free_bundle(&bundle);
    return 0;

fail:
    av_frame_free(&verify_frame);
    avcodec_free_context(&verify_context);
    free(expected);
    free_bundle(&bundle);
    return 1;
}
