#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define BENCH_EXPORT extern "C" EMSCRIPTEN_KEEPALIVE
#else
#include <time.h>
#define BENCH_EXPORT extern "C"
#endif

#include "aacdecoder_lib.h"

typedef struct AdtsFrame {
  const UCHAR *data;
  UINT size;
} AdtsFrame;

typedef struct Fixture {
  AdtsFrame *frames;
  UINT frame_count;
  UINT sample_rate;
} Fixture;

static uint32_t last_decoded_frames;
static double last_samples_per_channel;
static uint32_t last_checksum_high;
static uint32_t last_checksum_low;
static int last_error;

static const uint32_t sample_rates[] = {
    96000, 88200, 64000, 48000, 44100, 32000, 24000,
    22050, 16000, 12000, 11025, 8000, 7350,
};

static double bench_now_ms(void) {
#ifdef __EMSCRIPTEN__
  return emscripten_get_now();
#else
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
    return -1.0;
  }
  return (double)now.tv_sec * 1000.0 + (double)now.tv_nsec / 1000000.0;
#endif
}

static void free_fixture(Fixture *fixture) {
  free(fixture->frames);
  memset(fixture, 0, sizeof(*fixture));
}

static int parse_fixture(const uint8_t *data, size_t size, Fixture *fixture) {
  size_t offset = 0;
  UINT capacity = 0;
  memset(fixture, 0, sizeof(*fixture));

  while (offset + 7 <= size) {
    if (data[offset] != 0xff || (data[offset + 1] & 0xf6) != 0xf0) {
      return -1;
    }

    const UINT header_size = (data[offset + 1] & 1) != 0 ? 7 : 9;
    const UINT frame_size = ((data[offset + 3] & 3) << 11) |
                            (data[offset + 4] << 3) |
                            (data[offset + 5] >> 5);
    if (frame_size <= header_size || offset + frame_size > size ||
        (data[offset + 6] & 3) != 0) {
      return -1;
    }

    if (fixture->frame_count == capacity) {
      const UINT next_capacity = capacity == 0 ? 1024 : capacity * 2;
      AdtsFrame *next = (AdtsFrame *)realloc(
          fixture->frames, (size_t)next_capacity * sizeof(*fixture->frames));
      if (next == NULL) {
        return -2;
      }
      fixture->frames = next;
      capacity = next_capacity;
    }

    fixture->frames[fixture->frame_count++] = {
        (const UCHAR *)(data + offset), frame_size};

    if (fixture->frame_count == 1) {
      const UINT sample_rate_index = (data[offset + 2] >> 2) & 15;
      if (sample_rate_index >=
          sizeof(sample_rates) / sizeof(sample_rates[0])) {
        return -1;
      }
      fixture->sample_rate = sample_rates[sample_rate_index];
    }

    offset += frame_size;
  }

  return offset == size && fixture->frame_count != 0 ? 0 : -1;
}

static uint64_t checksum_sample(uint64_t checksum, INT_PCM sample) {
  checksum ^= (uint32_t)sample;
  return checksum * UINT64_C(0x100000001b3);
}

static int decode_pass(HANDLE_AACDECODER decoder, INT_PCM *pcm,
                       UINT pcm_capacity, const Fixture *fixture, int collect,
                       uint32_t *decoded_frames,
                       uint64_t *samples_per_channel, uint64_t *checksum) {
  for (UINT index = 0; index < fixture->frame_count; ++index) {
    UCHAR *input = (UCHAR *)fixture->frames[index].data;
    const UINT input_size = fixture->frames[index].size;
    UINT bytes_valid = input_size;
    AAC_DECODER_ERROR error =
        aacDecoder_Fill(decoder, &input, &input_size, &bytes_valid);
    if (error != AAC_DEC_OK) {
      return (int)error;
    }
    if (bytes_valid == input_size) {
      return -3;
    }

    error = aacDecoder_DecodeFrame(decoder, pcm, (INT)pcm_capacity, 0);
    if (error == AAC_DEC_NOT_ENOUGH_BITS) {
      continue;
    }
    if (error != AAC_DEC_OK) {
      return (int)error;
    }

    CStreamInfo *info = aacDecoder_GetStreamInfo(decoder);
    if (info == NULL || info->frameSize <= 0 || info->numChannels <= 0 ||
        (UINT)(info->frameSize * info->numChannels) > pcm_capacity) {
      return -4;
    }
    if (!collect) {
      continue;
    }

    ++*decoded_frames;
    *samples_per_channel += (uint64_t)info->frameSize;
    const INT points[] = {0, info->frameSize / 2, info->frameSize - 1};
    for (INT channel = 0; channel < info->numChannels; ++channel) {
      for (UINT point = 0; point < 3; ++point) {
        *checksum = checksum_sample(
            *checksum, pcm[points[point] * info->numChannels + channel]);
      }
    }
  }
  return 0;
}

BENCH_EXPORT double fdk_aac_bench(const uint8_t *data, size_t size,
                                  int iterations) {
  Fixture fixture;
  HANDLE_AACDECODER decoder = NULL;
  const UINT pcm_capacity = 16384;
  INT_PCM *pcm = NULL;
  double elapsed = -1.0;
  uint32_t decoded_frames = 0;
  uint64_t samples_per_channel = 0;
  uint64_t checksum = UINT64_C(0xcbf29ce484222325);

  last_decoded_frames = 0;
  last_samples_per_channel = 0;
  last_checksum_high = 0;
  last_checksum_low = 0;
  last_error = 0;
  if (iterations < 1) {
    iterations = 1;
  }

  int result = parse_fixture(data, size, &fixture);
  if (result != 0) {
    last_error = result;
    goto cleanup;
  }

  decoder = aacDecoder_Open(TT_MP4_ADTS, 1);
  pcm = (INT_PCM *)malloc((size_t)pcm_capacity * sizeof(*pcm));
  if (decoder == NULL || pcm == NULL) {
    last_error = -2;
    goto cleanup;
  }

  result = decode_pass(decoder, pcm, pcm_capacity, &fixture, 0,
                       &decoded_frames, &samples_per_channel, &checksum);
  if (result != 0) {
    last_error = result;
    goto cleanup;
  }

  {
    const double started = bench_now_ms();
    for (int iteration = 0; iteration < iterations; ++iteration) {
      result = decode_pass(decoder, pcm, pcm_capacity, &fixture, 1,
                           &decoded_frames, &samples_per_channel, &checksum);
      if (result != 0) {
        last_error = result;
        goto cleanup;
      }
    }
    elapsed = bench_now_ms() - started;
  }

  last_decoded_frames = decoded_frames;
  last_samples_per_channel = (double)samples_per_channel;
  last_checksum_high = (uint32_t)(checksum >> 32);
  last_checksum_low = (uint32_t)checksum;

cleanup:
  if (decoder != NULL) {
    aacDecoder_Close(decoder);
  }
  free(pcm);
  free_fixture(&fixture);
  return elapsed;
}

BENCH_EXPORT uint32_t fdk_aac_last_decoded_frames(void) {
  return last_decoded_frames;
}

BENCH_EXPORT double fdk_aac_last_samples_per_channel(void) {
  return last_samples_per_channel;
}

BENCH_EXPORT uint32_t fdk_aac_last_checksum_high(void) {
  return last_checksum_high;
}

BENCH_EXPORT uint32_t fdk_aac_last_checksum_low(void) {
  return last_checksum_low;
}

BENCH_EXPORT int fdk_aac_last_error(void) { return last_error; }
