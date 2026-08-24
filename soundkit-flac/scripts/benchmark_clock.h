#ifndef SOUNDKIT_FLAC_BENCHMARK_CLOCK_H
#define SOUNDKIT_FLAC_BENCHMARK_CLOCK_H

#include <stdint.h>

#if defined(__APPLE__)
#include <mach/mach_time.h>

static inline uint64_t benchmark_now_ticks(void) {
    return mach_absolute_time();
}

static inline double benchmark_elapsed_us(uint64_t started, uint64_t finished) {
    static mach_timebase_info_data_t timebase = {0, 0};
    if (timebase.denom == 0)
        mach_timebase_info(&timebase);
    return (double)(finished - started) * (double)timebase.numer /
           (double)timebase.denom / 1000.0;
}

#else
#include <time.h>

static inline uint64_t benchmark_now_ticks(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static inline double benchmark_elapsed_us(uint64_t started, uint64_t finished) {
    return (double)(finished - started) / 1000.0;
}
#endif

#endif
