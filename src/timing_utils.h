#ifndef TIMING_UTILS_H
#define TIMING_UTILS_H

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

// ============================================================================
// CPU Timing using RDTSC (Read Time-Stamp Counter)
// ============================================================================

// Read the CPU timestamp counter (high-resolution timing)
static inline uint64_t rdtsc() {
    uint32_t lo, hi;
    __asm__ __volatile__ (
        "rdtscp"
        : "=a"(lo), "=d"(hi)
        :: "%rcx"
    );
    return ((uint64_t)hi << 32) | lo;
}

// Serialize execution before timing measurement
static inline void cpu_barrier() {
    __asm__ __volatile__ (
        "mfence\n\t"
        "lfence\n\t"
        ::: "memory"
    );
}

// ============================================================================
// GPU Timing using CUDA Events
// ============================================================================

class CUDATimer {
private:
    cudaEvent_t start, stop;

public:
    CUDATimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~CUDATimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start() {
        cudaEventRecord(start);
    }

    void Stop() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    }

    float ElapsedMilliseconds() {
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

// ============================================================================
// GPU Device-Side Timing using clock64()
// ============================================================================

__device__ inline uint64_t gpu_clock() {
    return clock64();
}

// ============================================================================
// CUDA Error Checking Macro
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Statistics Utilities
// ============================================================================

// Calculate mean of an array
inline double calculate_mean(const uint64_t* data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum / n;
}

// Calculate standard deviation
inline double calculate_stddev(const uint64_t* data, int n, double mean) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = data[i] - mean;
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq / n);
}

// Find minimum value
inline uint64_t find_min(const uint64_t* data, int n) {
    uint64_t min_val = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] < min_val) {
            min_val = data[i];
        }
    }
    return min_val;
}

// Find maximum value
inline uint64_t find_max(const uint64_t* data, int n) {
    uint64_t max_val = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    return max_val;
}

// ============================================================================
// Cache Set Index Calculation (for targeting specific CPU cache sets)
// ============================================================================

// For a typical CPU LLC with:
// - Cache line size: 64 bytes
// - N-way set associative cache
// - Total cache size: e.g., 12 MB
//
// Cache set index = (Physical Address / 64) % (Total Sets)
inline uint64_t get_cache_set_index(void* addr, int cache_size_kb, int ways) {
    const int line_size = 64;
    int total_sets = (cache_size_kb * 1024) / (line_size * ways);
    uint64_t phys_addr = (uint64_t)addr;
    return (phys_addr / line_size) % total_sets;
}

#endif // TIMING_UTILS_H
