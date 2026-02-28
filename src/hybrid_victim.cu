#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "timing_utils.h"

// ============================================================================
// Configuration
// ============================================================================

#define NUM_CPU_MEASUREMENTS 1000
#define ARRAY_SIZE (256 * 1024)
#define PROBE_STRIDE 64  // Access every cache line

// Global variables for CPU-side measurement
volatile int* g_data = NULL;
volatile int g_run_measurement = 0;
uint64_t g_cpu_timings[NUM_CPU_MEASUREMENTS];
int g_measurement_count = 0;

// ============================================================================
// CPU Thread: Measure Access Latency from CPU Side
// ============================================================================

void* cpu_measurement_thread(void* arg) {
    printf("[CPU Thread] Started. Waiting for signal...\n");

    // Wait for signal to start measuring
    while (!g_run_measurement) {
        usleep(1000);
    }

    printf("[CPU Thread] Starting measurements...\n");

    for (int i = 0; i < NUM_CPU_MEASUREMENTS && g_run_measurement; i++) {
        // Calculate probe index
        int index = (i * PROBE_STRIDE) % ARRAY_SIZE;

        // Serialize and start timing
        cpu_barrier();
        uint64_t start = rdtsc();

        // Access memory (forces CPU cache check)
        volatile int value = g_data[index];

        // End timing
        uint64_t end = rdtsc();
        cpu_barrier();

        // Store timing
        g_cpu_timings[i] = end - start;
        g_measurement_count = i + 1;

        // Prevent optimization
        if (value == -1) {
            g_data[0] = 1;
        }

        // Small delay to spread measurements over time
        usleep(100);  // 100 microseconds
    }

    printf("[CPU Thread] Completed %d measurements.\n", g_measurement_count);

    return NULL;
}

// ============================================================================
// GPU Kernel: Concurrent GPU Access (Creates Contention)
// ============================================================================

__global__ void gpu_contention_kernel(int* data, int size, int num_iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int iter = 0; iter < num_iterations; iter++) {
        for (int i = tid; i < size; i += stride) {
            volatile int value = data[i];
            data[i] = value + 1;
        }
    }
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) {
    printf("=== HYBRID CPU-GPU CACHE MEASUREMENT ===\n\n");

    int enable_gpu_contention = (argc > 1) ? atoi(argv[1]) : 0;

    printf("Configuration:\n");
    printf("  Array Size: %d elements (%.2f MB)\n", ARRAY_SIZE, ARRAY_SIZE * sizeof(int) / 1e6);
    printf("  CPU Measurements: %d\n", NUM_CPU_MEASUREMENTS);
    printf("  GPU Contention: %s\n", enable_gpu_contention ? "ENABLED" : "DISABLED");
    printf("\n");

    // ========================================================================
    // Allocate Unified Memory
    // ========================================================================

    int* data;
    CUDA_CHECK(cudaMallocManaged(&data, ARRAY_SIZE * sizeof(int)));
    g_data = data;

    // Initialize on CPU
    printf("Initializing data on CPU...\n");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        data[i] = i;
    }

    // Prefetch to CPU
    CUDA_CHECK(cudaMemPrefetchAsync(data, ARRAY_SIZE * sizeof(int), cudaCpuDeviceId));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Data initialized and pinned to CPU memory.\n\n");

    // ========================================================================
    // Start CPU Measurement Thread
    // ========================================================================

    pthread_t cpu_thread;
    pthread_create(&cpu_thread, NULL, cpu_measurement_thread, NULL);

    // Give thread time to start
    sleep(1);

    // ========================================================================
    // Optionally Launch GPU Contention
    // ========================================================================

    g_run_measurement = 1;  // Signal CPU thread to start

    if (enable_gpu_contention) {
        printf("[Main] Launching GPU contention kernel...\n");

        int num_blocks = 8;
        int threads_per_block = 256;
        int num_iterations = 100;

        gpu_contention_kernel<<<num_blocks, threads_per_block>>>(data, ARRAY_SIZE, num_iterations);
        CUDA_CHECK(cudaGetLastError());

        printf("[Main] GPU kernel launched (running concurrently with CPU measurements).\n\n");
    } else {
        printf("[Main] No GPU contention - measuring baseline.\n\n");
    }

    // ========================================================================
    // Wait for Measurements to Complete
    // ========================================================================

    pthread_join(cpu_thread, NULL);

    if (enable_gpu_contention) {
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Main] GPU kernel completed.\n\n");
    }

    g_run_measurement = 0;

    // ========================================================================
    // Analyze CPU Timing Data
    // ========================================================================

    printf("=== CPU-Side Timing Analysis ===\n");

    double mean = calculate_mean(g_cpu_timings, g_measurement_count);
    double stddev = calculate_stddev(g_cpu_timings, g_measurement_count, mean);
    uint64_t min_latency = find_min(g_cpu_timings, g_measurement_count);
    uint64_t max_latency = find_max(g_cpu_timings, g_measurement_count);

    printf("Mean Latency:   %.2f CPU cycles\n", mean);
    printf("Std Deviation:  %.2f CPU cycles\n", stddev);
    printf("Min Latency:    %lu CPU cycles\n", min_latency);
    printf("Max Latency:    %lu CPU cycles\n", max_latency);
    printf("\n");

    // ========================================================================
    // Save Results
    // ========================================================================

    char filename[256];
    snprintf(filename, sizeof(filename), "results/hybrid_cpu_timings_%s.csv",
             enable_gpu_contention ? "contention" : "baseline");

    FILE* fp = fopen(filename, "w");
    if (fp) {
        fprintf(fp, "iteration,latency_cpu_cycles\n");
        for (int i = 0; i < g_measurement_count; i++) {
            fprintf(fp, "%d,%lu\n", i, g_cpu_timings[i]);
        }
        fclose(fp);
        printf("Results saved to: %s\n", filename);
    }

    // Display sample
    printf("\nFirst 20 CPU access latencies:\n");
    printf("Iteration | Latency (CPU cycles)\n");
    printf("----------+---------------------\n");
    for (int i = 0; i < 20 && i < g_measurement_count; i++) {
        printf("%9d | %19lu\n", i, g_cpu_timings[i]);
    }

    // ========================================================================
    // Cleanup
    // ========================================================================

    cudaFree(data);

    printf("\n=== Measurement Complete ===\n");

    return 0;
}
