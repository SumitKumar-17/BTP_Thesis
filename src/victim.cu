#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "timing_utils.h"

// ============================================================================
// Configuration Parameters
// ============================================================================

#define NUM_ITERATIONS 1000
#define ARRAY_SIZE (256 * 1024)  // 256K elements = 1 MB of data
#define ACCESS_STRIDE 16          // Access every 16th element (cache line = 64 bytes / 4 bytes per int = 16 ints)

// ============================================================================
// GPU Kernel: Memory Access Pattern
// ============================================================================

__global__ void victim_kernel(int* data, uint64_t* timings, int num_accesses) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {  // Single thread performs timed accesses
        for (int iter = 0; iter < num_accesses; iter++) {
            // Calculate index with predictable pattern
            int index = (iter * ACCESS_STRIDE) % ARRAY_SIZE;

            // Start timing
            uint64_t start = gpu_clock();

            // Perform memory access (forces GPU to fetch from CPU memory via UVM)
            volatile int value = data[index];

            // End timing
            uint64_t end = gpu_clock();

            // Store timing
            timings[iter] = end - start;

            // Use the value to prevent optimization
            if (value == -1) {
                data[0] = 1;
            }
        }
    }
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) {
    printf("=== GPU-CPU Cache Contention: VICTIM Process ===\n\n");

    // Parse command line arguments
    int num_accesses = NUM_ITERATIONS;
    if (argc > 1) {
        num_accesses = atoi(argv[1]);
    }

    printf("Configuration:\n");
    printf("  Array Size: %d elements (%.2f MB)\n", ARRAY_SIZE, ARRAY_SIZE * sizeof(int) / 1e6);
    printf("  Access Stride: %d elements\n", ACCESS_STRIDE);
    printf("  Number of Accesses: %d\n", num_accesses);
    printf("\n");

    // ========================================================================
    // Step 1: Allocate Unified Memory (accessible from both CPU and GPU)
    // ========================================================================

    int* data;
    uint64_t* timings_device;
    uint64_t* timings_host;

    printf("Allocating unified memory...\n");

    // Allocate data array in unified memory
    CUDA_CHECK(cudaMallocManaged(&data, ARRAY_SIZE * sizeof(int)));

    // Allocate timing arrays
    CUDA_CHECK(cudaMallocManaged(&timings_device, num_accesses * sizeof(uint64_t)));
    timings_host = (uint64_t*)malloc(num_accesses * sizeof(uint64_t));

    // ========================================================================
    // Step 2: Initialize data on CPU (ensures it's in CPU memory initially)
    // ========================================================================

    printf("Initializing data on CPU...\n");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        data[i] = i;
    }

    // Prefetch data to CPU memory
    CUDA_CHECK(cudaMemPrefetchAsync(data, ARRAY_SIZE * sizeof(int), cudaCpuDeviceId));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Data initialized and prefetched to CPU memory.\n\n");

    // ========================================================================
    // Step 3: Run the victim kernel
    // ========================================================================

    printf("Running victim kernel...\n");

    CUDATimer timer;
    timer.Start();

    // Launch kernel with single thread (to measure sequential access latency)
    victim_kernel<<<1, 1>>>(data, timings_device, num_accesses);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.Stop();

    printf("Kernel execution time: %.3f ms\n", timer.ElapsedMilliseconds());
    printf("\n");

    // ========================================================================
    // Step 4: Copy timing data to host and analyze
    // ========================================================================

    printf("Analyzing timing data...\n");

    // Copy timing data to host
    memcpy(timings_host, timings_device, num_accesses * sizeof(uint64_t));

    // Calculate statistics
    double mean = calculate_mean(timings_host, num_accesses);
    double stddev = calculate_stddev(timings_host, num_accesses, mean);
    uint64_t min_latency = find_min(timings_host, num_accesses);
    uint64_t max_latency = find_max(timings_host, num_accesses);

    printf("\n=== Timing Statistics (GPU clock cycles) ===\n");
    printf("Mean Latency:   %.2f cycles\n", mean);
    printf("Std Deviation:  %.2f cycles\n", stddev);
    printf("Min Latency:    %lu cycles\n", min_latency);
    printf("Max Latency:    %lu cycles\n", max_latency);
    printf("\n");

    // ========================================================================
    // Step 5: Save results to file
    // ========================================================================

    char filename[256];
    snprintf(filename, sizeof(filename), "results/victim_timings_%d.csv", num_accesses);

    FILE* fp = fopen(filename, "w");
    if (fp) {
        fprintf(fp, "iteration,latency_cycles\n");
        for (int i = 0; i < num_accesses; i++) {
            fprintf(fp, "%d,%lu\n", i, timings_host[i]);
        }
        fclose(fp);
        printf("Results saved to: %s\n", filename);
    } else {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
    }

    // ========================================================================
    // Step 6: Display sample of timing data
    // ========================================================================

    printf("\nFirst 20 access latencies:\n");
    printf("Iteration | Latency (cycles)\n");
    printf("----------+-----------------\n");
    for (int i = 0; i < 20 && i < num_accesses; i++) {
        printf("%9d | %15lu\n", i, timings_host[i]);
    }

    // ========================================================================
    // Cleanup
    // ========================================================================

    cudaFree(data);
    cudaFree(timings_device);
    free(timings_host);

    printf("\n=== VICTIM Process Complete ===\n");

    return 0;
}
