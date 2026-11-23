#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "timing_utils.h"

// ============================================================================
// Configuration Parameters
// ============================================================================

#define NUM_ITERATIONS 10000
#define CACHE_LINE_SIZE 64
#define LLC_SIZE_KB (12 * 1024)  // Assume 12 MB LLC on typical CPU
#define LLC_WAYS 16              // Assume 16-way set associative

// Calculate number of cache sets
#define NUM_CACHE_SETS (LLC_SIZE_KB * 1024 / CACHE_LINE_SIZE / LLC_WAYS)

// Eviction set size: number of addresses needed to evict a cache line
#define EVICTION_SET_SIZE (LLC_WAYS + 2)

// ============================================================================
// GPU Kernel: Cache Eviction Attack
// ============================================================================

__global__ void attacker_kernel(char** eviction_sets, int num_sets, int num_iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {  // Single thread performs the attack
        for (int iter = 0; iter < num_iterations; iter++) {
            // Access all addresses in eviction sets to force cache eviction
            for (int set = 0; set < num_sets; set++) {
                for (int way = 0; way < EVICTION_SET_SIZE; way++) {
                    // Force memory access (volatile to prevent optimization)
                    volatile char value = eviction_sets[set][way * CACHE_LINE_SIZE];

                    // Write back to ensure cache pollution
                    eviction_sets[set][way * CACHE_LINE_SIZE] = (char)(value + 1);
                }
            }

            // Small delay between iterations to allow victim to run
            if (iter % 10 == 0) {
                __nanosleep(100);  // 100 ns delay
            }
        }
    }
}

// ============================================================================
// GPU Kernel: Aggressive Memory Thrashing
// ============================================================================

__global__ void thrashing_kernel(char* memory, size_t size, int num_iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int iter = 0; iter < num_iterations; iter++) {
        // Each thread accesses memory with stride to maximize cache pollution
        for (size_t i = tid * CACHE_LINE_SIZE; i < size; i += stride * CACHE_LINE_SIZE) {
            volatile char value = memory[i];
            memory[i] = (char)(value + 1);
        }
    }
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) {
    printf("=== GPU-CPU Cache Contention: ATTACKER Process ===\n\n");

    // Parse command line arguments
    int attack_mode = 1;  // 1 = eviction set, 2 = thrashing
    int num_iterations = NUM_ITERATIONS;

    if (argc > 1) {
        attack_mode = atoi(argv[1]);
    }
    if (argc > 2) {
        num_iterations = atoi(argv[2]);
    }

    printf("Configuration:\n");
    printf("  Attack Mode: %d (%s)\n", attack_mode,
           attack_mode == 1 ? "Eviction Set" : "Memory Thrashing");
    printf("  LLC Size: %d KB\n", LLC_SIZE_KB);
    printf("  Cache Line Size: %d bytes\n", CACHE_LINE_SIZE);
    printf("  Estimated Cache Sets: %d\n", NUM_CACHE_SETS);
    printf("  Number of Iterations: %d\n", num_iterations);
    printf("\n");

    if (attack_mode == 1) {
        // ====================================================================
        // Attack Mode 1: Targeted Eviction Set Attack
        // ====================================================================

        printf("Building eviction sets...\n");

        // Number of sets to target (subset of all cache sets)
        int num_target_sets = 1024;  // Target 1024 cache sets

        // Allocate array of pointers to eviction sets
        char** eviction_sets_host = (char**)malloc(num_target_sets * sizeof(char*));
        char** eviction_sets_device;

        CUDA_CHECK(cudaMalloc(&eviction_sets_device, num_target_sets * sizeof(char*)));

        // Allocate each eviction set in unified memory
        for (int i = 0; i < num_target_sets; i++) {
            CUDA_CHECK(cudaMallocManaged(&eviction_sets_host[i],
                                        EVICTION_SET_SIZE * CACHE_LINE_SIZE));

            // Initialize memory
            for (int j = 0; j < EVICTION_SET_SIZE * CACHE_LINE_SIZE; j++) {
                eviction_sets_host[i][j] = (char)j;
            }

            // Prefetch to CPU memory
            CUDA_CHECK(cudaMemPrefetchAsync(eviction_sets_host[i],
                                           EVICTION_SET_SIZE * CACHE_LINE_SIZE,
                                           cudaCpuDeviceId));
        }

        // Copy array of pointers to device
        CUDA_CHECK(cudaMemcpy(eviction_sets_device, eviction_sets_host,
                             num_target_sets * sizeof(char*),
                             cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaDeviceSynchronize());

        printf("Eviction sets built. Starting attack...\n");

        CUDATimer timer;
        timer.Start();

        // Launch attack kernel
        attacker_kernel<<<1, 1>>>(eviction_sets_device, num_target_sets, num_iterations);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.Stop();

        printf("Attack completed in %.3f ms\n", timer.ElapsedMilliseconds());

        // Cleanup
        for (int i = 0; i < num_target_sets; i++) {
            cudaFree(eviction_sets_host[i]);
        }
        cudaFree(eviction_sets_device);
        free(eviction_sets_host);

    } else {
        // ====================================================================
        // Attack Mode 2: Memory Thrashing Attack
        // ====================================================================

        printf("Allocating memory for thrashing attack...\n");

        // Allocate large memory region (16 MB to exceed typical LLC)
        size_t mem_size = 16 * 1024 * 1024;  // 16 MB
        char* memory;

        CUDA_CHECK(cudaMallocManaged(&memory, mem_size));

        // Initialize memory on CPU
        printf("Initializing memory...\n");
        for (size_t i = 0; i < mem_size; i++) {
            memory[i] = (char)i;
        }

        // Prefetch to CPU memory
        CUDA_CHECK(cudaMemPrefetchAsync(memory, mem_size, cudaCpuDeviceId));
        CUDA_CHECK(cudaDeviceSynchronize());

        printf("Starting thrashing attack...\n");

        CUDATimer timer;
        timer.Start();

        // Launch thrashing kernel with multiple threads for aggressive contention
        int num_blocks = 8;
        int threads_per_block = 256;
        thrashing_kernel<<<num_blocks, threads_per_block>>>(memory, mem_size, num_iterations);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.Stop();

        printf("Attack completed in %.3f ms\n", timer.ElapsedMilliseconds());

        // Cleanup
        cudaFree(memory);
    }

    printf("\n=== ATTACKER Process Complete ===\n");

    return 0;
}
