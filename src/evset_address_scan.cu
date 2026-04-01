#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include "timing_utils.h"

#define ARRAY_SIZE (16 * 1024 * 1024)
#define CACHELINE_INTS 16
#define MAX_CONGRUENT_INDICES 8192
#define DEFAULT_LLC_SIZE_KB (12 * 1024)
#define DEFAULT_LLC_WAYS 16

__global__ void gpu_targeted_set_kernel(int* data,
                                        const int* target_indices,
                                        int num_indices,
                                        int num_iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int iter = 0; iter < num_iterations; iter++) {
        for (int j = tid; j < num_indices; j += stride) {
            int idx = target_indices[j];
            volatile int value = data[idx];
            data[idx] = value + 1;
        }
    }
}

struct Stats {
    uint64_t median;
    uint64_t p99;
    uint64_t maxv;
};

static Stats compute_stats(std::vector<uint64_t>& samples) {
    std::sort(samples.begin(), samples.end());
    int n = (int)samples.size();
    int p99_idx = (int)(0.99 * (n - 1));
    Stats s;
    s.median = samples[n / 2];
    s.p99 = samples[p99_idx];
    s.maxv = samples[n - 1];
    return s;
}

static int collect_congruent_indices(volatile int* data,
                                     int array_size,
                                     uint64_t target_set,
                                     int cache_size_kb,
                                     int ways,
                                     int* out_indices,
                                     int max_out) {
    int count = 0;
    for (int i = 0; i < array_size; i += CACHELINE_INTS) {
        uint64_t pa = 0;
        int rc = virtual_to_physical((void*)&data[i], &pa);
        if (rc != 0) {
            continue;
        }
        uint64_t set_idx = get_cache_set_index_from_phys(pa, cache_size_kb, ways);
        if (set_idx == target_set && count < max_out) {
            out_indices[count++] = i;
        }
    }
    return count;
}

int main(int argc, char** argv) {
    int target_set = (argc > 1) ? atoi(argv[1]) : 0;
    int points = (argc > 2) ? atoi(argv[2]) : 88;
    int reps = (argc > 3) ? atoi(argv[3]) : 64;
    int gpu_iters = (argc > 4) ? atoi(argv[4]) : 20000;
    int cache_kb = (argc > 5) ? atoi(argv[5]) : DEFAULT_LLC_SIZE_KB;
    int ways = (argc > 6) ? atoi(argv[6]) : DEFAULT_LLC_WAYS;

    printf("=== EVICTION SET ADDRESS SCAN ===\n");
    printf("Target set: %d\nPoints: %d\nSamples per address: %d\nGPU iterations: %d\n\n",
           target_set, points, reps, gpu_iters);

    int* data = NULL;
    CUDA_CHECK(cudaMallocManaged(&data, ARRAY_SIZE * sizeof(int)));
    for (int i = 0; i < ARRAY_SIZE; i++) {
        data[i] = i;
    }
    CUDA_CHECK(cudaMemPrefetchAsync(data, ARRAY_SIZE * sizeof(int), cudaCpuDeviceId));
    CUDA_CHECK(cudaDeviceSynchronize());

    int* idx_host = (int*)malloc(MAX_CONGRUENT_INDICES * sizeof(int));
    int found = collect_congruent_indices(data, ARRAY_SIZE, (uint64_t)target_set,
                                          cache_kb, ways, idx_host, MAX_CONGRUENT_INDICES);
    if (found < ways + 2) {
        fprintf(stderr, "Not enough physically-congruent lines found: %d\n", found);
        cudaFree(data);
        free(idx_host);
        return 1;
    }

    int n = std::min(points, found);
    printf("Congruent indices found: %d, using first %d\n", found, n);

    int* idx_dev = NULL;
    CUDA_CHECK(cudaMallocManaged(&idx_dev, n * sizeof(int)));
    for (int i = 0; i < n; i++) {
        idx_dev[i] = idx_host[i];
    }
    CUDA_CHECK(cudaMemPrefetchAsync(idx_dev, n * sizeof(int), 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<Stats> base_stats(n), atk_stats(n);

    // Baseline: gather samples per address.
    for (int i = 0; i < n; i++) {
        std::vector<uint64_t> s;
        s.reserve(reps);
        for (int r = 0; r < reps; r++) {
            cpu_barrier();
            uint64_t start = rdtsc();
            volatile int value = data[idx_host[i]];
            load_barrier();
            uint64_t end = rdtsc();
            cpu_barrier();
            if (value == -1) {
                data[0] = 1;
            }
            s.push_back(end - start);
        }
        base_stats[i] = compute_stats(s);
    }

    // Attack: run targeted kernel while sampling same addresses.
    gpu_targeted_set_kernel<<<8, 256>>>(data, idx_dev, n, gpu_iters);
    CUDA_CHECK(cudaGetLastError());

    for (int i = 0; i < n; i++) {
        std::vector<uint64_t> s;
        s.reserve(reps);
        for (int r = 0; r < reps; r++) {
            cpu_barrier();
            uint64_t start = rdtsc();
            volatile int value = data[idx_host[i]];
            load_barrier();
            uint64_t end = rdtsc();
            cpu_barrier();
            if (value == -1) {
                data[0] = 1;
            }
            s.push_back(end - start);
        }
        atk_stats[i] = compute_stats(s);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    char out_csv[256];
    snprintf(out_csv, sizeof(out_csv), "results/evset_address_scan_set%d.csv", target_set);
    FILE* fp = fopen(out_csv, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open %s\n", out_csv);
        cudaFree(idx_dev);
        cudaFree(data);
        free(idx_host);
        return 1;
    }

    fprintf(fp, "order,index,baseline_median,attack_median,median_ratio,baseline_p99,attack_p99,p99_ratio,baseline_max,attack_max,max_ratio\n");
    for (int i = 0; i < n; i++) {
        double med_ratio = (base_stats[i].median > 0) ? (double)atk_stats[i].median / (double)base_stats[i].median : 0.0;
        double p99_ratio = (base_stats[i].p99 > 0) ? (double)atk_stats[i].p99 / (double)base_stats[i].p99 : 0.0;
        double max_ratio = (base_stats[i].maxv > 0) ? (double)atk_stats[i].maxv / (double)base_stats[i].maxv : 0.0;
        fprintf(fp, "%d,%d,%llu,%llu,%.6f,%llu,%llu,%.6f,%llu,%llu,%.6f\n",
                i, idx_host[i],
                (unsigned long long)base_stats[i].median,
                (unsigned long long)atk_stats[i].median,
                med_ratio,
                (unsigned long long)base_stats[i].p99,
                (unsigned long long)atk_stats[i].p99,
                p99_ratio,
                (unsigned long long)base_stats[i].maxv,
                (unsigned long long)atk_stats[i].maxv,
                max_ratio);
    }
    fclose(fp);

    printf("Saved: %s\n", out_csv);

    cudaFree(idx_dev);
    cudaFree(data);
    free(idx_host);
    return 0;
}
