#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "timing_utils.h"

#define NUM_CPU_MEASUREMENTS 1000
#define ARRAY_SIZE (16 * 1024 * 1024)
#define PROBE_STRIDE 64
#define CACHELINE_INTS 16
#define CPU_WORK_DIM 16 //128
#define GPU_WORK_DIM 128
#define MAX_CONGRUENT_INDICES 8192
#define DEFAULT_LLC_SIZE_KB (12 * 1024)
#define DEFAULT_LLC_WAYS 16
#define DEFAULT_GPU_ITERATIONS 200

volatile int* g_data = NULL;
volatile int g_run_measurement = 0;
uint64_t g_cpu_timings[NUM_CPU_MEASUREMENTS];
int g_measurement_count = 0;
int g_probe_index = 0;
int g_use_fixed_probe = 0;
volatile float g_cpu_work_checksum = 0.0f;
float* g_cpu_mat_a = NULL;
float* g_cpu_mat_b = NULL;
float* g_cpu_mat_c = NULL;

static void cpu_matrix_workload_step(int iter_seed) {
    int row_bias = iter_seed % CPU_WORK_DIM;
    float checksum = 0.0f;

    for (int i = 0; i < CPU_WORK_DIM; i++) {
        int row = (i + row_bias) % CPU_WORK_DIM;
        for (int j = 0; j < CPU_WORK_DIM; j++) {
            float acc = 0.0f;
            for (int k = 0; k < CPU_WORK_DIM; k++) {
                acc += g_cpu_mat_a[row * CPU_WORK_DIM + k] * g_cpu_mat_b[k * CPU_WORK_DIM + j];
            }
            g_cpu_mat_c[row * CPU_WORK_DIM + j] = acc;
            checksum += acc;
        }
    }

    g_cpu_work_checksum += checksum;
}

void* cpu_measurement_thread(void* arg) {
    printf("[CPU Thread] Started. Waiting for signal...\n");

    while (!g_run_measurement) {
        usleep(1000);
    }

    printf("[CPU Thread] Starting measurements...\n");

    for (int i = 0; i < NUM_CPU_MEASUREMENTS && g_run_measurement; i++) {
        int index = g_use_fixed_probe ? g_probe_index : ((i * PROBE_STRIDE) % ARRAY_SIZE);

        cpu_matrix_workload_step(i);

        cpu_barrier();
        uint64_t start = rdtsc();
        volatile int value = g_data[index];
        load_barrier();
        uint64_t end = rdtsc();
        cpu_barrier();

        g_cpu_timings[i] = end - start;
        g_measurement_count = i + 1;

        if (value == -1) {
            g_data[0] = 1;
        }

        usleep(100);
    }

    printf("[CPU Thread] Completed %d measurements.\n", g_measurement_count);
    return NULL;
}

__global__ void gpu_compute_contention_kernel(int* data, int size,
                                              const float* mat_a,
                                              const float* mat_b,
                                              float* mat_c, int dim,
                                              int num_iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total_outputs = dim * dim;

    for (int iter = 0; iter < num_iterations; iter++) {
        for (int out = tid; out < total_outputs; out += stride) {
            int row = out / dim;
            int col = out % dim;
            float acc = 0.0f;
            for (int k = 0; k < dim; k++) {
                acc += mat_a[row * dim + k] * mat_b[k * dim + col];
            }
            mat_c[out] = acc + (float)iter;
        }

        for (int i = tid; i < size; i += stride) {
            volatile int value = data[i];
            data[i] = value + 1;
        }
    }
}

__global__ void gpu_targeted_set_kernel(int* data,
                                        const int* target_indices,
                                        int num_indices,
                                        const float* mat_a,
                                        const float* mat_b,
                                        float* mat_c,
                                        int dim,
                                        int num_iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total_outputs = dim * dim;

    for (int iter = 0; iter < num_iterations; iter++) {
        for (int out = tid; out < total_outputs; out += stride) {
            int row = out / dim;
            int col = out % dim;
            float acc = 0.0f;
            for (int k = 0; k < dim; k++) {
                acc += mat_a[row * dim + k] * mat_b[k * dim + col];
            }
            mat_c[out] = acc + (float)iter;
        }

        for (int j = tid; j < num_indices; j += stride) {
            int idx = target_indices[j];
            volatile int value = data[idx];
            data[idx] = value + 1;
        }
    }
}

uint64_t resolve_set_index(void* addr, int cache_size_kb, int ways, int* used_physical) {
    uint64_t pa = 0;
    int rc = virtual_to_physical(addr, &pa);
    if (rc == 0) {
        *used_physical = 1;
        return get_cache_set_index_from_phys(pa, cache_size_kb, ways);
    }
    *used_physical = 0;
    return get_cache_set_index(addr, cache_size_kb, ways);
}

int collect_congruent_indices(volatile int* data,
                              int array_size,
                              uint64_t target_set,
                              int cache_size_kb,
                              int ways,
                              int* out_indices,
                              int max_out,
                              int* physical_matches,
                              int* approx_matches,
                              int* vtop_failures) {
    int count = 0;
    *physical_matches = 0;
    *approx_matches = 0;
    *vtop_failures = 0;

    for (int i = 0; i < array_size; i += CACHELINE_INTS) {
        void* addr = (void*)&data[i];
        uint64_t pa = 0;
        int rc = virtual_to_physical(addr, &pa);

        uint64_t set_idx;
        if (rc == 0) {
            set_idx = get_cache_set_index_from_phys(pa, cache_size_kb, ways);
        } else {
            (*vtop_failures)++;
            set_idx = get_cache_set_index(addr, cache_size_kb, ways);
        }

        if (set_idx == target_set && count < max_out) {
            out_indices[count++] = i;
            if (rc == 0) {
                (*physical_matches)++;
            } else {
                (*approx_matches)++;
            }
        }
    }

    return count;
}

void print_sample_mapping(volatile int* data,
                          int probe_index,
                          const int* indices,
                          int count,
                          int cache_size_kb,
                          int ways) {
    int sample_count = (count < 8) ? count : 8;
    printf("\n[Validation] Mapping sample (%d entries):\n", sample_count + 1);

    void* probe_addr = (void*)&data[probe_index];
    uint64_t probe_pa = 0;
    int probe_rc = virtual_to_physical(probe_addr, &probe_pa);
    if (probe_rc == 0) {
        uint64_t set = get_cache_set_index_from_phys(probe_pa, cache_size_kb, ways);
        printf("  Probe idx=%d, VA=%p, PA=0x%llx, Set=%llu (physical)\n",
               probe_index, probe_addr, (unsigned long long)probe_pa, (unsigned long long)set);
    } else {
        uint64_t set = get_cache_set_index(probe_addr, cache_size_kb, ways);
        printf("  Probe idx=%d, VA=%p, PA=<unavailable: %s>, Set=%llu (virtual approx)\n",
               probe_index, probe_addr, vtop_error_string(probe_rc), (unsigned long long)set);
    }

    for (int i = 0; i < sample_count; i++) {
        int idx = indices[i];
        void* addr = (void*)&data[idx];
        uint64_t pa = 0;
        int rc = virtual_to_physical(addr, &pa);
        if (rc == 0) {
            uint64_t set = get_cache_set_index_from_phys(pa, cache_size_kb, ways);
            printf("  Target[%d] idx=%d, VA=%p, PA=0x%llx, Set=%llu (physical)\n",
                   i, idx, addr, (unsigned long long)pa, (unsigned long long)set);
        } else {
            uint64_t set = get_cache_set_index(addr, cache_size_kb, ways);
            printf("  Target[%d] idx=%d, VA=%p, PA=<unavailable>, Set=%llu (virtual approx)\n",
                   i, idx, addr, (unsigned long long)set);
        }
    }
}

int main(int argc, char** argv) {
    printf("=== HYBRID CPU-GPU CACHE MEASUREMENT (WITH REAL WORKLOADS) ===\n\n");

    int enable_gpu_contention = (argc > 1) ? atoi(argv[1]) : 0;
    int enable_set_validation = (argc > 2) ? atoi(argv[2]) : 1;
    int cache_size_kb = (argc > 3) ? atoi(argv[3]) : DEFAULT_LLC_SIZE_KB;
    int cache_ways = (argc > 4) ? atoi(argv[4]) : DEFAULT_LLC_WAYS;
    int target_set_override = (argc > 5) ? atoi(argv[5]) : -1;
    int gpu_iterations = (argc > 6) ? atoi(argv[6]) : DEFAULT_GPU_ITERATIONS;

    int total_sets = (cache_size_kb * 1024) / (64 * cache_ways);
    if (target_set_override >= total_sets) {
        printf("[Config] target_set_override=%d out of range [0,%d). Ignoring override.\n",
               target_set_override, total_sets);
        target_set_override = -1;
    }

    printf("Configuration:\n");
    printf("  Array Size: %d elements (%.2f MB)\n", ARRAY_SIZE, ARRAY_SIZE * sizeof(int) / 1e6);
    printf("  CPU Measurements: %d\n", NUM_CPU_MEASUREMENTS);
    printf("  GPU Contention: %s\n", enable_gpu_contention ? "ENABLED" : "DISABLED");
    printf("  Set Validation: %s\n", enable_set_validation ? "ENABLED" : "DISABLED");
    printf("  LLC Geometry: %d KB, %d ways (%d sets)\n", cache_size_kb, cache_ways, total_sets);
    printf("  Target Set Override: %d\n", target_set_override);
    printf("  GPU Iterations: %d\n", gpu_iterations);
    printf("  CPU workload: %dx%d matrix multiply per probe iteration\n", CPU_WORK_DIM, CPU_WORK_DIM);
    printf("  GPU workload: %dx%d matrix multiply fused with contention\n\n", GPU_WORK_DIM, GPU_WORK_DIM);

    int* data;
    CUDA_CHECK(cudaMallocManaged(&data, ARRAY_SIZE * sizeof(int)));
    g_data = data;

    size_t cpu_work_bytes = CPU_WORK_DIM * CPU_WORK_DIM * sizeof(float);
    CUDA_CHECK(cudaMallocManaged(&g_cpu_mat_a, cpu_work_bytes));
    CUDA_CHECK(cudaMallocManaged(&g_cpu_mat_b, cpu_work_bytes));
    CUDA_CHECK(cudaMallocManaged(&g_cpu_mat_c, cpu_work_bytes));

    float *gpu_mat_a, *gpu_mat_b, *gpu_mat_c;
    size_t gpu_work_bytes = GPU_WORK_DIM * GPU_WORK_DIM * sizeof(float);
    CUDA_CHECK(cudaMalloc(&gpu_mat_a, gpu_work_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_mat_b, gpu_work_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_mat_c, gpu_work_bytes));

    printf("Initializing data on CPU...\n");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        data[i] = i;
    }
    for (int i = 0; i < CPU_WORK_DIM * CPU_WORK_DIM; i++) {
        g_cpu_mat_a[i] = 1.0f + (float)(i % 7);
        g_cpu_mat_b[i] = 0.5f + (float)(i % 5);
        g_cpu_mat_c[i] = 0.0f;
    }

    float* gpu_init_a = (float*)malloc(gpu_work_bytes);
    float* gpu_init_b = (float*)malloc(gpu_work_bytes);
    for (int i = 0; i < GPU_WORK_DIM * GPU_WORK_DIM; i++) {
        gpu_init_a[i] = 1.0f + (float)(i % 11);
        gpu_init_b[i] = 0.25f + (float)(i % 13);
    }
    CUDA_CHECK(cudaMemcpy(gpu_mat_a, gpu_init_a, gpu_work_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_mat_b, gpu_init_b, gpu_work_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(gpu_mat_c, 0, gpu_work_bytes));
    free(gpu_init_a);
    free(gpu_init_b);

    CUDA_CHECK(cudaMemPrefetchAsync(data, ARRAY_SIZE * sizeof(int), cudaCpuDeviceId));
    CUDA_CHECK(cudaMemPrefetchAsync(g_cpu_mat_a, cpu_work_bytes, cudaCpuDeviceId));
    CUDA_CHECK(cudaMemPrefetchAsync(g_cpu_mat_b, cpu_work_bytes, cudaCpuDeviceId));
    CUDA_CHECK(cudaMemPrefetchAsync(g_cpu_mat_c, cpu_work_bytes, cudaCpuDeviceId));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Data initialized and pinned to CPU memory.\n\n");

    int* congruent_indices_host = (int*)malloc(MAX_CONGRUENT_INDICES * sizeof(int));
    int* congruent_indices_managed = NULL;
    int num_congruent = 0;
    int use_targeted_kernel = 0;

    g_probe_index = 0;
    g_use_fixed_probe = enable_set_validation ? 1 : 0;

    if (enable_set_validation) {
        int probe_used_physical = 0;
        uint64_t probe_set = resolve_set_index((void*)&data[g_probe_index], cache_size_kb, cache_ways, &probe_used_physical);
        uint64_t target_set = (target_set_override >= 0) ? (uint64_t)target_set_override : probe_set;

        int physical_matches = 0;
        int approx_matches = 0;
        int vtop_failures = 0;

        num_congruent = collect_congruent_indices(g_data, ARRAY_SIZE, target_set, cache_size_kb, cache_ways,
                                                  congruent_indices_host, MAX_CONGRUENT_INDICES,
                                                  &physical_matches, &approx_matches, &vtop_failures);

        printf("[Validation] Probe set index: %llu (%s)\n",
               (unsigned long long)probe_set,
               probe_used_physical ? "physical" : "virtual approximation");
        printf("[Validation] Target set index: %llu (%s)\n",
               (unsigned long long)target_set,
               target_set_override >= 0 ? "override" : "probe-derived");
        printf("[Validation] Congruent candidates found: %d\n", num_congruent);
        printf("[Validation] Candidates with physical mapping: %d\n", physical_matches);
        printf("[Validation] Candidates with virtual approximation: %d\n", approx_matches);
        printf("[Validation] vtop failures while scanning: %d\n", vtop_failures);

        if (num_congruent > 0) {
            g_probe_index = congruent_indices_host[0];
        }

        if (!probe_used_physical || physical_matches == 0) {
            printf("[Validation] Warning: exact set congruence is not fully proven on this run (pagemap restricted).\n");
        }

        if (num_congruent >= (cache_ways + 2)) {
            CUDA_CHECK(cudaMallocManaged(&congruent_indices_managed, num_congruent * sizeof(int)));
            for (int i = 0; i < num_congruent; i++) {
                congruent_indices_managed[i] = congruent_indices_host[i];
            }
            CUDA_CHECK(cudaMemPrefetchAsync(congruent_indices_managed, num_congruent * sizeof(int), 0));
            CUDA_CHECK(cudaDeviceSynchronize());
            use_targeted_kernel = 1;
            print_sample_mapping(g_data, g_probe_index, congruent_indices_host, num_congruent, cache_size_kb, cache_ways);
        } else {
            printf("[Validation] Not enough congruent lines for targeted eviction. Falling back to broad contention.\n");
        }

        printf("\n");
    }

    pthread_t cpu_thread;
    pthread_create(&cpu_thread, NULL, cpu_measurement_thread, NULL);
    sleep(1);

    g_run_measurement = 1;

    if (enable_gpu_contention) {
        int num_blocks = 8;
        int threads_per_block = 256;

        if (use_targeted_kernel) {
            printf("[Main] Launching TARGETED GPU kernel on %d congruent indices...\n", num_congruent);
            gpu_targeted_set_kernel<<<num_blocks, threads_per_block>>>(data, congruent_indices_managed, num_congruent,
                                                                       gpu_mat_a, gpu_mat_b, gpu_mat_c,
                                                                       GPU_WORK_DIM, gpu_iterations);
        } else {
            printf("[Main] Launching BROAD GPU compute+contention kernel...\n");
            gpu_compute_contention_kernel<<<num_blocks, threads_per_block>>>(data, ARRAY_SIZE,
                                                                             gpu_mat_a, gpu_mat_b, gpu_mat_c,
                                                                             GPU_WORK_DIM, gpu_iterations);
        }

        CUDA_CHECK(cudaGetLastError());
        printf("[Main] GPU kernel launched (running concurrently with CPU measurements).\n\n");
    } else {
        printf("[Main] No GPU contention - measuring baseline.\n\n");
    }

    pthread_join(cpu_thread, NULL);

    if (enable_gpu_contention) {
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Main] GPU kernel completed.\n\n");
    }

    g_run_measurement = 0;

    printf("=== CPU-Side Timing Analysis ===\n");
    double mean = calculate_mean(g_cpu_timings, g_measurement_count);
    double stddev = calculate_stddev(g_cpu_timings, g_measurement_count, mean);
    uint64_t min_latency = find_min(g_cpu_timings, g_measurement_count);
    uint64_t max_latency = find_max(g_cpu_timings, g_measurement_count);

    printf("Mean Latency:   %.2f CPU cycles\n", mean);
    printf("Std Deviation:  %.2f CPU cycles\n", stddev);
    printf("Min Latency:    %llu CPU cycles\n", (unsigned long long)min_latency);
    printf("Max Latency:    %llu CPU cycles\n", (unsigned long long)max_latency);
    printf("CPU workload checksum: %.2f\n\n", g_cpu_work_checksum);

    char filename[256];
    snprintf(filename, sizeof(filename), "results/hybrid_cpu_timings_%s.csv",
             enable_gpu_contention ? "contention" : "baseline");

    FILE* fp = fopen(filename, "w");
    if (fp) {
        fprintf(fp, "iteration,latency_cpu_cycles\n");
        for (int i = 0; i < g_measurement_count; i++) {
            fprintf(fp, "%d,%llu\n", i, (unsigned long long)g_cpu_timings[i]);
        }
        fclose(fp);
        printf("Results saved to: %s\n", filename);
    }

    printf("\nFirst 20 CPU access latencies:\n");
    printf("Iteration | Latency (CPU cycles)\n");
    printf("----------+---------------------\n");
    for (int i = 0; i < 20 && i < g_measurement_count; i++) {
        printf("%9d | %19llu\n", i, (unsigned long long)g_cpu_timings[i]);
    }

    if (congruent_indices_managed) {
        cudaFree(congruent_indices_managed);
    }
    free(congruent_indices_host);
    cudaFree(gpu_mat_a);
    cudaFree(gpu_mat_b);
    cudaFree(gpu_mat_c);
    cudaFree(g_cpu_mat_a);
    cudaFree(g_cpu_mat_b);
    cudaFree(g_cpu_mat_c);
    cudaFree(data);

    printf("\n=== Measurement Complete ===\n");
    return 0;
}
