/*
 * evset_workload_scan.cu
 *
 * Eviction-set heatmap scanner with selectable GPU + CPU workloads.
 * GPU: fuses a real compute workload WITH the cache eviction kernel.
 * CPU: performs a real compute workload between each timing probe.
 *
 * GPU workload types (--gpu-workload):
 *   0 = none          (pure cache eviction, no compute)
 *   1 = matmul_small  (128x128 GEMM)
 *   2 = matmul_large  (512x512 GEMM)
 *   3 = conv2d        (simulated 3x3 convolution on 64x64x32 feature map)
 *   4 = bandwidth     (large array sequential read/write — simulates attention)
 *
 * CPU workload types (--cpu-workload):
 *   0 = none
 *   1 = matmul_small  (64x64 GEMM)
 *   2 = matmul_large  (256x256 GEMM)
 *   3 = fft_like      (sum-of-sines across array, approximates frequency domain work)
 *
 * Usage:
 *   evset_workload_scan <target_set> <points> <reps> <gpu_iters>
 *                       <cache_kb> <ways> <gpu_workload> <cpu_workload>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include "timing_utils.h"

#define ARRAY_SIZE       (16 * 1024 * 1024)
#define CACHELINE_INTS   16
#define MAX_CONGRUENT    8192
#define DEFAULT_LLC_KB   (12 * 1024)
#define DEFAULT_WAYS     16

// ----------------------------------------------------------------
// GPU kernels
// ----------------------------------------------------------------

// W0: pure cache eviction (no compute)
__global__ void gpu_evict_only(int* data, const int* indices, int n, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int it = 0; it < iters; it++)
        for (int j = tid; j < n; j += stride) {
            volatile int v = data[indices[j]];
            data[indices[j]] = v + 1;
        }
}

// W1: matmul 128x128 fused with eviction
#define MM1 128
__global__ void gpu_matmul128_evict(int* data, const int* indices, int n,
                                    const float* A, const float* B, float* C, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int it = 0; it < iters; it++) {
        // matmul portion
        for (int out = tid; out < MM1*MM1; out += stride) {
            int r = out / MM1, c = out % MM1;
            float acc = 0.f;
            for (int k = 0; k < MM1; k++) acc += A[r*MM1+k] * B[k*MM1+c];
            C[out] = acc + (float)it;
        }
        // eviction portion
        for (int j = tid; j < n; j += stride) {
            volatile int v = data[indices[j]]; data[indices[j]] = v + 1;
        }
    }
}

// W2: matmul 512x512 fused with eviction
#define MM2 512
__global__ void gpu_matmul512_evict(int* data, const int* indices, int n,
                                    const float* A, const float* B, float* C, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int it = 0; it < iters; it++) {
        for (int out = tid; out < MM2*MM2; out += stride) {
            int r = out / MM2, c = out % MM2;
            float acc = 0.f;
            for (int k = 0; k < MM2; k++) acc += A[r*MM2+k] * B[k*MM2+c];
            C[out] = acc + (float)it;
        }
        for (int j = tid; j < n; j += stride) {
            volatile int v = data[indices[j]]; data[indices[j]] = v + 1;
        }
    }
}

// W3: simulated conv2d (3x3 kernel, 64x64 input, 32 channels) fused with eviction
#define CONV_H 64
#define CONV_W 64
#define CONV_C 32
__global__ void gpu_conv2d_evict(int* data, const int* indices, int n,
                                 const float* input, const float* kernel, float* output,
                                 int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = CONV_H * CONV_W * CONV_C;
    for (int it = 0; it < iters; it++) {
        // 3x3 conv simulation
        for (int out = tid; out < total; out += stride) {
            int c = out % CONV_C;
            int hw = out / CONV_C;
            int h = hw / CONV_W, w = hw % CONV_W;
            float acc = 0.f;
            for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++) {
                    int ih = h + kh - 1, iw = w + kw - 1;
                    if (ih >= 0 && ih < CONV_H && iw >= 0 && iw < CONV_W)
                        acc += input[(ih*CONV_W+iw)*CONV_C+c] * kernel[(kh*3+kw)*CONV_C+c];
                }
            output[out] = acc + (float)it;
        }
        for (int j = tid; j < n; j += stride) {
            volatile int v = data[indices[j]]; data[indices[j]] = v + 1;
        }
    }
}

// W4: large bandwidth (simulates attention: reads big arrays) fused with eviction
#define BW_SIZE (4 * 1024 * 1024)   // 4M floats = 16MB
__global__ void gpu_bandwidth_evict(int* data, const int* indices, int n,
                                    const float* big_arr, float* out_arr, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int it = 0; it < iters; it++) {
        float acc = 0.f;
        for (int i = tid; i < BW_SIZE; i += stride) {
            acc += big_arr[i];
            if (i % 4096 == 0) out_arr[i/4096] = acc;
        }
        for (int j = tid; j < n; j += stride) {
            volatile int v = data[indices[j]]; data[indices[j]] = v + 1;
        }
    }
}

// ----------------------------------------------------------------
// CPU workload helpers (called between timing probes)
// ----------------------------------------------------------------
static float cpu_mat_a[256*256], cpu_mat_b[256*256], cpu_mat_c[256*256];
static volatile float cpu_sink = 0.f;

static void cpu_work_none() {}

static void cpu_work_matmul64(int seed) {
    int N = 64, bias = seed % N;
    float s = 0.f;
    for (int i = 0; i < N; i++) {
        int r = (i + bias) % N;
        for (int j = 0; j < N; j++) {
            float acc = 0.f;
            for (int k = 0; k < N; k++)
                acc += cpu_mat_a[r*N+k] * cpu_mat_b[k*N+j];
            cpu_mat_c[r*N+j] = acc; s += acc;
        }
    }
    cpu_sink += s;
}

static void cpu_work_matmul256(int seed) {
    int N = 256, bias = seed % N;
    float s = 0.f;
    for (int i = 0; i < N; i++) {
        int r = (i + bias) % N;
        for (int j = 0; j < N; j++) {
            float acc = 0.f;
            for (int k = 0; k < N; k++)
                acc += cpu_mat_a[r*N+k] * cpu_mat_b[k*N+j];
            cpu_mat_c[r*N+j] = acc; s += acc;
        }
    }
    cpu_sink += s;
}

#define FFT_N 8192
static float fft_arr[FFT_N];
static void cpu_work_fft_like(int seed) {
    float s = 0.f;
    for (int k = 0; k < 256; k++) {
        float re = 0.f, im = 0.f;
        float freq = 2.f * 3.14159f * k / FFT_N;
        for (int n = 0; n < FFT_N; n++) {
            re += fft_arr[n] * cosf(freq * n);
            im += fft_arr[n] * sinf(freq * n);
        }
        s += sqrtf(re*re + im*im);
    }
    cpu_sink += s;
}

// ----------------------------------------------------------------
// Stats
// ----------------------------------------------------------------
struct Stats { uint64_t median, p99, maxv; };

static Stats compute_stats(std::vector<uint64_t>& s) {
    std::sort(s.begin(), s.end());
    int n = (int)s.size();
    return {s[n/2], s[(int)(0.99*(n-1))], s[n-1]};
}

// ----------------------------------------------------------------
// Congruent index collection
// ----------------------------------------------------------------
static int collect_congruent(volatile int* data, int array_size, uint64_t target_set,
                              int cache_kb, int ways, int* out, int max_out) {
    int count = 0;
    for (int i = 0; i < array_size; i += CACHELINE_INTS) {
        uint64_t pa = 0;
        if (virtual_to_physical((void*)&data[i], &pa) != 0) continue;
        if (get_cache_set_index_from_phys(pa, cache_kb, ways) == (uint64_t)target_set
            && count < max_out)
            out[count++] = i;
    }
    return count;
}

// ----------------------------------------------------------------
// Main
// ----------------------------------------------------------------
int main(int argc, char** argv) {
    int target_set  = (argc > 1) ? atoi(argv[1]) : 0;
    int points      = (argc > 2) ? atoi(argv[2]) : 64;
    int reps        = (argc > 3) ? atoi(argv[3]) : 64;
    int gpu_iters   = (argc > 4) ? atoi(argv[4]) : 5000;
    int cache_kb    = (argc > 5) ? atoi(argv[5]) : DEFAULT_LLC_KB;
    int ways        = (argc > 6) ? atoi(argv[6]) : DEFAULT_WAYS;
    int gpu_wl      = (argc > 7) ? atoi(argv[7]) : 0;
    int cpu_wl      = (argc > 8) ? atoi(argv[8]) : 0;

    const char* gpu_names[] = {"none","matmul_128","matmul_512","conv2d","bandwidth"};
    const char* cpu_names[] = {"none","matmul_64","matmul_256","fft_like"};

    printf("=== EVSET WORKLOAD SCAN ===\n");
    printf("Set=%d  Points=%d  Reps=%d  GPUiters=%d\n", target_set, points, reps, gpu_iters);
    printf("GPU workload: %s  CPU workload: %s\n\n",
           gpu_names[gpu_wl % 5], cpu_names[cpu_wl % 4]);

    // init CPU work arrays
    for (int i = 0; i < 256*256; i++) {
        cpu_mat_a[i] = 1.f + (float)(i % 7);
        cpu_mat_b[i] = 0.5f + (float)(i % 5);
    }
    for (int i = 0; i < FFT_N; i++) fft_arr[i] = sinf(0.01f * i) + 0.1f * (i % 13);

    // main managed array
    int* data = NULL;
    CUDA_CHECK(cudaMallocManaged(&data, ARRAY_SIZE * sizeof(int)));
    for (int i = 0; i < ARRAY_SIZE; i++) data[i] = i;
    CUDA_CHECK(cudaMemPrefetchAsync(data, ARRAY_SIZE * sizeof(int), cudaCpuDeviceId));
    CUDA_CHECK(cudaDeviceSynchronize());

    // collect congruent indices
    int* idx_host = (int*)malloc(MAX_CONGRUENT * sizeof(int));
    int found = collect_congruent(data, ARRAY_SIZE, (uint64_t)target_set,
                                  cache_kb, ways, idx_host, MAX_CONGRUENT);
    if (found < ways + 2) {
        fprintf(stderr, "Not enough congruent lines: %d\n", found);
        cudaFree(data); free(idx_host); return 1;
    }
    int n = std::min(points, found);
    printf("Congruent found: %d, using %d\n", found, n);

    int* idx_dev = NULL;
    CUDA_CHECK(cudaMallocManaged(&idx_dev, n * sizeof(int)));
    for (int i = 0; i < n; i++) idx_dev[i] = idx_host[i];
    CUDA_CHECK(cudaMemPrefetchAsync(idx_dev, n * sizeof(int), 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    // allocate GPU workload buffers
    float *gA=NULL, *gB=NULL, *gC=NULL, *gBig=NULL, *gBigOut=NULL;
    float *gConv_in=NULL, *gConv_ker=NULL, *gConv_out=NULL;
    size_t mm1 = MM1*MM1*sizeof(float), mm2 = MM2*MM2*sizeof(float);
    size_t conv_in  = CONV_H*CONV_W*CONV_C*sizeof(float);
    size_t conv_ker = 3*3*CONV_C*sizeof(float);
    size_t conv_out = CONV_H*CONV_W*CONV_C*sizeof(float);
    size_t bw_sz    = BW_SIZE*sizeof(float);

    if (gpu_wl == 1) {
        CUDA_CHECK(cudaMalloc(&gA, mm1)); CUDA_CHECK(cudaMalloc(&gB, mm1));
        CUDA_CHECK(cudaMalloc(&gC, mm1)); CUDA_CHECK(cudaMemset(gA,1,mm1));
        CUDA_CHECK(cudaMemset(gB,1,mm1)); CUDA_CHECK(cudaMemset(gC,0,mm1));
    } else if (gpu_wl == 2) {
        CUDA_CHECK(cudaMalloc(&gA, mm2)); CUDA_CHECK(cudaMalloc(&gB, mm2));
        CUDA_CHECK(cudaMalloc(&gC, mm2)); CUDA_CHECK(cudaMemset(gA,1,mm2));
        CUDA_CHECK(cudaMemset(gB,1,mm2)); CUDA_CHECK(cudaMemset(gC,0,mm2));
    } else if (gpu_wl == 3) {
        CUDA_CHECK(cudaMalloc(&gConv_in,  conv_in));
        CUDA_CHECK(cudaMalloc(&gConv_ker, conv_ker));
        CUDA_CHECK(cudaMalloc(&gConv_out, conv_out));
        CUDA_CHECK(cudaMemset(gConv_in,1,conv_in));
        CUDA_CHECK(cudaMemset(gConv_ker,1,conv_ker));
        CUDA_CHECK(cudaMemset(gConv_out,0,conv_out));
    } else if (gpu_wl == 4) {
        CUDA_CHECK(cudaMalloc(&gBig, bw_sz));
        CUDA_CHECK(cudaMalloc(&gBigOut, (BW_SIZE/4096)*sizeof(float)));
        CUDA_CHECK(cudaMemset(gBig,1,bw_sz));
    }

    // ---- BASELINE: measure without GPU kernel ----
    std::vector<Stats> base_stats(n);
    for (int i = 0; i < n; i++) {
        std::vector<uint64_t> s; s.reserve(reps);
        for (int r = 0; r < reps; r++) {
            // CPU workload before probe
            if      (cpu_wl == 1) cpu_work_matmul64(r);
            else if (cpu_wl == 2) cpu_work_matmul256(r);
            else if (cpu_wl == 3) cpu_work_fft_like(r);

            cpu_barrier();
            uint64_t t0 = rdtsc();
            volatile int v = data[idx_host[i]];
            load_barrier();
            uint64_t t1 = rdtsc();
            cpu_barrier();
            (void)v;
            s.push_back(t1 - t0);
        }
        base_stats[i] = compute_stats(s);
    }

    // ---- ATTACK: launch GPU kernel, measure concurrently ----
    int blocks = 8, threads = 256;
    switch (gpu_wl) {
        case 0: gpu_evict_only<<<blocks,threads>>>(data, idx_dev, n, gpu_iters); break;
        case 1: gpu_matmul128_evict<<<blocks,threads>>>(data, idx_dev, n, gA, gB, gC, gpu_iters); break;
        case 2: gpu_matmul512_evict<<<blocks,threads>>>(data, idx_dev, n, gA, gB, gC, gpu_iters); break;
        case 3: gpu_conv2d_evict<<<blocks,threads>>>(data, idx_dev, n, gConv_in, gConv_ker, gConv_out, gpu_iters); break;
        case 4: gpu_bandwidth_evict<<<blocks,threads>>>(data, idx_dev, n, gBig, gBigOut, gpu_iters); break;
    }
    CUDA_CHECK(cudaGetLastError());

    std::vector<Stats> atk_stats(n);
    for (int i = 0; i < n; i++) {
        std::vector<uint64_t> s; s.reserve(reps);
        for (int r = 0; r < reps; r++) {
            if      (cpu_wl == 1) cpu_work_matmul64(r);
            else if (cpu_wl == 2) cpu_work_matmul256(r);
            else if (cpu_wl == 3) cpu_work_fft_like(r);

            cpu_barrier();
            uint64_t t0 = rdtsc();
            volatile int v = data[idx_host[i]];
            load_barrier();
            uint64_t t1 = rdtsc();
            cpu_barrier();
            (void)v;
            s.push_back(t1 - t0);
        }
        atk_stats[i] = compute_stats(s);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Write CSV ----
    char out_csv[256];
    snprintf(out_csv, sizeof(out_csv),
             "results/workload_scan_set%d_gpu%d_cpu%d.csv", target_set, gpu_wl, cpu_wl);
    FILE* fp = fopen(out_csv, "w");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", out_csv); return 1; }
    fprintf(fp, "order,index,gpu_workload,cpu_workload,"
                "baseline_median,attack_median,median_ratio,"
                "baseline_p99,attack_p99,p99_ratio,"
                "baseline_max,attack_max,max_ratio\n");
    for (int i = 0; i < n; i++) {
        auto& b = base_stats[i]; auto& a = atk_stats[i];
        fprintf(fp, "%d,%d,%s,%s,%llu,%llu,%.6f,%llu,%llu,%.6f,%llu,%llu,%.6f\n",
                i, idx_host[i],
                gpu_names[gpu_wl%5], cpu_names[cpu_wl%4],
                (unsigned long long)b.median, (unsigned long long)a.median,
                b.median>0 ? (double)a.median/b.median : 0.0,
                (unsigned long long)b.p99,    (unsigned long long)a.p99,
                b.p99>0   ? (double)a.p99/b.p99     : 0.0,
                (unsigned long long)b.maxv,   (unsigned long long)a.maxv,
                b.maxv>0  ? (double)a.maxv/b.maxv   : 0.0);
    }
    fclose(fp);
    printf("Saved: %s\n", out_csv);

    // cleanup
    if (gA) cudaFree(gA); if (gB) cudaFree(gB); if (gC) cudaFree(gC);
    if (gConv_in) cudaFree(gConv_in); if (gConv_ker) cudaFree(gConv_ker);
    if (gConv_out) cudaFree(gConv_out);
    if (gBig) cudaFree(gBig); if (gBigOut) cudaFree(gBigOut);
    cudaFree(idx_dev); cudaFree(data); free(idx_host);
    return 0;
}
