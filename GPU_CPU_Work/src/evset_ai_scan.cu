/*
 * evset_ai_scan.cu
 *
 * LLC eviction-set heatmap scanner with a REAL AI model as the GPU workload.
 * The AI model (.pt TorchScript) is loaded and executed via LibTorch C++ API
 * on one CUDA stream, while the cache-eviction kernel runs on another stream.
 * Both share the same GPU process — identical to evset_workload_scan.cu,
 * but with a real model instead of a synthetic kernel.
 *
 * GPU stream 0: AI model inference (LibTorch forward pass, looped)
 * GPU stream 1: targeted LLC eviction kernel on congruent lines
 * CPU main   : rdtsc timing probes measuring LLC miss latency
 *
 * Usage:
 *   evset_ai_scan <model.pt> <target_set> <points> <reps>
 *                 <gpu_iters> <model_iters> <cache_kb> <ways>
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
#include <math.h>
#include <algorithm>
#include <vector>
#include <thread>
#include <atomic>

#define ARRAY_SIZE      (16 * 1024 * 1024)   // 64 MB managed
#define CACHELINE_INTS  16
#define MAX_CONGRUENT   8192
#define DEFAULT_LLC_KB  (12 * 1024)
#define DEFAULT_WAYS    16

// ── timing ───────────────────────────────────────────────────
static inline uint64_t rdtsc() {
    uint32_t lo, hi;
    __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi) :: "%rcx");
    return ((uint64_t)hi << 32) | lo;
}
static inline void cpu_barrier()  { __asm__ __volatile__("mfence\nlfence" ::: "memory"); }
static inline void load_barrier() { __asm__ __volatile__("lfence" ::: "memory"); }

// ── physical address → LLC set ────────────────────────────────
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

static uint64_t llc_set(void* addr, int cache_kb, int ways) {
    uint64_t pa = 0;
    if (vtop(addr, &pa) == 0)
        return (pa / 64) % ((uint64_t)(cache_kb * 1024) / (64 * ways));
    return ((uint64_t)addr / 64) % ((uint64_t)(cache_kb * 1024) / (64 * ways));
}

// ── GPU eviction kernel (stream 1) ───────────────────────────
__global__ void evict_kernel(int* data, const int* idx, int n, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int it = 0; it < iters; it++)
        for (int j = tid; j < n; j += stride) {
            volatile int v = data[idx[j]];
            data[idx[j]] = v + 1;
        }
}

// ── stats ─────────────────────────────────────────────────────
struct Stats { uint64_t med, p99, maxv; };
static Stats compute(std::vector<uint64_t>& s) {
    std::sort(s.begin(), s.end());
    int n = s.size();
    return { s[n/2], s[(int)(0.99*(n-1))], s[n-1] };
}

// ── model inference thread ────────────────────────────────────
// Runs LibTorch forward pass in a loop on stream 0 while main thread probes.
static std::atomic<bool> g_run_model{false};
static std::atomic<int>  g_model_iters{0};

void model_thread_fn(torch::jit::script::Module* model,
                     std::vector<torch::jit::IValue> inputs) {
    // Model runs on PyTorch's default CUDA stream for this thread.
    // The eviction kernel runs on a separate explicit stream — both
    // run concurrently on the GPU.
    while (g_run_model.load()) {
        torch::NoGradGuard ng;
        auto out = model->forward(inputs);
        g_model_iters.fetch_add(1);
    }
}

// ── main ──────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <model.pt> <target_set> [points] [reps] "
            "[gpu_iters] [model_iters] [cache_kb] [ways]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    int target_set  = atoi(argv[2]);
    int points      = (argc > 3) ? atoi(argv[3]) : 64;
    int reps        = (argc > 4) ? atoi(argv[4]) : 64;
    int gpu_iters   = (argc > 5) ? atoi(argv[5]) : 10000;
    int model_iters = (argc > 6) ? atoi(argv[6]) : 0;  // 0 = run until done
    int cache_kb    = (argc > 7) ? atoi(argv[7]) : DEFAULT_LLC_KB;
    int ways        = (argc > 8) ? atoi(argv[8]) : DEFAULT_WAYS;

    printf("=== EVSET AI SCAN ===\n");
    printf("Model: %s\nSet: %d  Points: %d  Reps: %d  GPU_iters: %d\n\n",
           model_path, target_set, points, reps, gpu_iters);

    // ── load model ──
    printf("[Model] Loading %s ...\n", model_path);
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path, torch::kCUDA);
        model.eval();
    } catch (const c10::Error& e) {
        fprintf(stderr, "Failed to load model: %s\n", e.what());
        return 1;
    }
    printf("[Model] Loaded OK\n");

    // ── build dummy input matching each model ──
    std::string mpath(model_path);
    std::vector<torch::jit::IValue> inputs;

    if (mpath.find("distilbert") != std::string::npos) {
        // DistilBERT: (input_ids, attention_mask) — int64
        auto ids  = torch::randint(0, 30522, {2, 128},
                        torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
        auto mask = torch::ones({2, 128},
                        torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
        inputs.push_back(ids);
        inputs.push_back(mask);
    } else if (mpath.find("minitransformer") != std::string::npos) {
        // MiniTransformer: single input_ids tensor — int64
        auto ids = torch::randint(0, 30522, {2, 128},
                        torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
        inputs.push_back(ids);
    } else if (mpath.find("tinyyolo") != std::string::npos) {
        inputs.push_back(torch::randn({2, 3, 416, 416},
                         torch::TensorOptions().device(torch::kCUDA)));
    } else {
        // MobileNet, SqueezeNet, EfficientNet: batch=4, 224x224 RGB
        inputs.push_back(torch::randn({4, 3, 224, 224},
                         torch::TensorOptions().device(torch::kCUDA)));
    }

    // ── warmup model ──
    printf("[Model] Warming up...\n");
    {
        torch::NoGradGuard ng;
        for (int i = 0; i < 5; i++) model.forward(inputs);
        cudaDeviceSynchronize();
    }
    printf("[Model] Warmup done\n\n");

    // ── create CUDA streams ──
    cudaStream_t evict_stream;
    cudaStreamCreate(&evict_stream);

    // ── allocate managed memory ──
    int* data = nullptr;
    cudaMallocManaged(&data, ARRAY_SIZE * sizeof(int));
    for (int i = 0; i < ARRAY_SIZE; i++) data[i] = i;
    cudaMemPrefetchAsync(data, ARRAY_SIZE * sizeof(int), cudaCpuDeviceId, 0);
    cudaDeviceSynchronize();

    // ── collect congruent indices ──
    printf("[Evict] Scanning for congruent lines (set %d)...\n", target_set);
    std::vector<int> cong;
    for (int i = 0; i < ARRAY_SIZE; i += CACHELINE_INTS) {
        if ((int)llc_set((void*)&data[i], cache_kb, ways) == target_set) {
            cong.push_back(i);
            if ((int)cong.size() >= MAX_CONGRUENT) break;
        }
    }
    if ((int)cong.size() < ways + 2) {
        fprintf(stderr, "Not enough congruent lines: %d\n", (int)cong.size());
        cudaFree(data); return 1;
    }
    int n_used = std::min(points, (int)cong.size());
    printf("[Evict] Found %d congruent lines, using %d\n\n", (int)cong.size(), n_used);

    // copy indices to managed memory (GPU-accessible)
    int* idx_dev = nullptr;
    cudaMallocManaged(&idx_dev, n_used * sizeof(int));
    for (int i = 0; i < n_used; i++) idx_dev[i] = cong[i];
    cudaMemPrefetchAsync(idx_dev, n_used * sizeof(int), 0, 0); // prefetch to GPU
    cudaDeviceSynchronize();

    // ── BASELINE: measure without model or eviction ──
    printf("[Measure] Baseline...\n");
    std::vector<Stats> base(n_used);
    for (int i = 0; i < n_used; i++) {
        std::vector<uint64_t> s; s.reserve(reps);
        for (int r = 0; r < reps; r++) {
            cpu_barrier();
            uint64_t t0 = rdtsc();
            volatile int v = data[cong[i]];
            load_barrier();
            uint64_t t1 = rdtsc();
            cpu_barrier(); (void)v;
            s.push_back(t1 - t0);
        }
        base[i] = compute(s);
    }

    // ── ATTACK: start model on stream 0, eviction on stream 1, probe on CPU ──
    printf("[Attack] Launching model inference thread + eviction kernel...\n");

    // start model inference thread
    g_run_model.store(true);
    g_model_iters.store(0);
    std::thread model_thr(model_thread_fn, &model, inputs);

    // launch eviction kernel on evict_stream (async)
    evict_kernel<<<8, 256, 0, evict_stream>>>(data, idx_dev, n_used, gpu_iters);

    // small delay so both GPU workloads are running before we start probing
    usleep(50000);   // 50 ms

    std::vector<Stats> atk(n_used);
    for (int i = 0; i < n_used; i++) {
        std::vector<uint64_t> s; s.reserve(reps);
        for (int r = 0; r < reps; r++) {
            cpu_barrier();
            uint64_t t0 = rdtsc();
            volatile int v = data[cong[i]];
            load_barrier();
            uint64_t t1 = rdtsc();
            cpu_barrier(); (void)v;
            s.push_back(t1 - t0);
        }
        atk[i] = compute(s);
    }

    // stop model thread, wait for eviction kernel
    g_run_model.store(false);
    model_thr.join();
    cudaStreamSynchronize(evict_stream);

    printf("[Attack] Model ran %d inference iters\n", g_model_iters.load());
    printf("[Attack] Done\n\n");

    // ── write CSV ──
    // derive model name from path
    std::string mp(model_path);
    std::string mname = mp.substr(mp.rfind('/')+1);
    if (mname.size() > 3) mname = mname.substr(0, mname.size()-3); // strip .pt

    char out_csv[512];
    snprintf(out_csv, sizeof(out_csv),
             "results/ai_scan_%s_set%d.csv", mname.c_str(), target_set);
    FILE* fp = fopen(out_csv, "w");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", out_csv); return 1; }

    fprintf(fp, "order,index,model,set_id,"
                "baseline_median,attack_median,median_ratio,"
                "baseline_p99,attack_p99,p99_ratio,"
                "baseline_max,attack_max,max_ratio\n");
    for (int i = 0; i < n_used; i++) {
        auto& b = base[i]; auto& a = atk[i];
        auto ratio = [](uint64_t num, uint64_t den) -> double {
            return den > 0 ? (double)num / den : 0.0;
        };
        fprintf(fp, "%d,%d,%s,%d,%llu,%llu,%.6f,%llu,%llu,%.6f,%llu,%llu,%.6f\n",
                i, cong[i], mname.c_str(), target_set,
                (unsigned long long)b.med,  (unsigned long long)a.med,  ratio(a.med,  b.med),
                (unsigned long long)b.p99,  (unsigned long long)a.p99,  ratio(a.p99,  b.p99),
                (unsigned long long)b.maxv, (unsigned long long)a.maxv, ratio(a.maxv, b.maxv));
    }
    fclose(fp);
    printf("Saved: %s\n", out_csv);

    // ── cleanup ──
    cudaStreamDestroy(evict_stream);
    cudaFree(idx_dev);
    cudaFree(data);
    return 0;
}
