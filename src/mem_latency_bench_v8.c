/*
 * mem_latency_bench.c  (v8)
 * - Full-chain traversal for accurate cache hierarchy latency.
 * - DRAM: clflush entire chain after build, then cold-traverse.
 * Compile: gcc -O2 -o bin/mem_latency_bench src/mem_latency_bench_v8.c
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <x86intrin.h>

#define SAMPLES_CACHE  50
#define SAMPLES_DRAM   30
#define SAMPLES_PM    200

static inline uint64_t rdtsc_fence() {
    uint32_t lo, hi;
    __asm__ __volatile__("mfence\nlfence\nrdtsc\nlfence"
                         : "=a"(lo), "=d"(hi) :: "memory");
    return ((uint64_t)hi << 32) | lo;
}
static inline void clflush_line(void* p) {
    __asm__ __volatile__("clflush (%0)" :: "r"(p) : "memory");
}

static void shuffle(size_t* a, size_t n) {
    for (size_t i = n-1; i > 0; i--) {
        size_t j = (size_t)rand() % (i+1);
        size_t t = a[i]; a[i] = a[j]; a[j] = t;
    }
}

static void build_random_chain(uint8_t* buf, size_t region, size_t stride) {
    size_t n = region / stride;
    size_t* perm = (size_t*)malloc(n * sizeof(size_t));
    if (!perm) { perror("malloc"); exit(1); }
    for (size_t i = 0; i < n; i++) perm[i] = i;
    shuffle(perm, n);
    for (size_t i = 0; i < n; i++)
        *(void**)(buf + perm[i]*stride) = (void*)(buf + perm[(i+1)%n]*stride);
    free(perm);
}

static void flush_region(uint8_t* buf, size_t bytes) {
    for (size_t i = 0; i < bytes; i += 64)
        clflush_line(buf + i);
    __asm__ __volatile__("mfence" ::: "memory");
}

/*
 * Traverse n_nodes hops.
 * warm=1: one warm pass first (L1/L2/LLC measurement).
 * warm=0: cold measurement (DRAM; caller must have flushed chain).
 */
static uint64_t traverse_avg(void* start, size_t n_nodes, int warm) {
    volatile void** p = (volatile void**)start;
    if (warm) {
        for (size_t i = 0; i < n_nodes; i++) p = (volatile void**)*p;
        p = (volatile void**)start;
    }
    __asm__ __volatile__("mfence\nlfence" ::: "memory");
    uint64_t t0 = rdtsc_fence();
    for (size_t i = 0; i < n_nodes; i++) p = (volatile void**)*p;
    uint64_t t1 = rdtsc_fence();
    __asm__ __volatile__("" :: "r"(p) : "memory");
    return (t1 - t0) / n_nodes;
}

static uint64_t median(uint64_t* a, int n) {
    for (int i = 1; i < n; i++) {
        uint64_t k = a[i]; int j = i-1;
        while (j >= 0 && a[j] > k) { a[j+1] = a[j]; j--; }
        a[j+1] = k;
    }
    return a[n/2];
}

int main(void) {
    srand((unsigned)time(NULL));
    printf("=== Memory Hierarchy Latency Benchmark ===\n\n");

    const size_t STRIDE  = 64;
    const size_t L1_SZ   = 16  * 1024;
    const size_t L2_SZ   = 384 * 1024;
    const size_t LLC_SZ  = 8   * 1024 * 1024UL;
    const size_t DRAM_SZ = 14  * 1024 * 1024UL;
    const size_t BUF_SZ  = 64  * 1024 * 1024UL;

    uint8_t* buf = (uint8_t*)aligned_alloc(64, BUF_SZ);
    if (!buf) { perror("aligned_alloc"); return 1; }
    memset(buf, 0, BUF_SZ); /* touch all pages */

    uint64_t samp[SAMPLES_CACHE > SAMPLES_DRAM ? SAMPLES_CACHE : SAMPLES_DRAM];
    size_t n_nodes;

    /* ── L1 ── */
    n_nodes = L1_SZ / STRIDE;
    build_random_chain(buf, L1_SZ, STRIDE);
    for (int s = 0; s < SAMPLES_CACHE; s++)
        samp[s] = traverse_avg((void*)buf, n_nodes, 1);
    uint64_t l1_m = median(samp, SAMPLES_CACHE);
    printf("CPU L1 cache:    %4llu cycles/access\n", (unsigned long long)l1_m);

    /* ── L2 ── */
    n_nodes = L2_SZ / STRIDE;
    build_random_chain(buf, L2_SZ, STRIDE);
    for (int s = 0; s < SAMPLES_CACHE; s++)
        samp[s] = traverse_avg((void*)buf, n_nodes, 1);
    uint64_t l2_m = median(samp, SAMPLES_CACHE);
    printf("CPU L2 cache:    %4llu cycles/access\n", (unsigned long long)l2_m);

    /* ── LLC ── */
    n_nodes = LLC_SZ / STRIDE;
    build_random_chain(buf, LLC_SZ, STRIDE);
    for (int s = 0; s < SAMPLES_CACHE; s++)
        samp[s] = traverse_avg((void*)buf, n_nodes, 1);
    uint64_t llc_m = median(samp, SAMPLES_CACHE);
    printf("CPU LLC:         %4llu cycles/access\n", (unsigned long long)llc_m);

    /* ── DRAM ──
     * Strategy: use a separate region (beyond LLC region) for DRAM chain.
     * Build once, then each sample: flush all chain lines → cold traverse.
     */
    {
        uint8_t* dram_buf = buf + (BUF_SZ - DRAM_SZ); /* offset to separate region */
        n_nodes = DRAM_SZ / STRIDE;
        build_random_chain(dram_buf, DRAM_SZ, STRIDE);

        for (int s = 0; s < SAMPLES_DRAM; s++) {
            /* flush entire chain region to DRAM */
            flush_region(dram_buf, DRAM_SZ);
            /* now cold traverse */
            samp[s] = traverse_avg((void*)dram_buf, n_nodes, 0);
        }
        uint64_t dram_m = median(samp, SAMPLES_DRAM);
        printf("CPU DRAM:        %4llu cycles/access\n", (unsigned long long)dram_m);

        /* ── Pagemap ── */
        uint64_t pm_m = 0;
        {
            int fd = open("/proc/self/pagemap", O_RDONLY);
            if (fd < 0) {
                perror("/proc/self/pagemap");
            } else {
                long pgsz = sysconf(_SC_PAGESIZE);
                uint64_t vpn = (uint64_t)buf / (uint64_t)pgsz;
                uint64_t entry, pm_samp[SAMPLES_PM];
                for (int s = 0; s < SAMPLES_PM; s++) {
                    uint64_t t0 = rdtsc_fence();
                    (void)pread(fd, &entry, 8, (off_t)(vpn * 8));
                    uint64_t t1 = rdtsc_fence();
                    pm_samp[s] = t1 - t0;
                }
                close(fd);
                pm_m = median(pm_samp, SAMPLES_PM);
                printf("Pagemap:         %4llu cycles  (page-table read via /proc)\n",
                       (unsigned long long)pm_m);
            }
        }

        /* ── CSV ── */
        FILE* fp = fopen("results/mem_latency.csv", "w");
        if (!fp) {
            fprintf(stderr, "Warning: cannot write results/mem_latency.csv\n");
        } else {
            fprintf(fp, "level,median_cycles,type\n");
            fprintf(fp, "CPU L1,%llu,cpu\n",      (unsigned long long)l1_m);
            fprintf(fp, "CPU L2,%llu,cpu\n",      (unsigned long long)l2_m);
            fprintf(fp, "CPU LLC,%llu,cpu\n",     (unsigned long long)llc_m);
            fprintf(fp, "CPU DRAM,%llu,cpu\n",    (unsigned long long)dram_m);
            fprintf(fp, "Pagemap,%llu,pagemap\n", (unsigned long long)pm_m);
            fclose(fp);
            printf("\nSaved: results/mem_latency.csv\n");
        }
    }

    free(buf);
    return 0;
}
