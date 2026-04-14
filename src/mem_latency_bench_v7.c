/*
 * mem_latency_bench.c  (v7)
 * CORRECT approach: traverse the ENTIRE chain for each level.
 * This ensures working-set pressure is exactly the target cache level.
 * - L1  (16 KB, 256 nodes)
 * - L2  (384 KB, 6144 nodes)
 * - LLC (8 MB, 131072 nodes)
 * - DRAM(14 MB, first cold traversal)
 *
 * Compile: gcc -O2 -o bin/mem_latency_bench src/mem_latency_bench_v7.c
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

#define SAMPLES  50     /* enough for stable median; LLC/DRAM are slow per sample */

static inline uint64_t rdtsc_fence() {
    uint32_t lo, hi;
    __asm__ __volatile__("mfence\nlfence\nrdtsc\nlfence"
                         : "=a"(lo), "=d"(hi) :: "memory");
    return ((uint64_t)hi << 32) | lo;
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
    if (!perm) { perror("malloc perm"); exit(1); }
    for (size_t i = 0; i < n; i++) perm[i] = i;
    shuffle(perm, n);
    for (size_t i = 0; i < n; i++)
        *(void**)(buf + perm[i]*stride) = (void*)(buf + perm[(i+1)%n]*stride);
    free(perm);
}

/* Traverse ALL n_nodes hops, return (total_cycles / n_nodes). */
static uint64_t traverse_full_avg(void* start, size_t n_nodes, int cold) {
    volatile void** p = (volatile void**)start;
    if (!cold) {
        /* warm: one full pass to bring chain into the target cache level */
        for (size_t i = 0; i < n_nodes; i++) p = (volatile void**)*p;
        p = (volatile void**)start; /* restart from same head */
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
    printf("=== Memory Hierarchy Latency Benchmark (v7) ===\n\n");

    const size_t STRIDE  = 64;
    const size_t L1_SZ   = 16  * 1024;
    const size_t L2_SZ   = 384 * 1024;
    const size_t LLC_SZ  = 8   * 1024 * 1024UL;
    const size_t DRAM_SZ = 14  * 1024 * 1024UL;  /* > 12 MB LLC */
    const size_t BUF_SZ  = 64  * 1024 * 1024UL;

    uint8_t* buf = (uint8_t*)aligned_alloc(64, BUF_SZ);
    if (!buf) { perror("aligned_alloc"); return 1; }
    memset(buf, 0, BUF_SZ);

    uint64_t samp[SAMPLES];
    size_t n_nodes;

    /* ── L1 ── */
    n_nodes = L1_SZ / STRIDE;
    build_random_chain(buf, L1_SZ, STRIDE);
    for (int s = 0; s < SAMPLES; s++)
        samp[s] = traverse_full_avg((void*)buf, n_nodes, /*cold=*/0);
    uint64_t l1_m = median(samp, SAMPLES);
    printf("CPU L1 cache:    %4llu cycles/access  (%zu nodes)\n",
           (unsigned long long)l1_m, n_nodes);

    /* ── L2 ── */
    n_nodes = L2_SZ / STRIDE;
    build_random_chain(buf, L2_SZ, STRIDE);
    for (int s = 0; s < SAMPLES; s++)
        samp[s] = traverse_full_avg((void*)buf, n_nodes, /*cold=*/0);
    uint64_t l2_m = median(samp, SAMPLES);
    printf("CPU L2 cache:    %4llu cycles/access  (%zu nodes)\n",
           (unsigned long long)l2_m, n_nodes);

    /* ── LLC ── */
    n_nodes = LLC_SZ / STRIDE;
    build_random_chain(buf, LLC_SZ, STRIDE);
    for (int s = 0; s < SAMPLES; s++)
        samp[s] = traverse_full_avg((void*)buf, n_nodes, /*cold=*/0);
    uint64_t llc_m = median(samp, SAMPLES);
    printf("CPU LLC:         %4llu cycles/access  (%zu nodes)\n",
           (unsigned long long)llc_m, n_nodes);

    /* ── DRAM ── */
    /* Cold traversal of >LLC region; first touch = DRAM. */
    n_nodes = DRAM_SZ / STRIDE;
    /* Build in a fresh region (offset past LLC region to avoid overlap) */
    uint8_t* dram_buf = buf + (BUF_SZ - DRAM_SZ);
    build_random_chain(dram_buf, DRAM_SZ, STRIDE);
    for (int s = 0; s < SAMPLES; s++) {
        /* Rebuild chain each sample to ensure cold access
         * (rebuilding touches new memory and breaks prefetcher pattern) */
        build_random_chain(dram_buf, DRAM_SZ, STRIDE);
        samp[s] = traverse_full_avg((void*)dram_buf, n_nodes, /*cold=*/1);
    }
    uint64_t dram_m = median(samp, SAMPLES);
    printf("CPU DRAM:        %4llu cycles/access  (%zu nodes)\n",
           (unsigned long long)dram_m, n_nodes);

    /* ── Pagemap ── */
    uint64_t pm_m = 0;
    {
        int fd = open("/proc/self/pagemap", O_RDONLY);
        if (fd < 0) {
            perror("open /proc/self/pagemap");
        } else {
            long pgsz = sysconf(_SC_PAGESIZE);
            uint64_t vpn = (uint64_t)buf / (uint64_t)pgsz;
            uint64_t entry;
            uint64_t pm_samp[200];
            for (int s = 0; s < 200; s++) {
                uint64_t t0 = rdtsc_fence();
                (void)pread(fd, &entry, 8, (off_t)(vpn * 8));
                uint64_t t1 = rdtsc_fence();
                pm_samp[s] = t1 - t0;
            }
            close(fd);
            pm_m = median(pm_samp, 200);
            printf("Pagemap:         %4llu cycles  (software page-table read)\n",
                   (unsigned long long)pm_m);
        }
    }

    /* ── Save CSV ── */
    FILE* fp = fopen("results/mem_latency.csv", "w");
    if (!fp) {
        fprintf(stderr, "Warning: cannot open results/mem_latency.csv\n");
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

    free(buf);
    return 0;
}
