/*
 * mem_latency_bench.c  (v6)
 * Random pointer chaining.
 * - L1/L2/LLC: build chain in region that fits that cache level.
 *   Warm the chain first, then time subsequent traversals.
 * - DRAM: use random chain > LLC (14 MB), time FIRST traversal (cold from DRAM).
 * - Pagemap: time pread() syscall.
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

#define SAMPLES  200
#define HOPS     512

static inline uint64_t rdtsc_fence() {
    uint32_t lo, hi;
    __asm__ __volatile__("mfence\nlfence\nrdtsc\nlfence"
                         : "=a"(lo),"=d"(hi) :: "memory");
    return ((uint64_t)hi << 32) | lo;
}

static void shuffle(size_t* a, size_t n) {
    for (size_t i = n-1; i > 0; i--) {
        size_t j = rand() % (i+1);
        size_t t = a[i]; a[i] = a[j]; a[j] = t;
    }
}

static void build_random_chain(uint8_t* buf, size_t region, size_t stride) {
    size_t n = region / stride;
    size_t* perm = (size_t*)malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; i++) perm[i] = i;
    shuffle(perm, n);
    for (size_t i = 0; i < n; i++)
        *(void**)(buf + perm[i]*stride) = (void*)(buf + perm[(i+1)%n]*stride);
    free(perm);
}

/*
 * Warm traverse: run HOPS twice (first pass warms cache, second pass is timed).
 * Used for L1/L2/LLC where we want the HOT latency.
 */
static uint64_t traverse_warm(void* start) {
    volatile void** p = (volatile void**)start;
    for (int i = 0; i < HOPS; i++) p = (volatile void**)*p; /* warm */
    __asm__ __volatile__("mfence\nlfence" ::: "memory");
    uint64_t t0 = rdtsc_fence();
    for (int i = 0; i < HOPS; i++) p = (volatile void**)*p; /* timed */
    uint64_t t1 = rdtsc_fence();
    __asm__ __volatile__("" :: "r"(p) : "memory");
    return t1 - t0;
}

/*
 * Cold traverse: no warm-up.
 * Used for DRAM where chain was never accessed (first touch after build).
 */
static uint64_t traverse_cold(void* start) {
    volatile void** p = (volatile void**)start;
    __asm__ __volatile__("mfence\nlfence" ::: "memory");
    uint64_t t0 = rdtsc_fence();
    for (int i = 0; i < HOPS; i++) p = (volatile void**)*p;
    uint64_t t1 = rdtsc_fence();
    __asm__ __volatile__("" :: "r"(p) : "memory");
    return t1 - t0;
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
    printf("=== Memory Hierarchy Latency Benchmark (v6) ===\n\n");

    const size_t STRIDE = 64;
    const size_t L1_SZ  = 16  * 1024;           /* fits in 32 KB L1   */
    const size_t L2_SZ  = 384 * 1024;           /* fits in 512 KB L2  */
    const size_t LLC_SZ = 8   * 1024 * 1024UL;  /* fits in 12 MB LLC  */
    /* For DRAM: region > LLC so cold accesses MUST go to DRAM */
    const size_t DRAM_SZ = 14 * 1024 * 1024UL;  /* > LLC, never cached */
    const size_t BUF_SZ  = 64 * 1024 * 1024UL;

    uint8_t* buf = (uint8_t*)aligned_alloc(64, BUF_SZ);
    if (!buf) { perror("aligned_alloc"); return 1; }
    memset(buf, 0, BUF_SZ);

    uint64_t samp[SAMPLES];

    /* ── L1 ── */
    build_random_chain(buf, L1_SZ, STRIDE);
    for (int s = 0; s < SAMPLES; s++)
        samp[s] = traverse_warm((void*)buf) / HOPS;
    uint64_t l1_m = median(samp, SAMPLES);
    printf("CPU L1 cache:    %4llu cycles/access\n", (unsigned long long)l1_m);

    /* ── L2 ── */
    build_random_chain(buf, L2_SZ, STRIDE);
    for (int s = 0; s < SAMPLES; s++)
        samp[s] = traverse_warm((void*)buf) / HOPS;
    uint64_t l2_m = median(samp, SAMPLES);
    printf("CPU L2 cache:    %4llu cycles/access\n", (unsigned long long)l2_m);

    /* ── LLC ── */
    build_random_chain(buf, LLC_SZ, STRIDE);
    for (int s = 0; s < SAMPLES; s++)
        samp[s] = traverse_warm((void*)buf) / HOPS;
    uint64_t llc_m = median(samp, SAMPLES);
    printf("CPU LLC:         %4llu cycles/access\n", (unsigned long long)llc_m);

    /* ── DRAM ──
     * Build chain > LLC. Each sample: re-build (touches physical pages,
     * maps to different physical locations) and cold-traverse.
     * Re-building shuffles addresses so the chain is fresh each time.
     * Using a different physical region (offset by sample) avoids
     * hardware prefetcher learning the pattern.
     */
    {
        /* Build fresh chain each sample in DRAM_SZ region */
        uint8_t* dram_buf = buf + LLC_SZ; /* start past LLC region */
        for (int s = 0; s < SAMPLES; s++) {
            build_random_chain(dram_buf, DRAM_SZ, STRIDE);
            samp[s] = traverse_cold((void*)dram_buf) / HOPS;
        }
        uint64_t dram_m = median(samp, SAMPLES);
        printf("CPU DRAM:        %4llu cycles/access\n", (unsigned long long)dram_m);

        /* ── Pagemap ── */
        uint64_t pm_m = 0;
        int fd = open("/proc/self/pagemap", O_RDONLY);
        if (fd < 0) {
            perror("open pagemap");
        } else {
            long pgsz = sysconf(_SC_PAGESIZE);
            uint64_t vpn = (uint64_t)buf / (uint64_t)pgsz;
            uint64_t entry;
            for (int s = 0; s < SAMPLES; s++) {
                uint64_t t0 = rdtsc_fence();
                (void)pread(fd, &entry, 8, (off_t)(vpn * 8));
                uint64_t t1 = rdtsc_fence();
                samp[s] = t1 - t0;
            }
            close(fd);
            pm_m = median(samp, SAMPLES);
            printf("Pagemap:         %4llu cycles  (software page-table read)\n",
                   (unsigned long long)pm_m);
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
    }

    free(buf);
    return 0;
}
