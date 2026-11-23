/*
 * mem_latency_bench.c  (v5 — random pointer chaining)
 * Random order prevents compiler from precomputing addresses.
 * Each load is truly data-dependent on the previous result.
 *
 * Compile:  gcc -O2 -o bin/mem_latency_bench src/mem_latency_bench_v5.c
 * Run:      ./bin/mem_latency_bench   (or with sudo for pagemap PFN)
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

#define SAMPLES  500
#define HOPS     1024

static inline uint64_t rdtsc_fence() {
    uint32_t lo, hi;
    __asm__ __volatile__("mfence\nlfence\nrdtsc\nlfence" : "=a"(lo),"=d"(hi) :: "memory");
    return ((uint64_t)hi << 32) | lo;
}
static inline void clflush_line(void* p) {
    __asm__ __volatile__("clflush (%0)\nmfence" :: "r"(p) : "memory");
}

/* Fisher-Yates shuffle */
static void shuffle(size_t* a, size_t n) {
    for (size_t i = n-1; i > 0; i--) {
        size_t j = (size_t)rand() % (i+1);
        size_t tmp = a[i]; a[i] = a[j]; a[j] = tmp;
    }
}

/*
 * Build a RANDOM pointer chain within buf[0..region_bytes).
 * stride = node spacing (>= 8 bytes, typically 64 for cache-line aligned).
 * Fills the first 8 bytes of each node with a pointer to a random other node.
 */
static void build_random_chain(uint8_t* buf, size_t region, size_t stride) {
    size_t n = region / stride;
    /* build permutation array */
    size_t* perm = (size_t*)malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; i++) perm[i] = i;
    shuffle(perm, n);
    /* link: perm[0] -> perm[1] -> ... -> perm[n-1] -> perm[0] */
    for (size_t i = 0; i < n; i++) {
        size_t src  = perm[i];
        size_t dst  = perm[(i+1) % n];
        *(void**)(buf + src*stride) = (void*)(buf + dst*stride);
    }
    free(perm);
}

/* Traverse HOPS pointer hops, return (end_time - start_time) in cycles */
static uint64_t traverse(void* start) {
    volatile void** p = (volatile void**)start;
    /* warm-up: one full pass */
    for (int i = 0; i < HOPS; i++) p = (volatile void**)*p;
    /* timed pass */
    uint64_t t0 = rdtsc_fence();
    for (int i = 0; i < HOPS; i++) p = (volatile void**)*p;
    uint64_t t1 = rdtsc_fence();
    __asm__ __volatile__("" :: "r"(p) : "memory"); /* keep p live */
    return t1 - t0;
}

static uint64_t median(uint64_t* a, int n) {
    /* insertion sort */
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

    const size_t STRIDE = 64;
    /* Working set sizes — tune to your CPU */
    const size_t L1_SZ  = 16  * 1024;               /* < 32 KB L1 */
    const size_t L2_SZ  = 384 * 1024;               /* < 512 KB L2, > L1 */
    const size_t LLC_SZ = 8   * 1024 * 1024UL;      /* < 12 MB LLC, > L2 */
    const size_t BUF_SZ = 64  * 1024 * 1024UL;

    uint8_t* buf = (uint8_t*)aligned_alloc(64, BUF_SZ);
    if (!buf) { perror("aligned_alloc"); return 1; }
    memset(buf, 0, BUF_SZ);

    uint64_t samp[SAMPLES];

    /* ── L1 ──────────────────────────────────────────── */
    build_random_chain(buf, L1_SZ, STRIDE);
    for (int s = 0; s < SAMPLES; s++)
        samp[s] = traverse((void*)buf) / HOPS;
    uint64_t l1_m = median(samp, SAMPLES);
    printf("CPU L1 cache:    %4llu cycles/access\n", (unsigned long long)l1_m);

    /* ── L2 ──────────────────────────────────────────── */
    build_random_chain(buf, L2_SZ, STRIDE);
    for (int s = 0; s < SAMPLES; s++)
        samp[s] = traverse((void*)buf) / HOPS;
    uint64_t l2_m = median(samp, SAMPLES);
    printf("CPU L2 cache:    %4llu cycles/access\n", (unsigned long long)l2_m);

    /* ── LLC ─────────────────────────────────────────── */
    build_random_chain(buf, LLC_SZ, STRIDE);
    for (int s = 0; s < SAMPLES; s++)
        samp[s] = traverse((void*)buf) / HOPS;
    uint64_t llc_m = median(samp, SAMPLES);
    printf("CPU LLC:         %4llu cycles/access\n", (unsigned long long)llc_m);

    /* ── DRAM (clflush) ──────────────────────────────── */
    /*
     * Build a small random chain (4 KB = 64 nodes).
     * Flush all 64 cache lines before each traverse.
     */
    const size_t DFLUSH_SZ = 4096;
    build_random_chain(buf, DFLUSH_SZ, STRIDE);
    for (int s = 0; s < SAMPLES; s++) {
        for (size_t i = 0; i < DFLUSH_SZ; i += 64)
            clflush_line(buf + i);
        samp[s] = traverse((void*)buf) / HOPS;
    }
    uint64_t dram_m = median(samp, SAMPLES);
    printf("CPU DRAM:        %4llu cycles/access\n", (unsigned long long)dram_m);

    /* ── Pagemap (software page-table read) ──────────── */
    /*
     * /proc/self/pagemap: 8-byte entry per virtual page.
     * Kernel zeroes PFN for unprivileged; sudo gives real PFN.
     * Cost = syscall overhead + kernel page-table walk.
     */
    uint64_t pm_m = 0;
    {
        int fd = open("/proc/self/pagemap", O_RDONLY);
        if (fd < 0) { perror("open pagemap"); }
        else {
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
    }

    /* ── CSV output ──────────────────────────────────── */
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
