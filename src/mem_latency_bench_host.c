/*
 * mem_latency_bench.cu  (v4)
 * Uses pointer chains with noinline + O0 traversal to defeat compiler opts.
 * Measures: CPU L1, L2, LLC, DRAM, Pagemap latency.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <x86intrin.h>

#define SAMPLES    300
#define HOPS       512

static inline uint64_t rdtsc_start() {
    uint32_t lo, hi;
    __asm__ __volatile__("mfence\nlfence\nrdtsc" : "=a"(lo),"=d"(hi) :: "memory");
    return ((uint64_t)hi<<32)|lo;
}
static inline uint64_t rdtsc_end() {
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc\nlfence" : "=a"(lo),"=d"(hi) :: "memory");
    return ((uint64_t)hi<<32)|lo;
}
static inline void clflush_addr(void* p) {
    __asm__ __volatile__("clflush (%0)\nmfence" :: "r"(p) : "memory");
}

/* Build circular pointer chain: node i at buf+i*stride, stores &node[i+1] */
static void build_chain(uint8_t* buf, size_t region, size_t stride) {
    size_t n = region / stride;
    for (size_t i = 0; i < n - 1; i++)
        *(void**)(buf + i*stride) = (void*)(buf + (i+1)*stride);
    *(void**)(buf + (n-1)*stride) = (void*)buf;
}

/* Traverse HOPS steps — noinline + O0 to prevent loop elimination */
__attribute__((noinline))
#ifndef __NVCC__
__attribute__((optimize("O0")))
#endif
static uint64_t traverse(void* head) {
    volatile void** p = (volatile void**)head;
    /* warm */
    for (int i = 0; i < HOPS; i++) p = (volatile void**)*p;
    uint64_t t0 = rdtsc_start();
    for (int i = 0; i < HOPS; i++) p = (volatile void**)*p;
    uint64_t t1 = rdtsc_end();
    /* prevent dead-code elimination of p */
    __asm__ __volatile__("" :: "r"(p) : "memory");
    return t1 - t0;
}

static uint64_t median(uint64_t* a, int n) {
    for (int i = 1; i < n; i++) {
        uint64_t k = a[i]; int j = i-1;
        while (j >= 0 && a[j] > k) { a[j+1]=a[j]; j--; }
        a[j+1] = k;
    }
    return a[n/2];
}

int main() {
    printf("=== Memory Hierarchy Latency Benchmark (v4) ===\n\n");

    const size_t STRIDE  = 64;
    const size_t L1_SZ   = 16  * 1024;
    const size_t L2_SZ   = 384 * 1024;
    const size_t LLC_SZ  = 8   * 1024 * 1024UL;
    const size_t BUF_SZ  = 64  * 1024 * 1024UL;

    uint8_t* buf = (uint8_t*)aligned_alloc(64, BUF_SZ);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }
    memset(buf, 0, BUF_SZ);

    uint64_t samp[SAMPLES];

#define MEAS(label, setup, body, var) \
    do { \
        setup; \
        for (int _s = 0; _s < SAMPLES; _s++) { body; samp[_s] = traverse((void*)buf)/HOPS; } \
        var = median(samp, SAMPLES); \
        printf("%-14s %4llu cycles/access\n", label, (unsigned long long)var); \
    } while(0)

    uint64_t l1_m, l2_m, llc_m, dram_m;

    MEAS("CPU L1:",   build_chain(buf, L1_SZ, STRIDE),  (void)0,     l1_m);
    MEAS("CPU L2:",   build_chain(buf, L2_SZ, STRIDE),  (void)0,     l2_m);
    MEAS("CPU LLC:",  build_chain(buf, LLC_SZ, STRIDE), (void)0,     llc_m);

    /* DRAM: build 4KB chain, clflush before each traverse */
    {
        const size_t DF = 4096;
        build_chain(buf, DF, STRIDE);
        for (int s = 0; s < SAMPLES; s++) {
            for (size_t i = 0; i < DF; i += 64) clflush_addr(buf+i);
            samp[s] = traverse((void*)buf) / HOPS;
        }
        dram_m = median(samp, SAMPLES);
        printf("%-14s %4llu cycles/access\n", "CPU DRAM:", (unsigned long long)dram_m);
    }

    /* Pagemap */
    uint64_t pm_m;
    {
        int fd = open("/proc/self/pagemap", O_RDONLY);
        long pgsz = sysconf(_SC_PAGESIZE);
        uint64_t vpn = (uint64_t)buf / (uint64_t)pgsz;
        uint64_t entry;
        for (int s = 0; s < SAMPLES; s++) {
            uint64_t t0 = rdtsc_start();
            (void)pread(fd, &entry, 8, (off_t)(vpn * 8));
            uint64_t t1 = rdtsc_end();
            samp[s] = t1 - t0;
        }
        close(fd);
        pm_m = median(samp, SAMPLES);
        printf("%-14s %4llu cycles (software page-table read)\n",
               "Pagemap:", (unsigned long long)pm_m);
    }

    /* Save CSV */
    {
        FILE* fp = fopen("results/mem_latency.csv", "w");
        if (!fp) { fprintf(stderr, "cannot open results/mem_latency.csv\n"); }
        else {
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
