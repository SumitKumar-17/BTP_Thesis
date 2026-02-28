#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("=== Device %d: %s ===\n", dev, prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Total Global Memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
        printf("Unified Addressing: %s\n", prop.unifiedAddressing ? "YES" : "NO");
        printf("Managed Memory: %s\n", prop.managedMemory ? "YES" : "NO");
        printf("Concurrent Kernels: %s\n", prop.concurrentKernels ? "YES" : "NO");
        printf("L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
        printf("Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
        printf("\n");
    }

    return 0;
}
