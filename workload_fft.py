#!/usr/bin/env python3
"""
Background CPU workload: continuous large FFT via NumPy.
This is memory-bandwidth heavy — large arrays thrash LLC from CPU side.
"""
import time, numpy as np, os

# 8MB float32 array → large enough to overflow LLC working set
N = 1024 * 1024  # 4 MB real + 4 MB complex output
arr = np.random.randn(N).astype(np.float32)

print(f"[FFT workload] Large FFT running (N={N}, PID {os.getpid()})", flush=True)

iters = 0
t0 = time.time()
try:
    while True:
        out = np.fft.rfft(arr)
        # also do array sum to force reads
        _ = out.real.sum()
        iters += 1
        if iters % 200 == 0:
            elapsed = time.time() - t0
            print(f"[FFT workload] {iters} FFTs in {elapsed:.1f}s ({iters/elapsed:.1f} fft/s)", flush=True)
except KeyboardInterrupt:
    print(f"[FFT workload] Stopped after {iters} FFTs.", flush=True)
