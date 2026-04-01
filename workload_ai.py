#!/usr/bin/env python3
"""
Background CPU workload: continuous MobileNetV2 inference on CPU.
Run this as a background process during evset_address_scan to create
realistic AI inference LLC pressure on the CPU side.
"""
import time, sys, signal
import torch
import torchvision.models as models
import numpy as np

torch.set_num_threads(2)  # leave cores for the attack measurement

model = models.mobilenet_v2(weights=None)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

# warmup
with torch.no_grad():
    for _ in range(5):
        _ = model(dummy_input)

print(f"[AI workload] MobileNetV2 CPU inference running (PID {__import__('os').getpid()})", flush=True)

iters = 0
t0 = time.time()
try:
    while True:
        with torch.no_grad():
            out = model(dummy_input)
        iters += 1
        if iters % 50 == 0:
            elapsed = time.time() - t0
            print(f"[AI workload] {iters} inferences in {elapsed:.1f}s ({iters/elapsed:.1f} inf/s)", flush=True)
except KeyboardInterrupt:
    print(f"[AI workload] Stopped after {iters} inferences.", flush=True)
