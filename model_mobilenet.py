#!/usr/bin/env python3
"""MobileNetV2 GPU inference loop — background workload for LLC attack experiment."""
import torch, torchvision.models as models, time, os, sys

device = torch.device("cuda")
model = models.mobilenet_v2(weights=None).to(device).eval()
dummy = torch.randn(4, 3, 224, 224, device=device)  # batch=4

# warmup
with torch.no_grad():
    for _ in range(10): _ = model(dummy)
torch.cuda.synchronize()

print(f"[MobileNetV2] GPU inference running  PID={os.getpid()}", flush=True)
i = 0
t0 = time.time()
try:
    while True:
        with torch.no_grad():
            out = model(dummy)
        torch.cuda.synchronize()
        i += 1
        if i % 100 == 0:
            print(f"[MobileNetV2] {i} iters  {i/(time.time()-t0):.1f} it/s", flush=True)
except KeyboardInterrupt:
    print(f"[MobileNetV2] stopped at {i} iters", flush=True)
