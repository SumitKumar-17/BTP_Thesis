#!/usr/bin/env python3
"""SqueezeNet1_0 GPU inference loop — background workload for LLC attack experiment."""
import torch, torchvision.models as models, time, os

device = torch.device("cuda")
model = models.squeezenet1_0(weights=None).to(device).eval()
dummy = torch.randn(4, 3, 224, 224, device=device)

with torch.no_grad():
    for _ in range(10): _ = model(dummy)
torch.cuda.synchronize()

print(f"[SqueezeNet1_0] GPU inference running  PID={os.getpid()}", flush=True)
i = 0; t0 = time.time()
try:
    while True:
        with torch.no_grad():
            out = model(dummy)
        torch.cuda.synchronize()
        i += 1
        if i % 100 == 0:
            print(f"[SqueezeNet1_0] {i} iters  {i/(time.time()-t0):.1f} it/s", flush=True)
except KeyboardInterrupt:
    print(f"[SqueezeNet1_0] stopped at {i} iters", flush=True)
