#!/usr/bin/env python3
"""EfficientNet-Lite0 GPU inference loop — background workload for LLC attack experiment."""
import torch, timm, time, os

device = torch.device("cuda")
model = timm.create_model('efficientnet_lite0', pretrained=False).to(device).eval()
dummy = torch.randn(4, 3, 224, 224, device=device)

with torch.no_grad():
    for _ in range(10): _ = model(dummy)
torch.cuda.synchronize()

print(f"[EfficientNet-Lite0] GPU inference running  PID={os.getpid()}", flush=True)
i = 0; t0 = time.time()
try:
    while True:
        with torch.no_grad():
            out = model(dummy)
        torch.cuda.synchronize()
        i += 1
        if i % 100 == 0:
            print(f"[EfficientNet-Lite0] {i} iters  {i/(time.time()-t0):.1f} it/s", flush=True)
except KeyboardInterrupt:
    print(f"[EfficientNet-Lite0] stopped at {i} iters", flush=True)
