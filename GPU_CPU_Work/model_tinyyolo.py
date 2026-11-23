#!/usr/bin/env python3
"""
Tiny YOLO GPU inference loop — background workload for LLC attack experiment.
Implements YOLOv2-tiny architecture (8 conv layers, ~15MB) in pure PyTorch.
No ultralytics needed.
"""
import torch, torch.nn as nn, time, os

def conv_bn_lrelu(in_c, out_c, k, s=1, p=0):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.1, inplace=True)
    )

class TinyYOLOv2(nn.Module):
    """YOLOv2-tiny: 8 conv layers, input 416x416, ~15MB params."""
    def __init__(self, num_classes=20):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_lrelu(3,   16,  3, 1, 1), nn.MaxPool2d(2, 2),   # 208
            conv_bn_lrelu(16,  32,  3, 1, 1), nn.MaxPool2d(2, 2),   # 104
            conv_bn_lrelu(32,  64,  3, 1, 1), nn.MaxPool2d(2, 2),   # 52
            conv_bn_lrelu(64,  128, 3, 1, 1), nn.MaxPool2d(2, 2),   # 26
            conv_bn_lrelu(128, 256, 3, 1, 1), nn.MaxPool2d(2, 2),   # 13
            conv_bn_lrelu(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 1, padding=1),
            conv_bn_lrelu(512, 1024, 3, 1, 1),
            conv_bn_lrelu(1024, 256, 3, 1, 1),
        )
        self.head = nn.Conv2d(256, 5 * (5 + num_classes), 1)

    def forward(self, x):
        return self.head(self.features(x))

device = torch.device("cuda")
model = TinyYOLOv2(num_classes=20).to(device).eval()
dummy = torch.randn(2, 3, 416, 416, device=device)  # batch=2, 416x416 input

with torch.no_grad():
    for _ in range(5): _ = model(dummy)
torch.cuda.synchronize()

n_params = sum(p.numel() for p in model.parameters())
print(f"[TinyYOLOv2] {n_params/1e6:.1f}M params  GPU inference running  PID={os.getpid()}", flush=True)
i = 0; t0 = time.time()
try:
    while True:
        with torch.no_grad():
            out = model(dummy)
        torch.cuda.synchronize()
        i += 1
        if i % 50 == 0:
            print(f"[TinyYOLOv2] {i} iters  {i/(time.time()-t0):.1f} it/s", flush=True)
except KeyboardInterrupt:
    print(f"[TinyYOLOv2] stopped at {i} iters", flush=True)
