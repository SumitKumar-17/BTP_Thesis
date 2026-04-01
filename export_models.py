#!/usr/bin/env python3
"""
Export all 5 lightweight AI models to TorchScript (.pt) so they can be
loaded and called directly from the C++/CUDA binary via LibTorch.
"""
import torch, torchvision.models as tvm, timm, os, sys

OUTDIR = "/home/rtx3060/Desktop/sumit-GPU-Codes/gpu_cpu_work_1/models"
os.makedirs(OUTDIR, exist_ok=True)
device = torch.device("cuda")

def export(model, name, dummy_input):
    model = model.to(device).eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)
    out = os.path.join(OUTDIR, f"{name}.pt")
    traced.save(out)
    sz = os.path.getsize(out) / 1e6
    print(f"  Saved {name}.pt  ({sz:.1f} MB)")

print("Exporting models to TorchScript...")

# 1. MobileNetV2 — ~14 MB
export(tvm.mobilenet_v2(weights=None), "mobilenet_v2",
       torch.randn(4, 3, 224, 224, device=device))

# 2. SqueezeNet1_0 — ~5 MB
export(tvm.squeezenet1_0(weights=None), "squeezenet1_0",
       torch.randn(4, 3, 224, 224, device=device))

# 3. EfficientNet-Lite0 (timm) — ~16 MB
export(timm.create_model("efficientnet_lite0", pretrained=False), "efficientnet_lite0",
       torch.randn(4, 3, 224, 224, device=device))

# 4. DistilBERT — ~250 MB
from transformers import DistilBertModel, DistilBertConfig
cfg   = DistilBertConfig()
bert  = DistilBertModel(cfg).to(device).eval()
# trace with fixed token inputs (seq_len=128, batch=2)
ids   = torch.randint(0, cfg.vocab_size, (2, 128), device=device)
mask  = torch.ones(2, 128, dtype=torch.long, device=device)
with torch.no_grad():
    traced = torch.jit.trace(bert, (ids, mask))
out = os.path.join(OUTDIR, "distilbert.pt")
traced.save(out)
print(f"  Saved distilbert.pt  ({os.path.getsize(out)/1e6:.1f} MB)")

# 5. TinyYOLOv2 — ~15 MB (custom architecture, no dependency needed)
import torch.nn as nn
def cbl(ic, oc, k, s=1, p=0):
    return nn.Sequential(nn.Conv2d(ic, oc, k, s, p, bias=False),
                         nn.BatchNorm2d(oc), nn.LeakyReLU(0.1, inplace=True))
class TinyYOLO(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            cbl(3,16,3,1,1), nn.MaxPool2d(2,2),
            cbl(16,32,3,1,1), nn.MaxPool2d(2,2),
            cbl(32,64,3,1,1), nn.MaxPool2d(2,2),
            cbl(64,128,3,1,1), nn.MaxPool2d(2,2),
            cbl(128,256,3,1,1), nn.MaxPool2d(2,2),
            cbl(256,512,3,1,1), nn.MaxPool2d(2,1,padding=1),
            cbl(512,1024,3,1,1), cbl(1024,256,3,1,1),
            nn.Conv2d(256, 125, 1))
    def forward(self, x): return self.net(x)

export(TinyYOLO(), "tinyyolo_v2",
       torch.randn(2, 3, 416, 416, device=device))

print("\nAll models exported to:", OUTDIR)
print("Files:", os.listdir(OUTDIR))
