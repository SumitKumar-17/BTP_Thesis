#!/usr/bin/env python3
import torch, torch.nn as nn, os

OUTDIR = "/home/rtx3060/Desktop/sumit-GPU-Codes/gpu_cpu_work_1/models"
os.makedirs(OUTDIR, exist_ok=True)
device = torch.device("cuda")

# DistilBERT-equivalent: 6-layer transformer encoder, same size (~66M params)
class MiniTransformer(nn.Module):
    def __init__(self, vocab=30522, seq=128, hidden=768, heads=12, layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.pos   = nn.Embedding(seq, hidden)
        enc = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=3072,
                                          batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.pool = nn.Linear(hidden, hidden)

    def forward(self, x):
        b, s = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0).expand(b, -1)
        h = self.embed(x) + self.pos(pos)
        h = self.encoder(h)
        return self.pool(h[:, 0])

model = MiniTransformer().to(device).eval()
dummy = torch.randint(0, 30522, (2, 128), device=device)
with torch.no_grad():
    traced = torch.jit.trace(model, dummy)
out = os.path.join(OUTDIR, "minitransformer.pt")
traced.save(out)
n = sum(p.numel() for p in model.parameters())
print(f"Saved minitransformer.pt  ({os.path.getsize(out)/1e6:.1f} MB, {n/1e6:.1f}M params)")

# TinyYOLOv2 (already done above but include here for completeness)
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
    def forward(self, x):
        return self.net(x)

yolo = TinyYOLO().to(device).eval()
dummy_yolo = torch.randn(2, 3, 416, 416, device=device)
with torch.no_grad():
    traced_y = torch.jit.trace(yolo, dummy_yolo)
out_y = os.path.join(OUTDIR, "tinyyolo_v2.pt")
traced_y.save(out_y)
ny = sum(p.numel() for p in yolo.parameters())
print(f"Saved tinyyolo_v2.pt  ({os.path.getsize(out_y)/1e6:.1f} MB, {ny/1e6:.1f}M params)")

print("\nAll models in:", os.listdir(OUTDIR))
