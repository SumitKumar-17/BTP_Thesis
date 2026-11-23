#!/usr/bin/env python3
"""DistilBERT GPU inference loop — background workload for LLC attack experiment."""
import torch, time, os
from transformers import DistilBertModel, DistilBertConfig

device = torch.device("cuda")
# Use default config (no pretrained weights needed — architecture drives LLC pattern)
config = DistilBertConfig()
model = DistilBertModel(config).to(device).eval()

# seq_len=128, batch=4 — typical inference shape
input_ids      = torch.randint(0, config.vocab_size, (4, 128), device=device)
attention_mask = torch.ones(4, 128, dtype=torch.long, device=device)

with torch.no_grad():
    for _ in range(5):
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
torch.cuda.synchronize()

print(f"[DistilBERT] GPU inference running  PID={os.getpid()}", flush=True)
i = 0; t0 = time.time()
try:
    while True:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()
        i += 1
        if i % 50 == 0:
            print(f"[DistilBERT] {i} iters  {i/(time.time()-t0):.1f} it/s", flush=True)
except KeyboardInterrupt:
    print(f"[DistilBERT] stopped at {i} iters", flush=True)
