#!/usr/bin/env python3
"""
analyze_reverse.py
Reads reverse_pp_set*.csv files and plots p99 ratio across sets.
Usage: python3 analyze_reverse.py
"""

import glob, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_files = sorted(glob.glob("reverse_pp_set*.csv"),
                   key=lambda f: int(f.split("set")[1].split(".")[0]))

if not csv_files:
    print("No reverse_pp_set*.csv files found. Run the experiment first.")
    exit(1)

sets, p99_ratios, base_p99s, atk_p99s = [], [], [], []

for f in csv_files:
    set_id = int(f.split("set")[1].split(".")[0])
    df = pd.read_csv(f)
    # take the max p99_ratio across all congruent lines in this set
    max_ratio = df["p99_ratio"].max()
    mean_base = df["base_p99"].mean()
    mean_atk  = df["atk_p99"].mean()
    sets.append(set_id)
    p99_ratios.append(max_ratio)
    base_p99s.append(mean_base)
    atk_p99s.append(mean_atk)

sets      = np.array(sets)
p99_ratios = np.array(p99_ratios)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Plot 1: p99 ratio across sets ---
ax = axes[0]
ax.bar(range(len(sets)), p99_ratios, color=[
    "crimson" if r > 1.2 else "steelblue" for r in p99_ratios])
ax.axhline(1.2, color="black", linestyle="--", linewidth=1, label="1.2× threshold")
ax.set_xticks(range(len(sets)))
ax.set_xticklabels([str(s) for s in sets], rotation=45, ha="right", fontsize=8)
ax.set_xlabel("LLC Set ID")
ax.set_ylabel("Max p99 ratio (attack / baseline)")
ax.set_title("Reverse Prime+Probe: p99 ratio per LLC set\n(CPU attacker, GPU-UVM victim)")
ax.legend()
ax.grid(axis="y", alpha=0.4)

# --- Plot 2: baseline vs attack p99 ---
ax2 = axes[1]
x = np.arange(len(sets))
w = 0.35
ax2.bar(x - w/2, base_p99s, w, label="Baseline p99", color="steelblue", alpha=0.8)
ax2.bar(x + w/2, atk_p99s,  w, label="Attack p99",   color="crimson",   alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels([str(s) for s in sets], rotation=45, ha="right", fontsize=8)
ax2.set_xlabel("LLC Set ID")
ax2.set_ylabel("p99 latency (CPU rdtsc cycles)")
ax2.set_title("Reverse Prime+Probe: Baseline vs Attack p99\n(mean across congruent lines)")
ax2.legend()
ax2.grid(axis="y", alpha=0.4)

plt.tight_layout()
out = "reverse_pp_results.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")
plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")

# --- Print summary table ---
print(f"\n{'Set':>8}  {'Max p99 ratio':>14}  {'Base p99':>10}  {'Atk p99':>10}  {'Signal?':>8}")
print("-" * 60)
for i in range(len(sets)):
    sig = "YES" if p99_ratios[i] > 1.2 else "no"
    print(f"{sets[i]:>8}  {p99_ratios[i]:>14.3f}  {base_p99s[i]:>10.0f}  {atk_p99s[i]:>10.0f}  {sig:>8}")
