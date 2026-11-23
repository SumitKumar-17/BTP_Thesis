#!/usr/bin/env python3
"""
Analyze and compare attack heatmaps across different CPU workload scenarios.
Produces side-by-side heatmaps, bar charts, and a summary report.
"""
import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings; warnings.filterwarnings("ignore")

BASE = "/home/rtx3060/Desktop/sumit-GPU-Codes/gpu_cpu_work_1/results/workload_comparison"
OUT_DIR = os.path.join(BASE, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

WORKLOAD_LABELS = {
    "W0_idle":         "W0: CPU idle",
    "W1_matmul":       "W1: Matmul (256×256)",
    "W2_ai_mobilenet": "W2: AI (MobileNetV2)",
    "W3_fft":          "W3: FFT (1M points)",
}

def load_workload(wdir):
    """Load all runs for a workload, return dict: set_id -> list of DataFrames (one per run)."""
    run_dirs = sorted(glob.glob(os.path.join(wdir, "run_*")))
    per_set = {}
    for rdir in run_dirs:
        csvs = sorted(glob.glob(os.path.join(rdir, "scan_set_*.csv")))
        for csv in csvs:
            m = re.search(r"scan_set_(\d+)\.csv", csv)
            if not m: continue
            sid = int(m.group(1))
            try:
                df = pd.read_csv(csv)
                per_set.setdefault(sid, []).append(df)
            except: pass
    return per_set

def mean_matrix(per_set, metric):
    """Build a (sets × max_addrs) matrix with mean values across runs."""
    sets = sorted(per_set.keys())
    rows = []
    for s in sets:
        dfs = per_set[s]
        if not dfs or metric not in dfs[0].columns:
            rows.append(np.array([]))
            continue
        max_len = max(len(df) for df in dfs)
        mat = np.full((len(dfs), max_len), np.nan)
        for i, df in enumerate(dfs):
            vals = df[metric].values[:max_len].astype(float)
            mat[i, :len(vals)] = vals
        rows.append(np.nanmean(mat, axis=0))
    max_len = max(len(r) for r in rows if len(r) > 0)
    matrix = np.full((len(sets), max_len), np.nan)
    for i, r in enumerate(rows):
        if len(r) > 0:
            matrix[i, :len(r)] = r
    return matrix, sets

# ---- Load all workloads ----
workloads = {}
workload_dirs = sorted(glob.glob(os.path.join(BASE, "W*")))
for wdir in workload_dirs:
    wname = os.path.basename(wdir)
    data = load_workload(wdir)
    if data:
        workloads[wname] = data
        total_runs = max(len(v) for v in data.values())
        print(f"Loaded {wname}: {len(data)} sets, {total_runs} runs")

if not workloads:
    print("No workload data found."); exit(1)

# ============================================================
# FIGURE 1 — Side-by-side p99_ratio heatmaps (one per workload)
# ============================================================
n = len(workloads)
fig, axes = plt.subplots(1, n, figsize=(7*n, 7), sharey=True)
if n == 1: axes = [axes]

all_mats = {}
for wname, data in workloads.items():
    mat, sets = mean_matrix(data, "p99_ratio")
    all_mats[wname] = (mat, sets)

all_vals = np.concatenate([m.ravel()[~np.isnan(m.ravel())] for m, _ in all_mats.values()])
vmax = max(np.nanpercentile(all_vals, 98), 2.0)

for ax, (wname, (mat, sets)) in zip(axes, all_mats.items()):
    label = WORKLOAD_LABELS.get(wname, wname)
    im = ax.imshow(mat, aspect="auto", origin="upper",
                   norm=mcolors.LogNorm(vmin=0.5, vmax=vmax),
                   cmap="YlOrRd")
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Congruent address index")
    if ax == axes[0]:
        ax.set_ylabel("LLC set ID")
        ax.set_yticks(range(len(sets)))
        ax.set_yticklabels([str(s) for s in sets], fontsize=7)

plt.colorbar(im, ax=axes[-1], label="mean p99_ratio (log scale)")
fig.suptitle("GPU→CPU LLC attack signal: p99 ratio across workload scenarios\n"
             "Brighter = stronger eviction signal detected by CPU probe", fontsize=12)
plt.tight_layout()
out = os.path.join(OUT_DIR, "workload_heatmap_comparison.png")
plt.savefig(out, dpi=120); plt.close()
print(f"Saved: {out}")

# ============================================================
# FIGURE 2 — Per-set mean p99_ratio bar chart per workload
# ============================================================
sets_all = sorted({s for data in workloads.values() for s in data})
x = np.arange(len(sets_all))
width = 0.8 / max(len(workloads), 1)
colors = plt.cm.tab10(np.linspace(0, 0.6, len(workloads)))

fig, ax = plt.subplots(figsize=(15, 5))
for i, (wname, data) in enumerate(workloads.items()):
    means = []
    for s in sets_all:
        dfs = data.get(s, [])
        if dfs and "p99_ratio" in dfs[0].columns:
            vals = [df["p99_ratio"].mean() for df in dfs]
            means.append(np.mean(vals))
        else:
            means.append(np.nan)
    offset = (i - len(workloads)/2 + 0.5) * width
    ax.bar(x + offset, means, width=width*0.9,
           label=WORKLOAD_LABELS.get(wname, wname), color=colors[i], alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in sets_all], rotation=45, ha="right", fontsize=8)
ax.set_xlabel("LLC set ID")
ax.set_ylabel("Mean p99_ratio (attack / baseline)")
ax.set_title("Attack strength per LLC set — grouped by workload\n"
             "Same set but different workload → different signal strength?", fontsize=11)
ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.5)
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
out = os.path.join(OUT_DIR, "workload_per_set_bar.png")
plt.savefig(out, dpi=120); plt.close()
print(f"Saved: {out}")

# ============================================================
# FIGURE 3 — Overall attack strength per workload (violin / box)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = [("p99_ratio", "P99 ratio"), ("max_ratio", "Max ratio"), ("median_ratio", "Median ratio")]

for ax, (metric, mlabel) in zip(axes, metrics):
    data_list = []
    xlabels = []
    for wname, data in workloads.items():
        vals = []
        for dfs in data.values():
            for df in dfs:
                if metric in df.columns:
                    vals.extend(df[metric].dropna().values.tolist())
        data_list.append(vals)
        xlabels.append(WORKLOAD_LABELS.get(wname, wname).replace(" ", "\n"))

    bp = ax.boxplot(data_list, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel(mlabel)
    ax.set_title(f"{mlabel} distribution\nacross all sets and runs")
    ax.axhline(1.0, color="red", ls="--", lw=1, alpha=0.6)

plt.suptitle("Distribution of attack signal metrics by workload\n"
             "Red dashed = no effect (ratio=1)", fontsize=11)
plt.tight_layout()
out = os.path.join(OUT_DIR, "workload_metric_distribution.png")
plt.savefig(out, dpi=120); plt.close()
print(f"Saved: {out}")

# ============================================================
# FIGURE 4 — Baseline latency comparison (does workload raise baseline?)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))
for i, (wname, data) in enumerate(workloads.items()):
    vals = []
    for dfs in data.values():
        for df in dfs:
            if "baseline_p99" in df.columns:
                vals.extend(df["baseline_p99"].dropna().values.tolist())
    if vals:
        offset = (i - len(workloads)/2 + 0.5) * 0.15
        ax.violinplot([vals], positions=[i], showmedians=True)

ax.set_xticks(range(len(workloads)))
ax.set_xticklabels([WORKLOAD_LABELS.get(w, w) for w in workloads], fontsize=9)
ax.set_ylabel("Baseline p99 latency (CPU cycles)")
ax.set_title("CPU probe baseline latency per workload\n"
             "Higher = workload already evicting cache lines (more LLC pressure)", fontsize=11)
plt.tight_layout()
out = os.path.join(OUT_DIR, "workload_baseline_latency.png")
plt.savefig(out, dpi=120); plt.close()
print(f"Saved: {out}")

# ============================================================
# TEXT SUMMARY
# ============================================================
print("\n===== WORKLOAD COMPARISON SUMMARY =====")
summary_rows = []
for wname, data in workloads.items():
    p99_vals, max_vals, base_vals = [], [], []
    for dfs in data.values():
        for df in dfs:
            if "p99_ratio" in df.columns:
                p99_vals.extend(df["p99_ratio"].dropna().tolist())
            if "max_ratio" in df.columns:
                max_vals.extend(df["max_ratio"].dropna().tolist())
            if "baseline_p99" in df.columns:
                base_vals.extend(df["baseline_p99"].dropna().tolist())
    summary_rows.append({
        "workload": WORKLOAD_LABELS.get(wname, wname),
        "mean_p99_ratio": np.mean(p99_vals) if p99_vals else np.nan,
        "mean_max_ratio": np.mean(max_vals) if max_vals else np.nan,
        "mean_baseline_p99_cycles": np.mean(base_vals) if base_vals else np.nan,
    })
    print(f"  {wname}: mean_p99_ratio={np.mean(p99_vals):.3f}  "
          f"mean_max_ratio={np.mean(max_vals):.1f}  "
          f"baseline_p99={np.mean(base_vals):.0f} cycles")

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUT_DIR, "workload_summary.csv"), index=False)
print(f"\nSaved summary CSV to {OUT_DIR}/workload_summary.csv")
print(f"All plots saved to {OUT_DIR}")
