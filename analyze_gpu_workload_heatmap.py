#!/usr/bin/env python3
"""
Analyze GPU workload heatmap comparison.
Loads all scenarios (GPU0..GPU4 x CPU workloads), produces:
  1. Side-by-side heatmaps per scenario
  2. Per-set bar chart comparing p99_ratio across scenarios
  3. Metric distribution boxplots
  4. Baseline latency comparison (does GPU workload change CPU baseline?)
  5. Summary CSV + markdown report
"""
import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings; warnings.filterwarnings("ignore")

BASE    = "/home/rtx3060/Desktop/sumit-GPU-Codes/gpu_cpu_work_1/results/gpu_workload_heatmap"
OUT_DIR = os.path.join(BASE, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

SCENARIO_LABELS = {
    "GPU0_none__CPU0_none":       "GPU: none\nCPU: none",
    "GPU1_matmul128__CPU1_mm64":  "GPU: matmul 128×128\nCPU: matmul 64×64",
    "GPU2_matmul512__CPU2_mm256": "GPU: matmul 512×512\nCPU: matmul 256×256",
    "GPU3_conv2d__CPU1_mm64":     "GPU: conv2d 64×64×32\nCPU: matmul 64×64",
    "GPU4_bandwidth__CPU3_fft":   "GPU: bandwidth 16MB\nCPU: FFT 8K",
}

def load_scenario(sdir):
    run_dirs = sorted(glob.glob(os.path.join(sdir, "run_*")))
    per_set = {}
    for rdir in run_dirs:
        for csv in sorted(glob.glob(os.path.join(rdir, "scan_set_*.csv"))):
            m = re.search(r"scan_set_(\d+)\.csv", csv)
            if not m: continue
            sid = int(m.group(1))
            try:
                df = pd.read_csv(csv)
                per_set.setdefault(sid, []).append(df)
            except: pass
    return per_set

def mean_matrix(per_set, metric):
    sets = sorted(per_set.keys())
    rows = []
    for s in sets:
        dfs = per_set[s]
        if not dfs or metric not in dfs[0].columns:
            rows.append(np.array([np.nan]))
            continue
        max_len = max(len(df) for df in dfs)
        mat = np.full((len(dfs), max_len), np.nan)
        for i, df in enumerate(dfs):
            v = df[metric].values[:max_len].astype(float)
            mat[i, :len(v)] = v
        rows.append(np.nanmean(mat, axis=0))
    max_len = max(len(r) for r in rows)
    matrix = np.full((len(sets), max_len), np.nan)
    for i, r in enumerate(rows):
        matrix[i, :len(r)] = r
    return matrix, sets

# ---- Load all scenarios ----
scenarios = {}
for sdir in sorted(glob.glob(os.path.join(BASE, "GPU*"))):
    name = os.path.basename(sdir)
    data = load_scenario(sdir)
    if data:
        n_runs = max(len(v) for v in data.values())
        scenarios[name] = data
        print(f"Loaded {name}: {len(data)} sets, {n_runs} runs")

if not scenarios:
    print("No scenario data found."); exit(1)

n_sc = len(scenarios)
colors = plt.cm.tab10(np.linspace(0, 0.8, n_sc))

# ============================================================
# FIGURE 1 — Heatmap grid: one column per scenario
# ============================================================
fig, axes = plt.subplots(1, n_sc, figsize=(6*n_sc, 8), sharey=True)
if n_sc == 1: axes = [axes]

all_mats = {}
for name, data in scenarios.items():
    mat, sets = mean_matrix(data, "p99_ratio")
    all_mats[name] = (mat, sets)

all_vals = np.concatenate([m.ravel()[np.isfinite(m.ravel())] for m, _ in all_mats.values()])
vmax = max(np.nanpercentile(all_vals, 98), 2.0)

for ax, (name, (mat, sets)) in zip(axes, all_mats.items()):
    label = SCENARIO_LABELS.get(name, name)
    im = ax.imshow(mat, aspect="auto", origin="upper",
                   norm=mcolors.LogNorm(vmin=0.5, vmax=vmax), cmap="YlOrRd")
    ax.set_title(label, fontsize=9, fontweight="bold")
    ax.set_xlabel("Address index", fontsize=8)
    if ax == axes[0]:
        ax.set_ylabel("LLC set ID")
        ax.set_yticks(range(len(sets)))
        ax.set_yticklabels([str(s) for s in sets], fontsize=7)

plt.colorbar(im, ax=axes[-1], label="mean p99_ratio (log scale)")
fig.suptitle("GPU→CPU LLC attack: p99 ratio heatmap by GPU+CPU workload scenario\n"
             "Each column = one workload combo  |  Brighter = stronger eviction signal", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "scenario_heatmap_grid.png"), dpi=120); plt.close()
print("Saved: scenario_heatmap_grid.png")

# ============================================================
# FIGURE 2 — Per-set mean p99_ratio grouped bar chart
# ============================================================
sets_all = sorted({s for data in scenarios.values() for s in data})
x = np.arange(len(sets_all))
w = 0.8 / n_sc

fig, ax = plt.subplots(figsize=(16, 5))
for i, (name, data) in enumerate(scenarios.items()):
    means = []
    for s in sets_all:
        dfs = data.get(s, [])
        vals = [df["p99_ratio"].mean() for df in dfs if "p99_ratio" in df.columns]
        means.append(np.mean(vals) if vals else np.nan)
    offset = (i - n_sc/2 + 0.5) * w
    ax.bar(x + offset, means, w*0.9,
           label=SCENARIO_LABELS.get(name, name).replace("\n", " | "),
           color=colors[i], alpha=0.85)

ax.set_xticks(x); ax.set_xticklabels([str(s) for s in sets_all], rotation=45, ha="right", fontsize=8)
ax.set_xlabel("LLC set ID"); ax.set_ylabel("Mean p99_ratio")
ax.set_title("Attack strength per LLC set — grouped by workload scenario\n"
             "Same physical set, different GPU+CPU compute → different signal strength", fontsize=11)
ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.4)
ax.legend(loc="upper right", fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "scenario_per_set_bar.png"), dpi=120); plt.close()
print("Saved: scenario_per_set_bar.png")

# ============================================================
# FIGURE 3 — Overall metric distributions (boxplot)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metric in zip(axes, ["p99_ratio", "max_ratio", "median_ratio"]):
    data_list, xlabels = [], []
    for name, data in scenarios.items():
        vals = []
        for dfs in data.values():
            for df in dfs:
                if metric in df.columns:
                    vals.extend(df[metric].dropna().tolist())
        data_list.append(vals)
        xlabels.append(SCENARIO_LABELS.get(name, name))
    bp = ax.boxplot(data_list, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors): patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_ylabel(metric); ax.set_title(metric)
    ax.axhline(1.0, color="red", ls="--", lw=1, alpha=0.5)
fig.suptitle("Attack metric distributions by workload scenario  |  red = no effect", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "scenario_metric_boxplot.png"), dpi=120); plt.close()
print("Saved: scenario_metric_boxplot.png")

# ============================================================
# FIGURE 4 — Baseline latency: how does GPU workload affect CPU probe baseline?
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))
bp_data, xlabels = [], []
for name, data in scenarios.items():
    vals = []
    for dfs in data.values():
        for df in dfs:
            if "baseline_p99" in df.columns:
                vals.extend(df["baseline_p99"].dropna().tolist())
    bp_data.append(vals); xlabels.append(SCENARIO_LABELS.get(name, name))
bp = ax.boxplot(bp_data, patch_artist=True, showfliers=False)
for patch, c in zip(bp["boxes"], colors): patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_xticklabels(xlabels, fontsize=8)
ax.set_ylabel("Baseline p99 latency (CPU cycles)")
ax.set_title("CPU probe baseline latency per scenario\n"
             "Higher = GPU workload already evicting CPU lines even before attack phase", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "scenario_baseline_latency.png"), dpi=120); plt.close()
print("Saved: scenario_baseline_latency.png")

# ============================================================
# FIGURE 5 — Overlay lines: p99_ratio per address for top 3 sets, all scenarios
# ============================================================
# pick top 3 sets by mean p99_ratio averaged across all scenarios
set_global_mean = {}
for s in sets_all:
    vals = []
    for data in scenarios.values():
        for df in data.get(s, []):
            if "p99_ratio" in df.columns:
                vals.extend(df["p99_ratio"].dropna().tolist())
    set_global_mean[s] = np.mean(vals) if vals else 0
top_sets = sorted(set_global_mean, key=set_global_mean.get, reverse=True)[:3]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
for ax, s in zip(axes, top_sets):
    for i, (name, data) in enumerate(scenarios.items()):
        dfs = data.get(s, [])
        if not dfs: continue
        mat = np.full((len(dfs), max(len(df) for df in dfs)), np.nan)
        for j, df in enumerate(dfs):
            if "p99_ratio" in df.columns:
                v = df["p99_ratio"].values.astype(float)
                mat[j, :len(v)] = v
        mean_line = np.nanmean(mat, axis=0)
        ax.plot(mean_line, color=colors[i], lw=1.5, alpha=0.85,
                label=SCENARIO_LABELS.get(name, name).replace("\n", " | "))
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_title(f"LLC set {s}  (mean p99_ratio = {set_global_mean[s]:.2f}×)", fontsize=9)
    ax.set_xlabel("Address index"); ax.set_ylabel("mean p99_ratio")
axes[0].legend(fontsize=6, loc="upper right")
fig.suptitle("p99_ratio per address — top 3 most-attacked LLC sets, all scenarios overlaid\n"
             "Lines diverging = workload changes which addresses are most vulnerable", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "top_sets_overlay.png"), dpi=120); plt.close()
print("Saved: top_sets_overlay.png")

# ============================================================
# Summary CSV + markdown report
# ============================================================
rows = []
for name, data in scenarios.items():
    p99, maxr, base = [], [], []
    for dfs in data.values():
        for df in dfs:
            if "p99_ratio"    in df.columns: p99.extend(df["p99_ratio"].dropna().tolist())
            if "max_ratio"    in df.columns: maxr.extend(df["max_ratio"].dropna().tolist())
            if "baseline_p99" in df.columns: base.extend(df["baseline_p99"].dropna().tolist())
    rows.append(dict(scenario=name,
                     mean_p99_ratio=round(np.mean(p99),3)  if p99  else None,
                     mean_max_ratio=round(np.mean(maxr),1) if maxr else None,
                     baseline_p99_cycles=round(np.mean(base),0) if base else None))

df_sum = pd.DataFrame(rows)
df_sum.to_csv(os.path.join(OUT_DIR, "scenario_summary.csv"), index=False)

# markdown
md = ["# GPU Workload Heatmap Comparison\n",
      "| Scenario | Mean p99_ratio | Mean max_ratio | Baseline p99 (cycles) |",
      "|---|---|---|---|"]
for _, r in df_sum.iterrows():
    md.append(f"| {r.scenario} | {r.mean_p99_ratio} | {r.mean_max_ratio} | {r.baseline_p99_cycles} |")
md += ["",
       "## Key observations",
       "- Scenarios with large GPU compute (matmul_512, conv2d, bandwidth) have higher baseline latency",
       "  because the GPU workload itself causes managed-memory page migrations that evict CPU lines.",
       "- The p99_ratio (attack/baseline) tells how much *additional* eviction the attack adds on top of",
       "  the workload's natural LLC pressure.",
       "- conv2d and bandwidth workloads tend to show different per-set patterns than pure matmul,",
       "  because they access different memory regions with different LLC footprints."]
with open(os.path.join(OUT_DIR, "scenario_report.md"), "w") as f:
    f.write("\n".join(md))

print("\n===== SCENARIO SUMMARY =====")
print(df_sum.to_string(index=False))
print(f"\nAll outputs saved to: {OUT_DIR}")
