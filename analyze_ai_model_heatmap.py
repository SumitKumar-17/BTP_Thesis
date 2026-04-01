#!/usr/bin/env python3
"""Analyze LLC attack heatmaps across 5 AI model workloads."""
import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings; warnings.filterwarnings("ignore")

BASE    = "/home/rtx3060/Desktop/sumit-GPU-Codes/gpu_cpu_work_1/results/ai_model_heatmap"
OUT_DIR = os.path.join(BASE, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_LABELS = {
    "MobileNetV2":        "MobileNetV2\n(~3.4M params)",
    "SqueezeNet1_0":      "SqueezeNet1_0\n(~1.2M params)",
    "EfficientNet_Lite0": "EfficientNet-Lite0\n(~4.7M params)",
    "DistilBERT":         "DistilBERT\n(~66M params)",
    "TinyYOLOv2":         "TinyYOLOv2\n(~15M params)",
}

def load_model(mdir):
    per_set = {}
    for rdir in sorted(glob.glob(os.path.join(mdir, "run_*"))):
        for csv in sorted(glob.glob(os.path.join(rdir, "scan_set_*.csv"))):
            m = re.search(r"scan_set_(\d+)\.csv", csv)
            if not m: continue
            try:
                df = pd.read_csv(csv)
                per_set.setdefault(int(m.group(1)), []).append(df)
            except: pass
    return per_set

def mean_matrix(per_set, metric):
    sets = sorted(per_set.keys())
    rows = []
    for s in sets:
        dfs = per_set[s]
        if not dfs or metric not in dfs[0].columns:
            rows.append(np.array([np.nan])); continue
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

# ---- load ----
models = {}
for mdir in sorted(glob.glob(os.path.join(BASE, "*"))):
    if not os.path.isdir(mdir) or os.path.basename(mdir) == "analysis": continue
    name = os.path.basename(mdir)
    data = load_model(mdir)
    if data:
        models[name] = data
        print(f"Loaded {name}: {len(data)} sets, {max(len(v) for v in data.values())} runs")

if not models:
    print("No model data found."); exit(1)

n_m = len(models)
colors = plt.cm.tab10(np.linspace(0, 0.8, n_m))

# ============================================================
# FIGURE 1 — Heatmap grid (one column per model, mean p99_ratio)
# ============================================================
fig, axes = plt.subplots(1, n_m, figsize=(6*n_m, 8), sharey=True)
if n_m == 1: axes = [axes]

all_mats = {}
for name, data in models.items():
    mat, sets = mean_matrix(data, "p99_ratio")
    all_mats[name] = (mat, sets)

vmax = max(np.nanpercentile(
    np.concatenate([m.ravel()[np.isfinite(m.ravel())] for m, _ in all_mats.values()]), 98), 2.0)

for ax, (name, (mat, sets)) in zip(axes, all_mats.items()):
    im = ax.imshow(mat, aspect="auto", origin="upper",
                   norm=mcolors.LogNorm(vmin=0.5, vmax=vmax), cmap="YlOrRd")
    ax.set_title(MODEL_LABELS.get(name, name), fontsize=9, fontweight="bold")
    ax.set_xlabel("Address index", fontsize=8)
    if ax == axes[0]:
        ax.set_ylabel("LLC set ID")
        ax.set_yticks(range(len(sets)))
        ax.set_yticklabels([str(s) for s in sets], fontsize=7)

plt.colorbar(im, ax=axes[-1], label="mean p99_ratio (log scale)")
fig.suptitle("GPU→CPU LLC attack: p99 ratio heatmap — 5 lightweight AI models\n"
             "Model runs GPU inference continuously while CPU probes measure LLC evictions",
             fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ai_model_heatmap_grid.png"), dpi=120); plt.close()
print("Saved: ai_model_heatmap_grid.png")

# ============================================================
# FIGURE 2 — Per-set bar chart grouped by model
# ============================================================
sets_all = sorted({s for data in models.values() for s in data})
x = np.arange(len(sets_all)); w = 0.8 / n_m

fig, ax = plt.subplots(figsize=(16, 5))
for i, (name, data) in enumerate(models.items()):
    means = []
    for s in sets_all:
        vals = [df["p99_ratio"].mean() for df in data.get(s, []) if "p99_ratio" in df.columns]
        means.append(np.mean(vals) if vals else np.nan)
    ax.bar(x + (i - n_m/2 + 0.5)*w, means, w*0.9,
           label=MODEL_LABELS.get(name, name).replace("\n", " "),
           color=colors[i], alpha=0.85)

ax.set_xticks(x); ax.set_xticklabels([str(s) for s in sets_all], rotation=45, ha="right", fontsize=8)
ax.set_xlabel("LLC set ID"); ax.set_ylabel("Mean p99_ratio")
ax.set_title("Attack strength per LLC set — grouped by AI model\n"
             "Which model causes the most LLC eviction pressure per set?", fontsize=11)
ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.4)
ax.legend(fontsize=7, ncol=3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ai_model_per_set_bar.png"), dpi=120); plt.close()
print("Saved: ai_model_per_set_bar.png")

# ============================================================
# FIGURE 3 — Metric distribution boxplots
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metric in zip(axes, ["p99_ratio", "max_ratio", "median_ratio"]):
    data_list, xlabels = [], []
    for name, data in models.items():
        vals = [v for dfs in data.values() for df in dfs
                if metric in df.columns for v in df[metric].dropna().tolist()]
        data_list.append(vals)
        xlabels.append(MODEL_LABELS.get(name, name))
    bp = ax.boxplot(data_list, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors): patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_ylabel(metric); ax.set_title(metric)
    ax.axhline(1.0, color="red", ls="--", lw=1, alpha=0.5)
fig.suptitle("Attack signal distribution by AI model  |  red = no effect (ratio=1)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ai_model_metric_boxplot.png"), dpi=120); plt.close()
print("Saved: ai_model_metric_boxplot.png")

# ============================================================
# FIGURE 4 — Baseline latency comparison per model
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))
bp_data, xlabels = [], []
for name, data in models.items():
    vals = [v for dfs in data.values() for df in dfs
            if "baseline_p99" in df.columns for v in df["baseline_p99"].dropna().tolist()]
    bp_data.append(vals); xlabels.append(MODEL_LABELS.get(name, name))
bp = ax.boxplot(bp_data, patch_artist=True, showfliers=False)
for patch, c in zip(bp["boxes"], colors): patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_xticklabels(xlabels, fontsize=8)
ax.set_ylabel("Baseline p99 latency (CPU cycles)")
ax.set_title("CPU probe baseline latency while each AI model runs on GPU\n"
             "Higher = model's GPU memory traffic already evicting CPU LLC lines", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ai_model_baseline_latency.png"), dpi=120); plt.close()
print("Saved: ai_model_baseline_latency.png")

# ============================================================
# FIGURE 5 — Per-run heatmaps for each model (Run1|Run2|Run3)
# ============================================================
CMP_DIR = os.path.join(BASE, "per_run_comparison")
os.makedirs(CMP_DIR, exist_ok=True)

for name, data in models.items():
    model_dir = os.path.join(BASE, name)
    run_dirs  = sorted(glob.glob(os.path.join(model_dir, "run_*")))
    run_mats  = []
    for rdir in run_dirs:
        frames = {}
        for csv in sorted(glob.glob(os.path.join(rdir, "scan_set_*.csv"))):
            m2 = re.search(r"scan_set_(\d+)\.csv", csv)
            if not m2: continue
            try: frames[int(m2.group(1))] = pd.read_csv(csv)
            except: pass
        if frames:
            sets2 = sorted(frames.keys())
            rows2 = []
            for s in sets2:
                df = frames[s]
                rows2.append(df["p99_ratio"].values.astype(float) if "p99_ratio" in df.columns else np.array([np.nan]))
            ml = max(len(r) for r in rows2)
            mat2 = np.full((len(sets2), ml), np.nan)
            for ii, r in enumerate(rows2): mat2[ii, :len(r)] = r
            run_mats.append((os.path.basename(rdir), mat2, sets2))

    if not run_mats: continue
    n = len(run_mats)
    all_v = np.concatenate([m.ravel()[np.isfinite(m.ravel())] for _, m, _ in run_mats])
    vmax2 = max(np.nanpercentile(all_v, 98), 1.5) if len(all_v) else 2.0

    fig, axes2 = plt.subplots(1, n, figsize=(7*n, 6), sharey=True)
    if n == 1: axes2 = [axes2]
    for ax, (rl, mat2, sets2) in zip(axes2, run_mats):
        im = ax.imshow(mat2, aspect="auto", origin="upper",
                       norm=mcolors.LogNorm(vmin=0.5, vmax=vmax2), cmap="YlOrRd")
        ax.set_title(re.sub(r"run_(\d+)_.*", r"Run \1", rl), fontsize=10, fontweight="bold")
        ax.set_xlabel("Address index")
        if ax == axes2[0]:
            ax.set_ylabel("LLC set ID")
            ax.set_yticks(range(len(sets2)))
            ax.set_yticklabels([str(s) for s in sets2], fontsize=7)
    plt.colorbar(im, ax=axes2[-1], label="p99_ratio (log)")
    fig.suptitle(f"{MODEL_LABELS.get(name, name)} — p99_ratio heatmap, all {n} runs", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(CMP_DIR, f"{name}_runs_comparison.png"), dpi=120); plt.close()
    print(f"Saved per-run comparison: {name}")

# ============================================================
# MEGA grid: all models x all runs
# ============================================================
print("Building mega-grid...")
model_run_mats = {}
for name, data in models.items():
    model_dir = os.path.join(BASE, name)
    mats = []
    for rdir in sorted(glob.glob(os.path.join(model_dir, "run_*"))):
        frames = {}
        for csv in sorted(glob.glob(os.path.join(rdir, "scan_set_*.csv"))):
            m2 = re.search(r"scan_set_(\d+)\.csv", csv)
            if not m2: continue
            try: frames[int(m2.group(1))] = pd.read_csv(csv)
            except: pass
        if frames:
            sets2 = sorted(frames.keys())
            rows2 = []
            for s in sets2:
                df = frames[s]
                rows2.append(df["p99_ratio"].values.astype(float) if "p99_ratio" in df.columns else np.array([np.nan]))
            ml = max(len(r) for r in rows2)
            mat2 = np.full((len(sets2), ml), np.nan)
            for ii, r in enumerate(rows2): mat2[ii, :len(r)] = r
            mats.append(mat2)
    if mats: model_run_mats[name] = (mats, sets2)

n_models = len(model_run_mats)
n_runs   = max(len(v[0]) for v in model_run_mats.values())
all_v = np.concatenate([m.ravel()[np.isfinite(m.ravel())]
                        for mats, _ in model_run_mats.values() for m in mats])
vmax3 = max(np.nanpercentile(all_v, 98), 1.5) if len(all_v) else 2.0

fig, axes3 = plt.subplots(n_models, n_runs, figsize=(7*n_runs, 5*n_models), squeeze=False)
for row, (name, (mats, sets2)) in enumerate(model_run_mats.items()):
    for col in range(n_runs):
        ax = axes3[row][col]
        if col < len(mats):
            im = ax.imshow(mats[col], aspect="auto", origin="upper",
                           norm=mcolors.LogNorm(vmin=0.5, vmax=vmax3), cmap="YlOrRd")
            ax.set_title(f"Run {col+1}", fontsize=9)
            ax.set_xlabel("addr idx", fontsize=7)
            if col == 0:
                ax.set_ylabel(MODEL_LABELS.get(name, name) + "\nLLC set", fontsize=8)
                ax.set_yticks(range(len(sets2)))
                ax.set_yticklabels([str(s) for s in sets2], fontsize=6)
            else:
                ax.set_yticks([])
        else:
            ax.axis("off")

plt.colorbar(im, ax=axes3[:, -1].tolist(), label="p99_ratio (log)", shrink=0.5)
fig.suptitle("p99_ratio — ALL 5 AI models × ALL 3 runs\nRows=models  Columns=runs",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(CMP_DIR, "MEGA_all_models_all_runs.png"), dpi=100, bbox_inches="tight")
plt.close(); print("Saved: MEGA_all_models_all_runs.png")

# ============================================================
# Summary CSV
# ============================================================
rows = []
for name, data in models.items():
    p99 = [v for dfs in data.values() for df in dfs if "p99_ratio" in df.columns
           for v in df["p99_ratio"].dropna().tolist()]
    maxr = [v for dfs in data.values() for df in dfs if "max_ratio" in df.columns
            for v in df["max_ratio"].dropna().tolist()]
    base = [v for dfs in data.values() for df in dfs if "baseline_p99" in df.columns
            for v in df["baseline_p99"].dropna().tolist()]
    rows.append(dict(model=name,
                     mean_p99_ratio=round(np.mean(p99), 3) if p99 else None,
                     mean_max_ratio=round(np.mean(maxr), 1) if maxr else None,
                     baseline_p99_cycles=round(np.mean(base), 0) if base else None))

df_sum = pd.DataFrame(rows)
df_sum.to_csv(os.path.join(OUT_DIR, "ai_model_summary.csv"), index=False)
print("\n===== AI MODEL SUMMARY =====")
print(df_sum.to_string(index=False))
print(f"\nAll outputs saved to: {OUT_DIR}")

# ============================================================
# FIGURE 6 — Individual per-run heatmaps (one PNG per model per run)
# ============================================================
PERRUN_DIR = os.path.join(BASE, "per_run_individual")
os.makedirs(PERRUN_DIR, exist_ok=True)

for name, data in models.items():
    model_dir = os.path.join(BASE, name)
    run_dirs  = sorted(glob.glob(os.path.join(model_dir, "run_*")))
    for run_idx, rdir in enumerate(run_dirs, start=1):
        frames = {}
        for csv in sorted(glob.glob(os.path.join(rdir, "scan_set_*.csv"))):
            m2 = re.search(r"scan_set_(\d+)\.csv", csv)
            if not m2: continue
            try: frames[int(m2.group(1))] = pd.read_csv(csv)
            except: pass
        if not frames: continue

        sets2 = sorted(frames.keys())
        rows2 = []
        for s in sets2:
            df = frames[s]
            rows2.append(df["p99_ratio"].values.astype(float) if "p99_ratio" in df.columns else np.array([np.nan]))
        ml = max(len(r) for r in rows2)
        mat2 = np.full((len(sets2), ml), np.nan)
        for ii, r in enumerate(rows2): mat2[ii, :len(r)] = r

        all_v2 = mat2.ravel()[np.isfinite(mat2.ravel())]
        vmax_r = max(np.nanpercentile(all_v2, 98), 1.5) if len(all_v2) else 2.0

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(mat2, aspect="auto", origin="upper",
                       norm=mcolors.LogNorm(vmin=0.5, vmax=vmax_r), cmap="YlOrRd")
        ax.set_title(f"{MODEL_LABELS.get(name, name)} — Run {run_idx}  |  p99_ratio heatmap",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Address index within LLC set")
        ax.set_ylabel("LLC set ID")
        ax.set_yticks(range(len(sets2)))
        ax.set_yticklabels([str(s) for s in sets2], fontsize=8)
        plt.colorbar(im, ax=ax, label="p99_ratio (log scale)")
        plt.tight_layout()
        fname = f"{name}_run{run_idx}_heatmap.png"
        plt.savefig(os.path.join(PERRUN_DIR, fname), dpi=120); plt.close()
        print(f"Saved individual: {fname}")

print(f"\nPer-run individual heatmaps -> {PERRUN_DIR}")
