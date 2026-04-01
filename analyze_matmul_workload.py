#!/usr/bin/env python3
"""
Matrix multiply workload analysis.
Loads all baseline/attack trial CSVs from results/matmul_analysis/run_*/
and produces:
  1. Per-trial latency distributions (violin + boxplot)
  2. Attack signal consistency across trials (mean, CI)
  3. CDF comparison baseline vs attack
  4. Spike detection analysis
"""
import os, glob, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as scipy_stats

BASE = Path("/home/rtx3060/Desktop/sumit-GPU-Codes/gpu_cpu_work_1")
MATMUL_DIR = BASE / "results" / "matmul_analysis"
OUT_DIR = BASE / "results" / "matmul_analysis" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_matmul_runs():
    run_dirs = sorted([d for d in MATMUL_DIR.iterdir()
                       if d.is_dir() and d.name.startswith("run_")])
    if not run_dirs:
        raise RuntimeError(f"No run_* dirs in {MATMUL_DIR}")
    print(f"Found {len(run_dirs)} matmul run(s): {[d.name for d in run_dirs]}")
    
    records = []
    for rdir in run_dirs:
        run_id = rdir.name
        for csv_path in sorted(rdir.glob("*.csv")):
            m = re.match(r"(baseline|attack)_trial_(\d+)\.csv", csv_path.name)
            if not m:
                continue
            mode = m.group(1)
            trial = int(m.group(2))
            try:
                df = pd.read_csv(csv_path)
                df["run_id"] = run_id
                df["mode"] = mode
                df["trial"] = trial
                records.append(df)
            except Exception as e:
                print(f"  Warning: {csv_path}: {e}")
    
    if not records:
        raise RuntimeError("No data loaded.")
    all_data = pd.concat(records, ignore_index=True)
    print(f"Total rows: {len(all_data)}")
    return all_data


def compute_trial_stats(df):
    rows = []
    for (run_id, mode, trial), grp in df.groupby(["run_id", "mode", "trial"]):
        lat = grp["latency_cpu_cycles"].values
        median_val = float(np.median(lat))
        rows.append({
            "run_id": run_id,
            "mode": mode,
            "trial": trial,
            "mean": float(np.mean(lat)),
            "median": median_val,
            "std": float(np.std(lat)),
            "p95": float(np.percentile(lat, 95)),
            "p99": float(np.percentile(lat, 99)),
            "p999": float(np.percentile(lat, 99.9)),
            "maxv": float(np.max(lat)),
            "spike_rate": float(np.sum(lat > 10 * median_val) / len(lat)),
            "n": len(lat),
        })
    return pd.DataFrame(rows)


def plot_violin_per_trial(df):
    """Violin plot of latency distribution for each trial (baseline vs attack)."""
    base_df = df[df["mode"] == "baseline"]
    atk_df = df[df["mode"] == "attack"]
    
    trials = sorted(df.trial.unique())
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for ax, sub, color, label in [
        (axes[0], base_df, "#4CAF50", "Baseline (with CPU matmul)"),
        (axes[1], atk_df, "#F44336", "Attack (GPU matmul + contention)"),
    ]:
        data_by_trial = [
            np.log10(np.clip(sub[sub.trial == t]["latency_cpu_cycles"].values, 1, None))
            for t in trials
        ]
        vp = ax.violinplot(data_by_trial, positions=trials, showmedians=True)
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.5)
        ax.set_xticks(trials)
        ax.set_xticklabels([f"T{t}" for t in trials])
        ax.set_ylabel("log10(CPU latency cycles)")
        ax.set_xlabel("Trial")
        ax.set_title(f"{label}\nLatency Distribution per Trial")
        ax.grid(axis="y", alpha=0.3)
    
    plt.suptitle("Matrix Multiply Workload: CPU Probe Latency Distributions", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "violin_per_trial.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_attack_signal_consistency(stats_df):
    """Show mean slowdown ratio ± std across trials."""
    base_stats = stats_df[stats_df["mode"] == "baseline"].set_index("trial")
    atk_stats  = stats_df[stats_df["mode"] == "attack"].set_index("trial")
    
    common_trials = sorted(set(base_stats.index) & set(atk_stats.index))
    
    metrics = [
        ("mean",   "Mean Latency Ratio",  "#2196F3"),
        ("p99",    "P99 Latency Ratio",   "#FF9800"),
        ("maxv",   "Max Latency Ratio",   "#F44336"),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, (metric, label, color) in zip(axes, metrics):
        ratios = []
        for t in common_trials:
            base_val = base_stats.loc[t, metric]
            atk_val  = atk_stats.loc[t, metric]
            ratios.append(atk_val / base_val if base_val > 0 else 0)
        
        x = np.arange(len(common_trials))
        bars = ax.bar(x, ratios, color=color, alpha=0.7)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="No effect")
        
        mean_r = float(np.mean(ratios))
        ax.axhline(mean_r, color=color, linestyle="-", linewidth=2,
                   label=f"Mean = {mean_r:.1f}x")
        
        ax.set_xticks(x)
        ax.set_xticklabels([f"T{t}" for t in common_trials])
        ax.set_ylabel(f"{label} (attack/baseline)")
        ax.set_xlabel("Trial")
        ax.set_title(f"{label}\nConsistency Across Trials")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        
        for bar, r in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{r:.1f}x", ha="center", va="bottom", fontsize=8)
    
    plt.suptitle("Attack Signal Consistency: Matrix Multiply Workload Trials", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "attack_signal_consistency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_cdf_comparison(df):
    """CDF of baseline vs all attack trials together."""
    base_all = df[df["mode"] == "baseline"]["latency_cpu_cycles"].values
    atk_all  = df[df["mode"] == "attack"]["latency_cpu_cycles"].values
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # CDF on linear (clipped)
    clip = int(np.percentile(atk_all, 99))
    for ax, clip_val, title in [
        (axes[0], int(np.percentile(base_all, 99.9)), "Full range (up to 99.9th pct)"),
        (axes[1], clip, "Zoomed (up to attack P99)"),
    ]:
        b_clip = np.clip(base_all, 0, clip_val)
        a_clip = np.clip(atk_all, 0, clip_val)
        
        b_sorted = np.sort(b_clip)
        a_sorted = np.sort(a_clip)
        b_cdf = np.arange(1, len(b_sorted)+1) / len(b_sorted)
        a_cdf = np.arange(1, len(a_sorted)+1) / len(a_sorted)
        
        ax.plot(b_sorted, b_cdf, color="#4CAF50", linewidth=2, label="Baseline")
        ax.plot(a_sorted, a_cdf, color="#F44336", linewidth=2, label="Attack")
        ax.set_xlabel("CPU Latency (cycles)")
        ax.set_ylabel("CDF")
        ax.set_title(f"Latency CDF — {title}")
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle("CDF: Baseline vs Attack (Matrix Multiply Workload)", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "cdf_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_spike_timeline(df):
    """Timeline showing spike events per trial."""
    trials = sorted(df.trial.unique())
    n_trials = len(trials)
    
    fig, axes = plt.subplots(n_trials, 2, figsize=(18, 4 * n_trials), squeeze=False)
    
    for row_idx, t in enumerate(trials):
        for col_idx, mode in enumerate(["baseline", "attack"]):
            ax = axes[row_idx][col_idx]
            sub = df[(df.trial == t) & (df["mode"] == mode)].sort_values("iteration")
            if sub.empty:
                ax.set_visible(False)
                continue
            
            lat = sub["latency_cpu_cycles"].values
            median_val = np.median(lat)
            threshold = 10 * median_val
            spike_mask = lat > threshold
            
            ax.scatter(sub["iteration"], lat, s=3, alpha=0.4,
                       c=["red" if s else "steelblue" for s in spike_mask])
            ax.axhline(threshold, color="orange", linestyle="--",
                       linewidth=1, label=f"10x median ({threshold:.0f})")
            ax.set_yscale("log")
            ax.set_ylabel("log(cycles)")
            ax.set_xlabel("Iteration")
            title = f"Trial {t} — {'Attack' if mode=='attack' else 'Baseline'}"
            title += f" | spikes: {spike_mask.sum()}/{len(lat)} ({100*spike_mask.mean():.1f}%)"
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)
    
    plt.suptitle("Spike Timeline: Matrix Multiply Workload Trials", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "spike_timeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def save_stats(stats_df):
    stats_df.to_csv(OUT_DIR / "trial_stats.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'trial_stats.csv'}")
    
    base = stats_df[stats_df["mode"] == "baseline"]
    atk  = stats_df[stats_df["mode"] == "attack"]
    
    print("\n=== Trial Statistics Summary ===")
    print(f"{'Metric':<12} {'Baseline Mean':>15} {'Attack Mean':>13} {'Ratio':>8}")
    print("-" * 52)
    for metric in ["mean", "median", "p99", "p999", "maxv", "spike_rate"]:
        b_val = base[metric].mean()
        a_val = atk[metric].mean()
        ratio = a_val / b_val if b_val > 0 else float("nan")
        print(f"{metric:<12} {b_val:>15.2f} {a_val:>13.2f} {ratio:>8.2f}x")


def main():
    print("=== Matrix Multiply Workload Analysis ===\n")
    df = load_matmul_runs()
    
    stats_df = compute_trial_stats(df)
    
    print("[1] Violin plots per trial...")
    plot_violin_per_trial(df)
    
    print("[2] Attack signal consistency...")
    plot_attack_signal_consistency(stats_df)
    
    print("[3] CDF comparison...")
    plot_cdf_comparison(df)
    
    print("[4] Spike timeline...")
    plot_spike_timeline(df)
    
    save_stats(stats_df)
    print(f"\nAll outputs in: {OUT_DIR}")

if __name__ == "__main__":
    main()
