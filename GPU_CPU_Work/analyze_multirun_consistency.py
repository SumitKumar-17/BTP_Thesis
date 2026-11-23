#!/usr/bin/env python3
"""
Multi-run consistency analysis for eviction-set heatmap experiments.
Loads all run_* directories under results/multirun_heatmap/,
computes mean/std/CV per (set, address), generates consistency plots.
"""
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/home/rtx3060/Desktop/sumit-GPU-Codes/gpu_cpu_work_1")
HEATMAP_DIR = BASE / "results" / "multirun_heatmap"
OUT_DIR = BASE / "results" / "multirun_heatmap" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load all runs ────────────────────────────────────────────────────────────
def load_runs():
    run_dirs = sorted([d for d in HEATMAP_DIR.iterdir()
                       if d.is_dir() and d.name.startswith("run_")])
    if not run_dirs:
        raise RuntimeError(f"No run_* directories found in {HEATMAP_DIR}")
    print(f"Found {len(run_dirs)} runs: {[d.name for d in run_dirs]}")
    
    records = []
    for run_dir in run_dirs:
        run_id = run_dir.name
        for csv_path in sorted(run_dir.glob("scan_set_*.csv")):
            m = re.search(r"scan_set_(\d+)\.csv$", csv_path.name)
            if not m:
                continue
            set_id = int(m.group(1))
            try:
                df = pd.read_csv(csv_path)
                df["run_id"] = run_id
                df["set_id"] = set_id
                records.append(df)
            except Exception as e:
                print(f"  Warning: could not read {csv_path}: {e}")
    
    if not records:
        raise RuntimeError("No CSV data loaded from any run.")
    
    all_data = pd.concat(records, ignore_index=True)
    print(f"Total rows loaded: {len(all_data)}")
    return all_data, run_dirs


# ── Per-set summary across runs ──────────────────────────────────────────────
def per_set_summary(df):
    """For each set: mean and std of key ratios across runs."""
    rows = []
    for set_id, grp in df.groupby("set_id"):
        for metric in ("p99_ratio", "max_ratio", "median_ratio"):
            if metric not in grp.columns:
                continue
            vals = grp[metric].values
            rows.append({
                "set_id": set_id,
                "metric": metric,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "cv": float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else 0,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n": len(vals),
            })
    return pd.DataFrame(rows)


# ── Per-address consistency ──────────────────────────────────────────────────
def per_address_consistency(df):
    """For each (set_id, order): mean and std of p99_ratio and max_ratio."""
    rows = []
    for (set_id, order), grp in df.groupby(["set_id", "order"]):
        for metric in ("p99_ratio", "max_ratio"):
            if metric not in grp.columns:
                continue
            vals = grp[metric].values
            rows.append({
                "set_id": set_id,
                "order": order,
                "metric": metric,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "cv": float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else 0,
            })
    return pd.DataFrame(rows)


# ── Plot 1: Per-set mean p99_ratio across runs (bar chart with error bars) ──
def plot_per_set_mean(summary_df):
    p99 = summary_df[summary_df.metric == "p99_ratio"].sort_values("set_id")
    maxr = summary_df[summary_df.metric == "max_ratio"].sort_values("set_id")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    
    for ax, data, label, color in [
        (axes[0], p99, "P99 Ratio (attack/baseline)", "#2196F3"),
        (axes[1], maxr, "Max Ratio (attack/baseline)", "#F44336"),
    ]:
        x = np.arange(len(data))
        ax.bar(x, data["mean"], yerr=data["std"], capsize=4,
               color=color, alpha=0.7, label="mean ± std across runs")
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{s}" for s in data["set_id"]], rotation=45, ha="right")
        ax.set_ylabel(label)
        ax.set_xlabel("LLC Set ID")
        ax.set_title(f"{label} — Mean ± Std Across {int(data['n'].max() // 64)} Runs")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="No effect (ratio=1)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    out = OUT_DIR / "per_set_mean_std.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 2: Consistency heatmap (CV per set × address for p99_ratio) ────────
def plot_consistency_heatmap(addr_df):
    for metric in ("p99_ratio", "max_ratio"):
        sub = addr_df[addr_df.metric == metric]
        sets = sorted(sub.set_id.unique())
        max_orders = int(sub.order.max()) + 1
        
        mat = np.full((len(sets), max_orders), np.nan)
        for i, s in enumerate(sets):
            sg = sub[sub.set_id == s]
            for _, row in sg.iterrows():
                mat[i, int(row["order"])] = row["mean"]
        
        cv_mat = np.full((len(sets), max_orders), np.nan)
        for i, s in enumerate(sets):
            sg = sub[sub.set_id == s]
            for _, row in sg.iterrows():
                cv_mat[i, int(row["order"])] = row["cv"]
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        # Mean heatmap
        im0 = axes[0].imshow(mat, aspect="auto", cmap="hot",
                             interpolation="nearest",
                             vmin=1.0, vmax=float(np.nanpercentile(mat, 95)))
        plt.colorbar(im0, ax=axes[0], label=f"Mean {metric}")
        axes[0].set_yticks(range(len(sets)))
        axes[0].set_yticklabels([str(s) for s in sets], fontsize=7)
        axes[0].set_xlabel("Address Order Within Set")
        axes[0].set_ylabel("LLC Set ID")
        axes[0].set_title(f"Mean {metric} Across Runs")
        
        # CV heatmap (consistency: low CV = consistent)
        vmax_cv = float(np.nanpercentile(cv_mat, 95)) if not np.all(np.isnan(cv_mat)) else 1
        im1 = axes[1].imshow(cv_mat, aspect="auto", cmap="RdYlGn_r",
                             interpolation="nearest",
                             vmin=0, vmax=vmax_cv)
        plt.colorbar(im1, ax=axes[1], label="CV (std/mean)")
        axes[1].set_yticks(range(len(sets)))
        axes[1].set_yticklabels([str(s) for s in sets], fontsize=7)
        axes[1].set_xlabel("Address Order Within Set")
        axes[1].set_ylabel("LLC Set ID")
        axes[1].set_title(f"Consistency (CV) of {metric}\n(green=consistent, red=noisy)")
        
        plt.tight_layout()
        out = OUT_DIR / f"consistency_heatmap_{metric}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


# ── Plot 3: Per-run overlay per set (spaghetti plot) ─────────────────────────
def plot_perrun_overlay(df, n_sets_to_show=6):
    """Show all runs overlaid for a few representative sets."""
    sets = sorted(df.set_id.unique())
    chosen = sets[:n_sets_to_show]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    runs = df.run_id.unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    
    for ax_idx, set_id in enumerate(chosen):
        ax = axes[ax_idx]
        set_data = df[df.set_id == set_id].sort_values("order")
        
        for r_idx, run_id in enumerate(runs):
            run_data = set_data[set_data.run_id == run_id].sort_values("order")
            ax.plot(run_data["order"], run_data["p99_ratio"],
                    color=colors[r_idx], alpha=0.6, linewidth=1,
                    label=f"Run {r_idx+1}")
        
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"LLC Set {set_id} — P99 Ratio per Run")
        ax.set_xlabel("Address Order")
        ax.set_ylabel("P99 Ratio (attack/baseline)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    
    plt.suptitle("Per-Address P99 Ratio Consistency Across Runs", fontsize=13)
    plt.tight_layout()
    out = OUT_DIR / "perrun_overlay_p99.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 4: Boxplot of ratios per set across all runs ────────────────────────
def plot_set_boxplot(df):
    sets = sorted(df.set_id.unique())
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    for ax, metric, color, title in [
        (axes[0], "p99_ratio", "#2196F3", "P99 Ratio"),
        (axes[1], "max_ratio", "#F44336", "Max Ratio"),
    ]:
        box_data = [df[df.set_id == s][metric].values for s in sets]
        bp = ax.boxplot(box_data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_xticks(range(1, len(sets) + 1))
        ax.set_xticklabels([f"S{s}" for s in sets], rotation=45, ha="right")
        ax.set_ylabel(f"{title} (attack/baseline)")
        ax.set_xlabel("LLC Set ID")
        ax.set_title(f"{title} Distribution per Set (All Runs Combined)")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    out = OUT_DIR / "set_boxplot_across_runs.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Summary table ─────────────────────────────────────────────────────────────
def save_summary(summary_df, addr_df):
    # Per-set summary
    pivot = summary_df.pivot_table(
        index="set_id",
        columns="metric",
        values=["mean", "std", "cv"]
    )
    pivot.to_csv(OUT_DIR / "per_set_summary.csv")
    print(f"Saved: {OUT_DIR / 'per_set_summary.csv'}")
    
    # Top 5 most vulnerable sets (by mean p99_ratio)
    p99_means = summary_df[summary_df.metric == "p99_ratio"].sort_values("mean", ascending=False)
    print("\nTop 5 most vulnerable LLC sets (by mean P99 ratio):")
    print(p99_means[["set_id", "mean", "std", "cv"]].head(5).to_string(index=False))
    
    # Consistency stats
    cv_p99 = addr_df[addr_df.metric == "p99_ratio"]["cv"]
    print(f"\nP99 ratio CV (consistency): mean={cv_p99.mean():.3f}, median={cv_p99.median():.3f}")
    print(f"  (CV < 0.3 = consistent, CV > 0.7 = noisy)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== Multi-Run Consistency Analysis ===\n")
    df, run_dirs = load_runs()
    
    print("\n[1] Per-set summary...")
    summary_df = per_set_summary(df)
    
    print("[2] Per-address consistency...")
    addr_df = per_address_consistency(df)
    
    print("[3] Plotting per-set mean±std bar chart...")
    plot_per_set_mean(summary_df)
    
    print("[4] Plotting consistency heatmaps...")
    plot_consistency_heatmap(addr_df)
    
    print("[5] Plotting per-run overlay...")
    plot_perrun_overlay(df)
    
    print("[6] Plotting set boxplots across runs...")
    plot_set_boxplot(df)
    
    print("[7] Saving summary table...")
    save_summary(summary_df, addr_df)
    
    print(f"\nAll analysis outputs in: {OUT_DIR}")

if __name__ == "__main__":
    main()
