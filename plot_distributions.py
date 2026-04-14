"""
plot_distributions.py
For each AI model, plots two things:
1. p99 ratio across LLC sets (line plot, one line per run) — shows which sets are hot
2. Baseline vs Attack probe timing distribution (CDF) — shows the timing separation

Saves all figures to results/thesis_plots/
"""
import csv, glob, os, collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUT = "results/thesis_plots"
os.makedirs(OUT, exist_ok=True)

MODELS = {
    'tinyyolo_v2':        'TinyYOLO-v2',
    'minitransformer_cpu':'MiniTransformer',
    'mobilenet_v2':       'MobileNet-V2',
    'efficientnet_lite0': 'EfficientNet-Lite0',
    'squeezenet1_0':      'SqueezeNet-1.0',
}

COLORS = ['#2980b9', '#e74c3c', '#27ae60']  # run 1, 2, 3

# ── Collect data ───────────────────────────────────────────────
# data[model][set_id] = list of (baseline_p99, attack_p99, p99_ratio) per line
# We want one ratio per (set, run) — take median ratio across lines in that CSV

def load_model(model_key):
    pattern = f"results/pp_ai_{model_key}_set*.csv"
    files = sorted(glob.glob(pattern))
    # group by set_id
    by_set = collections.defaultdict(list)
    for f in files:
        set_id = int(f.split('_set')[1].replace('.csv',''))
        ratios, b_p99s, a_p99s = [], [], []
        with open(f) as fp:
            for row in csv.DictReader(fp):
                ratios.append(float(row['p99_ratio']))
                b_p99s.append(float(row['baseline_p99']))
                a_p99s.append(float(row['attack_p99']))
        if ratios:
            by_set[set_id].append({
                'p99_ratio_med': float(np.median(ratios)),
                'p99_ratio_max': max(ratios),
                'baseline_p99':  list(b_p99s),
                'attack_p99':    list(a_p99s),
            })
    return by_set

# ── Plot 1: p99 ratio line plot across sets ────────────────────
def plot_ratio_lines(model_key, model_name, by_set):
    sets = sorted(by_set.keys())
    if not sets:
        return

    fig, ax = plt.subplots(figsize=(9, 4))

    # Each "run" is the nth entry in by_set[s] (up to 3 runs)
    max_runs = max(len(v) for v in by_set.values())
    for run_idx in range(max_runs):
        ys = []
        xs = []
        for s in sets:
            entries = by_set[s]
            if run_idx < len(entries):
                xs.append(s)
                ys.append(entries[run_idx]['p99_ratio_max'])
        ax.plot(xs, ys, marker='o', markersize=5,
                color=COLORS[run_idx % len(COLORS)],
                linewidth=1.8, label=f'Run {run_idx+1}', alpha=0.85)

    ax.axhline(y=1.0, color='#888', linewidth=1.2, linestyle='--', label='Baseline (ratio=1.0)')
    ax.axhline(y=1.2, color='#c0392b', linewidth=1.0, linestyle=':', alpha=0.6, label='Threshold 1.2×')

    ax.set_xlabel('LLC Set Index', fontsize=11)
    ax.set_ylabel('Max p99 Ratio  (attack / baseline)', fontsize=11)
    ax.set_title(f'{model_name}  —  p99 Probe Ratio Across LLC Sets  (RTX 3060)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_xlim(min(sets) - 200, max(sets) + 200)
    ax.set_ylim(0.9, max(2.6, ax.get_ylim()[1]))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = f"{OUT}/{model_key}_ratio_lines.pdf"
    plt.savefig(out, bbox_inches='tight', dpi=200)
    plt.savefig(out.replace('.pdf','.png'), bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  Saved {out}")

# ── Plot 2: CDF of probe timing (baseline vs attack) ──────────
def plot_timing_cdf(model_key, model_name, by_set):
    # Collect all baseline and attack p99 values across all sets/runs
    all_base = []
    all_atk  = []
    for s, entries in by_set.items():
        for e in entries:
            all_base.extend(e['baseline_p99'])
            all_atk.extend(e['attack_p99'])

    if not all_base:
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    for vals, label, color, ls in [
        (all_base, 'Baseline (CPU idle)',         '#2980b9', '-'),
        (all_atk,  'Attack (CPU runs inference)', '#c0392b', '-'),
    ]:
        sorted_v = np.sort(vals)
        cdf = np.arange(1, len(sorted_v)+1) / len(sorted_v)
        ax.plot(sorted_v, cdf, color=color, linewidth=2.0,
                linestyle=ls, label=label)

    ax.set_xlabel('Probe Latency (GPU clock cycles)', fontsize=11)
    ax.set_ylabel('CDF', fontsize=11)
    ax.set_title(f'{model_name}  —  Probe Timing Distribution  (RTX 3060)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xlim(left=1500)
    ax.grid(linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = f"{OUT}/{model_key}_timing_cdf.pdf"
    plt.savefig(out, bbox_inches='tight', dpi=200)
    plt.savefig(out.replace('.pdf','.png'), bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  Saved {out}")

# ── Combined grid: all models ratio lines in one figure ────────
def plot_combined_grid(all_data):
    model_keys = list(MODELS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=False)
    axes = axes.flatten()

    for idx, (mkey, mname) in enumerate(MODELS.items()):
        ax = axes[idx]
        by_set = all_data[mkey]
        sets = sorted(by_set.keys())
        if not sets:
            continue

        max_runs = max(len(v) for v in by_set.values())
        for run_idx in range(max_runs):
            xs, ys = [], []
            for s in sets:
                entries = by_set[s]
                if run_idx < len(entries):
                    xs.append(s)
                    ys.append(entries[run_idx]['p99_ratio_max'])
            ax.plot(xs, ys, marker='o', markersize=4,
                    color=COLORS[run_idx % len(COLORS)],
                    linewidth=1.6, label=f'Run {run_idx+1}', alpha=0.85)

        ax.axhline(1.0, color='#888', lw=1.1, ls='--')
        ax.axhline(1.2, color='#c0392b', lw=0.9, ls=':', alpha=0.6)
        ax.set_title(mname, fontsize=11, fontweight='bold')
        ax.set_xlabel('LLC Set', fontsize=9)
        ax.set_ylabel('Max p99 Ratio', fontsize=9)
        ax.set_ylim(0.9, None)
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=8, framealpha=0.9)

    # hide unused subplot
    axes[-1].set_visible(False)

    fig.suptitle('Prime+Probe p99 Ratio Across LLC Sets — RTX 3060  (3 runs per model)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = f"{OUT}/all_models_ratio_grid.pdf"
    plt.savefig(out, bbox_inches='tight', dpi=200)
    plt.savefig(out.replace('.pdf','.png'), bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  Saved {out}")

# ── Run ────────────────────────────────────────────────────────
print("Loading data and generating plots...\n")
all_data = {}
for mkey, mname in MODELS.items():
    print(f"[{mname}]")
    by_set = load_model(mkey)
    all_data[mkey] = by_set
    plot_ratio_lines(mkey, mname, by_set)
    plot_timing_cdf(mkey, mname, by_set)

plot_combined_grid(all_data)
print("\nAll done.")
