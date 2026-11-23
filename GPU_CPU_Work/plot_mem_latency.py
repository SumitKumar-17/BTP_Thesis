"""
plot_mem_latency.py  (v2)
Generates a horizontal bar chart of CPU memory hierarchy latency.
Output: results/mem_latency_plot.pdf  (and .png)
"""
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

# ── Load CSV ───────────────────────────────────────────────────
data_file = "results/mem_latency.csv"
levels, cycles, types = [], [], []
with open(data_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        levels.append(row['level'].strip())
        cycles.append(int(row['median_cycles'].strip()))
        types.append(row['type'].strip())

COLORS = {
    'cpu':     '#2a6496',
    'pagemap': '#d35400',
}
bar_colors = [COLORS[t] for t in types]

# ── Plot ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))

y_pos = np.arange(len(levels))
bars = ax.barh(y_pos, cycles, color=bar_colors, height=0.55,
               edgecolor='white', linewidth=0.8)

# ── Value labels to the right of each bar ─────────────────────
for bar, val in zip(bars, cycles):
    # For Pagemap (longest bar), put label inside the bar
    if bar.get_width() > 400:
        ax.text(bar.get_width() * 0.97, bar.get_y() + bar.get_height()/2,
                f'{val} cycles', va='center', ha='right',
                fontsize=10.5, color='white', fontweight='bold')
    else:
        ax.text(bar.get_width() * 1.08, bar.get_y() + bar.get_height()/2,
                f'{val} cycles', va='center', ha='left',
                fontsize=10.5, color='#222222')

# ── Axis setup ────────────────────────────────────────────────
ax.set_yticks(y_pos)
ax.set_yticklabels(levels, fontsize=12)
ax.invert_yaxis()
ax.set_xlabel('Median Access Latency (CPU cycles, rdtsc)', fontsize=11)
ax.set_title('CPU Memory Hierarchy Latency  —  RTX 3060 Host (Intel CPU)',
             fontsize=13, fontweight='bold', pad=12)

ax.set_xscale('log')
ax.set_xlim(1, 4000)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: str(int(x)) if x >= 1 else ''))
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.grid(axis='x', linestyle='--', alpha=0.35, which='both')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Annotation for Pagemap row ────────────────────────────────
# Place text box above the Pagemap bar, inside the plot
# pagemap_idx = levels.index('Pagemap')
# ax.annotate(
#     '/proc/self/pagemap: one pread() syscall\n'
#     'returns physical frame number (PFN)\n'
#     '→ determines exact LLC set address\n'
#     '→ requires root (Linux ≥ 4.0)',
#     xy=(1, pagemap_idx - 0.0),       # point to left of pagemap bar
#     xytext=(6, pagemap_idx - 1.6),   # text box above the DRAM bar
#     fontsize=8.5,
#     color='#7f2800',
#     arrowprops=dict(arrowstyle='->', color='#d35400', lw=1.3,
#                     connectionstyle='arc3,rad=0.15'),
#     bbox=dict(boxstyle='round,pad=0.4', fc='#fff4ec', ec='#d35400', lw=0.9),
# )

# ── Legend ────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=COLORS['cpu'],
                   label='Hardware cache/DRAM latency (pointer chasing)'),
    mpatches.Patch(color=COLORS['pagemap'],
                   label='Software page-table read (pagemap syscall)'),
]
ax.legend(handles=legend_patches, loc='upper right', fontsize=9,
          framealpha=0.92, edgecolor='#cccccc')

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/mem_latency_plot.pdf", bbox_inches='tight', dpi=200)
plt.savefig("results/mem_latency_plot.png", bbox_inches='tight', dpi=200)
print("Saved: results/mem_latency_plot.pdf  and  results/mem_latency_plot.png")
