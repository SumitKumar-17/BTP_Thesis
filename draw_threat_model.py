"""
draw_threat_model.py  (v4 — fixed overlaps)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

def rbox(ax, x, y, w, h, fc, ec, lw=1.8, r=0.22, z=2, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha))

def t(ax, x, y, s, fs=9.5, c='#222', bold=False, ha='center', va='center', z=6):
    ax.text(x, y, s, fontsize=fs, color=c,
            fontweight='bold' if bold else 'normal',
            ha=ha, va=va, zorder=z)

def arr(ax, x1, y1, x2, y2, c, lw=2.0, style='->', rad=0.0, ls='solid', z=5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=c, lw=lw,
                        linestyle=ls,
                        connectionstyle=f'arc3,rad={rad}'), zorder=z)

# ── Outer host frame ───────────────────────────────────────────
rbox(ax, 0.2, 1.6, 13.6, 7.0, '#f7f7f7', '#444', lw=2.8, r=0.4, z=1)
t(ax, 7.0, 8.3, 'Physical Host Machine  —  Shared Hardware',
  fs=13, c='#333', bold=True)

# ══════════════════════════════════
# TENANT A  (left)
# ══════════════════════════════════
rbox(ax, 0.5, 2.0, 3.9, 6.2, '#ebf5fb', '#2980b9', lw=2.0, r=0.28)
t(ax, 2.45, 7.9,  'Tenant A  (Victim)',  fs=11.5, c='#1a5276', bold=True)
t(ax, 2.45, 7.55, 'CPU-only allocation', fs=9,    c='#666')

# CPU inner box
rbox(ax, 0.8, 2.5, 3.3, 3.8, '#d6eaf8', '#2980b9', lw=1.5, r=0.18)
t(ax, 2.45, 6.0,  'CPU',                  fs=11,  c='#1a5276', bold=True)
t(ax, 2.45, 5.62, 'AI Model Inference',   fs=9.5, c='#2c3e50')
t(ax, 2.45, 5.28, '(CNN / Transformer)',  fs=8.5, c='#666')
t(ax, 2.45, 4.94, 'model weights in DRAM',fs=8.5, c='#666')
t(ax, 2.45, 4.60, 'forward pass on CPU',  fs=8.5, c='#666')

# Step 2 tag inside CPU box
rbox(ax, 0.9, 2.6, 3.1, 1.1, '#aed6f1', '#2980b9', lw=1.2, r=0.14, z=3)
t(ax, 2.45, 3.32, 'Step 2: runs inference',    fs=9,   c='#1a5276', bold=True)
t(ax, 2.45, 2.92, 'evicts GPU lines from LLC', fs=8.5, c='#c0392b', bold=True)

# ══════════════════════════════════
# TENANT B  (right)
# ══════════════════════════════════
rbox(ax, 9.6, 2.0, 3.9, 6.2, '#fef9e7', '#d4ac0d', lw=2.0, r=0.28)
t(ax, 11.55, 7.9,  'Tenant B  (Attacker)', fs=11.5, c='#7d6608', bold=True)
t(ax, 11.55, 7.55, 'CPU + GPU allocation', fs=9,    c='#666')

# GPU inner box
rbox(ax, 9.9, 2.5, 3.3, 3.8, '#fdebd0', '#d35400', lw=1.5, r=0.18)
t(ax, 11.55, 6.0,  'GPU (Attacker)',       fs=11,  c='#784212', bold=True)
t(ax, 11.55, 5.62, 'CUDA UVM pages',       fs=9.5, c='#2c3e50')
t(ax, 11.55, 5.28, 'Prime+Probe kernels',  fs=9.5, c='#2c3e50')
t(ax, 11.55, 4.94, 'clock64() timing',     fs=8.5, c='#666')
t(ax, 11.55, 4.60, 'pagemap phys addr',    fs=8.5, c='#666')

# Steps 1 & 3 tag inside GPU box
rbox(ax, 10.0, 2.6, 3.1, 1.1, '#f0b27a', '#d35400', lw=1.2, r=0.14, z=3)
t(ax, 11.55, 3.32, 'Step 1: Prime LLC sets',        fs=9,   c='#784212', bold=True)
t(ax, 11.55, 2.92, 'Step 3: Probe — measure timing',fs=8.5, c='#c0392b', bold=True)

# ══════════════════════════════════
# SHARED LLC  (centre)
# ══════════════════════════════════
rbox(ax, 5.05, 4.0, 3.9, 3.4, '#fdedec', '#c0392b', lw=2.5, r=0.25, z=3)
t(ax, 7.0, 7.1,  'Shared LLC',        fs=12.5, c='#922b21', bold=True)
t(ax, 7.0, 6.72, 'physically shared', fs=9.5,  c='#555')
t(ax, 7.0, 6.36, 'NO isolation',      fs=10.5, c='#c0392b', bold=True)
t(ax, 7.0, 5.98, 'RTX 3060: 12 MB, 16-way', fs=8.5, c='#777')
t(ax, 7.0, 5.62, 'A30:  60 MB, 15-way',     fs=8.5, c='#777')
t(ax, 7.0, 5.22, r'S = $\lfloor$P/64$\rfloor$ mod N$_{sets}$', fs=8.5, c='#999')
t(ax, 7.0, 4.62, 'attack surface',    fs=9.5, c='#c0392b', bold=True)
t(ax, 7.0, 4.22, 'both processors compete\nfor same cache sets', fs=8, c='#888')

# ══════════════════════════════════
# CPU DRAM  (bottom centre)
# ══════════════════════════════════
rbox(ax, 5.1, 1.75, 3.8, 1.1, '#eafaf1', '#1e8449', lw=1.8, r=0.18)
t(ax, 7.0, 2.43, 'CPU DRAM',                   fs=11,  c='#1e8449', bold=True)
t(ax, 7.0, 2.03, 'model weights + UVM pages',  fs=8.5, c='#666')

# ══════════════════════════════════
# ARROWS
# ══════════════════════════════════
# CPU <-> LLC
arr(ax, 4.1, 5.2, 5.05, 5.2, '#2980b9', lw=2.2, style='<->')

# GPU <-> LLC via PCIe
arr(ax, 8.95, 5.2, 9.9, 5.2, '#d35400', lw=2.2, style='<->')
t(ax, 9.42, 5.55, 'PCIe', fs=9.5, c='#d35400', bold=True)

# LLC <-> DRAM
arr(ax, 7.0, 4.0, 7.0, 2.85, '#1e8449', lw=1.8, style='<->')
t(ax, 7.6, 3.4, 'LLC\nmiss', fs=8.5, c='#1e8449', ha='center')

# ══════════════════════════════════
# Side-channel result arrow
# Dashed red: from LLC → GPU (timing signal)
# Placed ABOVE the tenant boxes cleanly
# ══════════════════════════════════
arr(ax, 8.0, 7.5, 10.5, 7.0, '#c0392b', lw=2.2,
    style='->', rad=-0.25, ls='dashed', z=7)

# Label for the dashed arrow — in its own clear space above LLC
rbox(ax, 4.5, 7.7, 5.0, 0.65, '#fff0ee', '#c0392b', lw=1.3, r=0.12, z=8)
t(ax, 7.0, 8.05,
  'Side-channel: slow probe timing  =  victim was in this LLC set',
  fs=9.5, c='#922b21', bold=True, z=9)

# ══════════════════════════════════
# Bottom flow strip
# ══════════════════════════════════
rbox(ax, 0.3, 1.6, 13.4, 0.55, '#ececec', '#aaa', lw=1.2, r=0.1, z=4)
t(ax, 7.0, 1.9,
  'Attack flow:    Step 1 — GPU Primes LLC      '
  'Step 2 — CPU Victim runs & evicts      '
  'Step 3 — GPU Probes & measures timing',
  fs=9.5, c='#333')

# ══════════════════════════════════
# Legend — bottom right, outside tenant boxes
# ══════════════════════════════════
legend_items = [
    mpatches.Patch(fc='#d6eaf8', ec='#2980b9', lw=1.5, label='Tenant A — CPU victim'),
    mpatches.Patch(fc='#fdebd0', ec='#d35400', lw=1.5, label='Tenant B — GPU attacker (PCIe)'),
    mpatches.Patch(fc='#fdedec', ec='#c0392b', lw=1.5, label='Shared LLC — no isolation'),
    mpatches.Patch(fc='#eafaf1', ec='#1e8449', lw=1.5, label='CPU DRAM'),
]
ax.legend(handles=legend_items, loc='lower right', fontsize=9,
          framealpha=0.97, edgecolor='#bbb',
          bbox_to_anchor=(0.99, 0.07))

plt.tight_layout(pad=0.1)
os.makedirs("22CS30056_BTP_2_Final_Report/Pictures", exist_ok=True)
plt.savefig("22CS30056_BTP_2_Final_Report/Pictures/threat_model.png",
            dpi=200, bbox_inches='tight')
plt.savefig("22CS30056_BTP_2_Final_Report/Pictures/threat_model.pdf",
            dpi=200, bbox_inches='tight')
print("Saved.")
