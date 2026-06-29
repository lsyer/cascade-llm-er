#!/usr/bin/env python3
"""
gen_threshold_figure.py — Generate dual-threshold sensitivity figure for paper.
Panel (a): pipeline accuracy heatmap (merge × reject thresholds)
Panel (b): L1 accuracy + interception curves at reject=-0.4
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT_DIR = "/root/workspace/tanshu_docs/paper-jiis"
DATA = "/root/workspace/tanshu_docs/experiments/release/results"

sweep = json.load(open(f"{DATA}/dual_threshold_sweep.json"))

# ============================================================
# Panel (a): Pipeline heatmap
# ============================================================
merge_ts = sorted(set(r['merge_threshold'] for r in sweep))
reject_ts = sorted(set(r['reject_threshold'] for r in sweep))

# Build matrix
pipe_matrix = np.full((len(merge_ts), len(reject_ts)), 0.0)
l2cost_matrix = np.full((len(merge_ts), len(reject_ts)), 0.0)

for r in sweep:
    mi = merge_ts.index(r['merge_threshold'])
    ri = reject_ts.index(r['reject_threshold'])
    pipe_matrix[mi, ri] = r['pipeline']
    l2cost_matrix[mi, ri] = r['l2_cost']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Custom colormap: white→blue
cmap = LinearSegmentedColormap.from_list('custom', ['#ffffff', '#2196F3', '#0d47a1'])

im = ax1.imshow(pipe_matrix, aspect='auto', cmap=cmap, vmin=80, vmax=93)

# Mark the chosen operating point (0.6, -0.4) — "Balanced" in cost-accuracy trade-off
# Place marker at top-right corner of cell to avoid covering the number
chosen_mi = merge_ts.index(0.6)
chosen_ri = reject_ts.index(-0.4)
ax1.plot(chosen_ri, chosen_mi - 0.35, 'r*', markersize=8,
         markeredgecolor='red', markerfacecolor='red', zorder=5)

ax1.set_xticks(range(len(reject_ts)))
ax1.set_xticklabels([f'{r:.2f}' for r in reject_ts], fontsize=8)
ax1.set_yticks(range(len(merge_ts)))
ax1.set_yticklabels([f'{m:.2f}' for m in merge_ts], fontsize=8)
ax1.set_xlabel('Reject Threshold', fontsize=10)
ax1.set_ylabel('Merge Threshold', fontsize=10)
ax1.set_title('(a) Pipeline Accuracy (%)', fontsize=11, fontweight='bold')

# Annotate values in heatmap
for i in range(len(merge_ts)):
    for j in range(len(reject_ts)):
        val = pipe_matrix[i, j]
        color = 'white' if val > 88 else 'black'
        ax1.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=6, color=color)

cbar1 = plt.colorbar(im, ax=ax1, shrink=0.8)
cbar1.set_label('Pipeline Acc. (%)', fontsize=9)

# ============================================================
# Panel (b): At reject=-0.4, show L1 acc + interception + pipeline vs merge threshold
# ============================================================
reject_fixed = -0.4
ri = reject_ts.index(reject_fixed)

merge_vals = []
l1_acc_vals = []
intercept_vals = []
pipe_vals = []
l2cost_vals = []

for mi, mt in enumerate(merge_ts):
    r = [x for x in sweep if x['merge_threshold'] == mt and x['reject_threshold'] == reject_fixed][0]
    merge_vals.append(mt)
    l1_acc_vals.append(r['l1_accuracy'])
    intercept_vals.append(r['interception'])
    pipe_vals.append(r['pipeline'])
    l2cost_vals.append(r['l2_cost'])

ax2.plot(merge_vals, l1_acc_vals, 'o-', color='#2196F3', linewidth=2, markersize=6, label='L1 Accuracy')
ax2.plot(merge_vals, pipe_vals, 's-', color='#4CAF50', linewidth=2, markersize=6, label='Pipeline Accuracy')
ax2.plot(merge_vals, intercept_vals, '^--', color='#FF9800', linewidth=1.5, markersize=5, label='L1 Interception Rate')
ax2.axvline(x=0.6, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
ax2.text(0.61, 77, 'θ=0.6', color='red', fontsize=9)

ax2.set_xlabel('Merge Threshold (reject=-0.4)', fontsize=10)
ax2.set_ylabel('Percentage (%)', fontsize=10)
ax2.set_title('(b) Sensitivity at Reject=-0.4', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0.0, 0.45))
ax2.set_ylim(30, 100)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(OUT_DIR, 'threshold_sensitivity.pdf')
plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
plt.savefig(output_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', dpi=150)
print(f"Saved: {output_path}")
plt.close()
