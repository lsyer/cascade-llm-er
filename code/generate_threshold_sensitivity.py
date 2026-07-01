#!/usr/bin/env python3
"""Generate the MINEC dual-threshold sensitivity figure from release results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BASE = Path('/root/workspace/tanshu_docs/experiments/release')
DATA = BASE / 'results' / 'dual_threshold_sweep.json'
OUT_DIR = BASE / 'results'
OUT_DIR.mkdir(exist_ok=True)

sweep = json.load(open(DATA))

merge_ts = sorted(set(r['merge_threshold'] for r in sweep))
reject_ts = sorted(set(r['reject_threshold'] for r in sweep))

pipe_matrix = np.full((len(merge_ts), len(reject_ts)), 0.0)
for r in sweep:
    mi = merge_ts.index(r['merge_threshold'])
    ri = reject_ts.index(r['reject_threshold'])
    pipe_matrix[mi, ri] = r['pipeline']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

cmap = LinearSegmentedColormap.from_list('cascade_blues', ['#ffffff', '#2196F3', '#0d47a1'])
im = ax1.imshow(pipe_matrix, aspect='auto', cmap=cmap, vmin=80, vmax=93)

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

for i in range(len(merge_ts)):
    for j in range(len(reject_ts)):
        val = pipe_matrix[i, j]
        color = 'white' if val > 88 else 'black'
        ax1.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=6, color=color)

cbar1 = plt.colorbar(im, ax=ax1, shrink=0.8)
cbar1.set_label('Pipeline Acc. (%)', fontsize=9)

reject_fixed = -0.4
merge_vals = []
l1_acc_vals = []
intercept_vals = []
pipe_vals = []

for mt in merge_ts:
    r = next(x for x in sweep if x['merge_threshold'] == mt and x['reject_threshold'] == reject_fixed)
    merge_vals.append(mt)
    l1_acc_vals.append(r['l1_accuracy'])
    intercept_vals.append(r['interception'])
    pipe_vals.append(r['pipeline'])

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
out_pdf = OUT_DIR / 'threshold_sensitivity.pdf'
out_png = OUT_DIR / 'threshold_sensitivity.png'
plt.savefig(out_pdf, format='pdf', bbox_inches='tight', dpi=300)
plt.savefig(out_png, format='png', bbox_inches='tight', dpi=150)
print(f'Saved: {out_pdf}')
plt.close()
