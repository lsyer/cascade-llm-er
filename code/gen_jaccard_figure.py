#!/usr/bin/env python3
"""Generate Jaccard threshold sensitivity figure for paper."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA = "/root/workspace/tanshu_docs/experiments/release/results/jaccard_fine_sweep.json"
OUT = "/root/workspace/tanshu_docs/paper-jiis"

results = json.load(open(DATA))
taus = [r['tau'] for r in results]
l0_acc = [r['l0_acc'] for r in results]
pipeline = [r['pipeline'] for r in results]
fp = [r['fp'] for r in results]
fn = [r['fn'] for r in results]
total_err = [r['total_errors'] for r in results]
merge_prec = [r['merge_prec'] for r in results]

fig, ax1 = plt.subplots(figsize=(7, 4.5))

# Left axis: L0 accuracy and pipeline
color1 = '#2196F3'
color2 = '#4CAF50'
ax1.plot(taus, l0_acc, 'o-', color=color1, linewidth=2, markersize=5, label='L1 Accuracy (fixed weights)', zorder=5)
ax1.plot(taus, pipeline, 's-', color=color2, linewidth=2, markersize=5, label='Pipeline Accuracy', zorder=5)
ax1.set_xlabel(r'Weak-field Jaccard threshold $\tau$', fontsize=11)
ax1.set_ylabel('Accuracy (%)', fontsize=11, color='#333')
ax1.set_ylim(60, 92)
ax1.set_xlim(0, 1.0)
ax1.axvline(x=0.65, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
ax1.text(0.66, 91, r'$\tau=0.65$', color='red', fontsize=9)
ax1.grid(True, alpha=0.2)
ax1.tick_params(axis='y')

# Right axis: error counts
ax2 = ax1.twinx()
color3 = '#FF6B6B'
color4 = '#FFA726'
ax2.plot(taus, fp, '^--', color=color3, linewidth=1.5, markersize=4, alpha=0.8, label='False Merges (FP)')
ax2.plot(taus, fn, 'v--', color=color4, linewidth=1.5, markersize=4, alpha=0.8, label='False Rejects (FN)')
ax2.set_ylabel('Error Count', fontsize=11, color='#666')
ax2.set_ylim(0, 400)
ax2.tick_params(axis='y')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8,
           ncol=2, framealpha=0.9, edgecolor='#ccc')

# Annotate the transition zone
ax1.annotate(r'Phase transition at $\tau \approx 0.55$' + '\n(strict match/conflict boundary)',
            xy=(0.55, 75.7), xytext=(0.15, 83),
            fontsize=7.5, color='#666',
            arrowprops=dict(arrowstyle='->', color='#999', lw=0.8))

plt.tight_layout()
plt.savefig(f"{OUT}/jaccard_sensitivity.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.savefig(f"{OUT}/jaccard_sensitivity.png", format='png', bbox_inches='tight', dpi=150)
print(f"Saved: {OUT}/jaccard_sensitivity.pdf")
plt.close()
