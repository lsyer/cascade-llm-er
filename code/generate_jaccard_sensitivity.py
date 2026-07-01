#!/usr/bin/env python3
"""Generate the MINEC Jaccard-threshold sensitivity figure from release results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('/root/workspace/tanshu_docs/experiments/release')
DATA = BASE / 'results' / 'jaccard_sensitivity.json'
OUT_DIR = BASE / 'results'
OUT_DIR.mkdir(exist_ok=True)

results = json.load(open(DATA))
taus = [r['tau'] for r in results]
l0_acc = [r['l0_acc'] for r in results]
pipeline = [r['pipeline'] for r in results]
fp = [r['fp'] for r in results]
fn = [r['fn'] for r in results]

fig, ax1 = plt.subplots(figsize=(7, 4.5))

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

ax2 = ax1.twinx()
color3 = '#FF6B6B'
color4 = '#FFA726'
ax2.plot(taus, fp, '^--', color=color3, linewidth=1.5, markersize=4, alpha=0.8, label='False Merges (FP)')
ax2.plot(taus, fn, 'v--', color=color4, linewidth=1.5, markersize=4, alpha=0.8, label='False Rejects (FN)')
ax2.set_ylabel('Error Count', fontsize=11, color='#666')
ax2.set_ylim(0, 400)
ax2.tick_params(axis='y')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8,
           ncol=2, framealpha=0.9, edgecolor='#ccc')

ax1.annotate(r'Phase transition at $\tau \approx 0.55$' + '\n(strict match/conflict boundary)',
             xy=(0.55, 75.7), xytext=(0.15, 83),
             fontsize=7.5, color='#666',
             arrowprops=dict(arrowstyle='->', color='#999', lw=0.8))

plt.tight_layout()
out_pdf = OUT_DIR / 'jaccard_sensitivity.pdf'
out_png = OUT_DIR / 'jaccard_sensitivity.png'
plt.savefig(out_pdf, format='pdf', bbox_inches='tight', dpi=300)
plt.savefig(out_png, format='png', bbox_inches='tight', dpi=150)
print(f'Saved: {out_pdf}')
plt.close()
