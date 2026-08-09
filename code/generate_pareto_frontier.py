#!/usr/bin/env python3
"""Sweep dual probability thresholds on LR/RF OOF probabilities.
For each (merge_t, reject_t) pair, compute interception rate and L1 accuracy
on the intercepted subset. Build the upper-envelope Pareto frontier:
for each interception level, the max achievable L1 accuracy.

Also loads the Fixed L1 dual-threshold sweep for comparison.

Outputs:
  - pareto_frontier_data.json  (all sweep points)
  - pareto_frontier.pdf        (the figure: 3 curves + L2 line)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), '..')
OOF  = os.path.join(BASE, 'results', 'oof_probabilities.npz')
FIXED_SWEEP = os.path.join(BASE, 'results', 'dual_threshold_sweep.json')
OUT_JSON = os.path.join(BASE, 'results', 'pareto_frontier_data.json')
OUT_PDF  = os.path.join(BASE, 'results', 'pareto_frontier.pdf')

# ---- Load OOF ----
d = np.load(OOF)
lr_proba, rf_proba, y = d['lr_proba'], d['rf_proba'], d['y']
N = len(y)
print(f"N = {N}")

L2_ACC = 91.6  # GLM-5-Turbo strict accuracy on ALL 2591 pairs (6 unresolved counted as incorrect)

# ---- Sweep thresholds for LR/RF ----
# probability grid: 0.50 to 0.95, step 0.05
# merge_t >= reject_t always; intercepted = proba >= merge_t OR proba <= (1 - merge_t_complement)
# Actually: intercept (decide) if proba >= merge_t (predict 1) or proba <= reject_t (predict 0)
# escalate if reject_t < proba < merge_t
def sweep_oof(proba, y, merge_grid, reject_grid):
    points = []
    for mt in merge_grid:
        for rt in reject_grid:
            if rt >= mt:
                continue  # invalid: reject >= merge
            decided_mask = (proba >= mt) | (proba <= rt)
            n_decided = decided_mask.sum()
            if n_decided == 0:
                continue
            preds = np.where(proba >= mt, 1, 0)  # for decided, those >=mt -> 1, those <=rt -> 0
            # but decided_mask includes both; for <=rt predict 0
            preds = np.where(proba <= rt, 0, preds)
            correct = (preds[decided_mask] == y[decided_mask]).sum()
            acc = correct / n_decided * 100
            intercept = n_decided / N * 100
            # pipeline accuracy: intercepted use L1 pred, escalated use L2
            # P(correct) = intercept/100 * acc/100 + (1 - intercept/100) * L2_ACC/100
            pipeline = intercept / 100 * acc / 100 + (1 - intercept / 100) * L2_ACC / 100
            points.append({
                'merge_t': round(float(mt), 3),
                'reject_t': round(float(rt), 3),
                'interception': round(intercept, 1),
                'l1_accuracy': round(acc, 1),
                'pipeline': round(pipeline * 100, 1),
            })
    return points

merge_grid = np.arange(0.50, 1.001, 0.05)
reject_grid = np.arange(0.00, 0.501, 0.05)

lr_points = sweep_oof(lr_proba, y, merge_grid, reject_grid)
rf_points = sweep_oof(rf_proba, y, merge_grid, reject_grid)
print(f"LR sweep points: {len(lr_points)}")
print(f"RF sweep points: {len(rf_points)}")

# ---- Fixed L1 sweep (already computed) ----
with open(FIXED_SWEEP) as f:
    fixed_sweep = json.load(f)
fixed_points = []
for r in fixed_sweep:
    fixed_points.append({
        'merge_t': r['merge_threshold'],
        'reject_t': r['reject_threshold'],
        'interception': r['interception'],
        'l1_accuracy': r['l1_accuracy'],
        'pipeline': r['pipeline'],
    })
print(f"Fixed sweep points: {len(fixed_points)}")

# ---- Build Pareto frontier (upper envelope): for each interception bin, max accuracy ----
def pareto_envelope(points, bin_width=2.0):
    """For each interception bin, find the point with max l1_accuracy."""
    intercepts = [p['interception'] for p in points]
    lo, hi = min(intercepts), max(intercepts)
    bins = np.arange(lo, hi + bin_width, bin_width)
    frontier = []
    for b in bins:
        in_bin = [p for p in points if abs(p['interception'] - b) < bin_width]
        if not in_bin:
            continue
        best = max(in_bin, key=lambda p: p['l1_accuracy'])
        frontier.append(best)
    return sorted(frontier, key=lambda p: p['interception'])

lr_frontier = pareto_envelope(lr_points)
rf_frontier = pareto_envelope(rf_points)
fixed_frontier = pareto_envelope(fixed_points)

print(f"\nLR frontier points: {len(lr_frontier)}")
print(f"RF frontier points: {len(rf_frontier)}")
print(f"Fixed frontier points: {len(fixed_frontier)}")

# ---- Save JSON ----
output = {
    'description': 'Pareto frontier: L1 accuracy vs interception rate for three L1 strategies. L2 (GLM-5-Turbo) = 90.7% horizontal.',
    'l2_accuracy': L2_ACC,
    'lr_all_points': lr_points,
    'rf_all_points': rf_points,
    'fixed_all_points': fixed_points,
    'lr_frontier': lr_frontier,
    'rf_frontier': rf_frontier,
    'fixed_frontier': fixed_frontier,
}
with open(OUT_JSON, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {OUT_JSON}")

# ---- Plot ----
fig, ax = plt.subplots(figsize=(8, 5.5))

def plot_frontier(ax, frontier, color, label, marker):
    xs = [p['interception'] for p in frontier]
    ys = [p['l1_accuracy'] for p in frontier]
    ax.plot(xs, ys, color=color, linewidth=2.2, marker=marker, markersize=5, label=label, zorder=3)

plot_frontier(ax, fixed_frontier, '#2196F3', 'Fixed L1 (hand-tuned score)', 'o')
plot_frontier(ax, lr_frontier, '#E53935', 'Unified LR (OOF, 5-fold CV)', 's')
plot_frontier(ax, rf_frontier, '#43A047', 'Random Forest (OOF, 5-fold CV)', '^')

# L2 horizontal line
ax.axhline(y=L2_ACC, color='#9E9E9E', linestyle='--', linewidth=1.5, alpha=0.8, zorder=2)
ax.text(2, L2_ACC + 0.5, f'L2: GLM-5-Turbo = {L2_ACC}%', fontsize=9, color='#616161')

ax.set_xlabel('L1 Interception Rate (%)', fontsize=12)
ax.set_ylabel('L1 Accuracy on Intercepted Pairs (%)', fontsize=12)
ax.set_title('L1 Accuracy vs. Interception Rate (Pareto Frontier)', fontsize=13, fontweight='bold')
ax.set_xlim(0, 100)
ax.set_ylim(60, 100)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PDF, format='pdf', bbox_inches='tight', dpi=300)
plt.savefig(OUT_PDF.replace('.pdf', '.png'), format='png', bbox_inches='tight', dpi=150)
print(f"Saved: {OUT_PDF}")
plt.close()
