#!/usr/bin/env python3
"""Verify: pipeline accuracy never exceeds pure-L2-on-all, for any threshold.

Uses REAL L2 verdicts from checkpoint (not estimated).
For each (merge_t, reject_t) on each L1 scorer (Fixed/LR/RF):
  intercepted = L1 decides (score in decisive range)
  escalated   = rest
  pipeline_acc = (L1_correct on intercepted + L2_correct on escalated) / N
  pure_l2_acc = L2_correct on ALL / N   (constant = 91.7%)

Question: does pipeline_acc ever exceed pure_l2_acc?
"""
import json, os, re
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), '..')
CKPT = os.path.join(BASE, 'results', 'checkpoints_v3', 'glm-5-turbo_checkpoint.json')
SCORED = os.path.join(BASE, 'results', 'scored_pairs.json')
OOF = os.path.join(BASE, 'results', 'oof_probabilities.npz')
OUT = os.path.join(BASE, 'results', 'pipeline_vs_pure_l2.json')

with open(CKPT) as f:
    ckpt = json.load(f)
with open(SCORED) as f:
    scored = json.load(f)

pair_ids = [p['pair_id'] for p in scored]
y = np.array([p['label'] for p in scored])
fixed_scores = np.array([p['l1_score_v2'] for p in scored])
N = len(y)

# Parse L2 verdicts for ALL pairs
def parse_verdict(v):
    if isinstance(v, dict):
        v = v.get('verdict', v.get('raw', ''))
    v = str(v).strip().lower()
    if re.search(r'\bsame\b|\bmatch\b|\bidentical\b|yes|是', v):
        return 1
    if re.search(r'\bdifferent\b|\bnot.*match\b|\bdistinct\b|no|否|不同', v):
        return 0
    return None

l2_preds = np.full(N, -1)
for i, pid in enumerate(pair_ids):
    entry = ckpt.get(pid)
    if entry:
        l2_preds[i] = parse_verdict(entry.get('verdict', entry.get('raw', '')))

l2_resolved_mask = l2_preds != -1
l2_correct_all = (l2_preds[l2_resolved_mask] == y[l2_resolved_mask]).sum()
pure_l2_acc = l2_correct_all / N * 100  # divide by N (treat unknown as wrong) OR by resolved?
pure_l2_acc_resolved = l2_correct_all / l2_resolved_mask.sum() * 100
print(f"N = {N}")
print(f"L2 resolved: {l2_resolved_mask.sum()}, unknown: {(~l2_resolved_mask).sum()}")
print(f"Pure L2 acc (resolved denom): {l2_correct_all}/{l2_resolved_mask.sum()} = {pure_l2_acc_resolved:.1f}%")
print(f"Pure L2 acc (N denom, unknown=wrong): {l2_correct_all}/{N} = {pure_l2_acc:.1f}%")

# For pipeline: escalated pairs use L2. If L2 unknown on some escalated pair -> counts as wrong (strict)
l2_correct_arr = np.where(l2_resolved_mask, (l2_preds == y).astype(int), 0)  # 1=correct, 0=wrong/unknown

# ---- Sweep function ----
def sweep_scorer(scores, y, merge_grid, reject_grid, score_type='score'):
    """For score-domain (fixed): intercept if score >= merge_t OR score <= reject_t.
       For proba-domain (LR/RF): intercept if proba >= merge_t OR proba <= reject_t."""
    results = []
    for mt in merge_grid:
        for rt in reject_grid:
            if score_type == 'score':
                decided_mask = (scores >= mt) | (scores <= rt)
            else:
                decided_mask = (scores >= mt) | (scores <= rt)
            if rt >= mt and score_type != 'score':
                continue
            escalated_mask = ~decided_mask
            n_decided = decided_mask.sum()
            if n_decided == 0:
                continue
            # L1 prediction on decided
            if score_type == 'score':
                l1_preds = np.where(scores >= mt, 1, np.where(scores <= rt, 0, -1))
            else:
                l1_preds = np.where(scores >= mt, 1, np.where(scores <= rt, 0, -1))
            l1_correct = (l1_preds[decided_mask] == y[decided_mask]).sum()
            # L2 correct on escalated
            l2_correct_esc = l2_correct_arr[escalated_mask].sum()
            pipeline_correct = l1_correct + l2_correct_esc
            pipeline_acc = pipeline_correct / N * 100
            intercept = n_decided / N * 100
            l1_acc = l1_correct / n_decided * 100
            results.append({
                'merge_t': round(float(mt), 3),
                'reject_t': round(float(rt), 3),
                'interception': round(intercept, 1),
                'l1_accuracy': round(l1_acc, 1),
                'pipeline_empirical': round(pipeline_acc, 2),
                'l2_cost_pct': round(100 - intercept, 1),
            })
    return results

# Fixed L1: score domain [-1, 1]
fixed_merge_grid = np.arange(0.40, 1.01, 0.05)
fixed_reject_grid = np.arange(-0.50, -0.04, 0.05)
fixed_results = sweep_scorer(fixed_scores, y, fixed_merge_grid, fixed_reject_grid, 'score')

# LR / RF: proba domain [0, 1]
d = np.load(OOF)
lr_proba, rf_proba = d['lr_proba'], d['rf_proba']
proba_merge_grid = np.arange(0.50, 1.001, 0.05)
proba_reject_grid = np.arange(0.00, 0.501, 0.05)
lr_results = sweep_scorer(lr_proba, y, proba_merge_grid, proba_reject_grid, 'proba')
rf_results = sweep_scorer(rf_proba, y, proba_merge_grid, proba_reject_grid, 'proba')

# ---- Analysis: does pipeline ever exceed pure L2? ----
print(f"\n{'='*70}")
print(f"PURE L2 (all {N} pairs) = {pure_l2_acc_resolved:.1f}% (resolved) / {pure_l2_acc:.1f}% (strict, unknown=wrong)")
print(f"{'='*70}")

for name, results in [('Fixed L1', fixed_results), ('Unified LR', lr_results), ('Random Forest', rf_results)]:
    best = max(results, key=lambda x: x['pipeline_empirical'])
    over = [r for r in results if r['pipeline_empirical'] > pure_l2_acc_resolved]
    print(f"\n{name}: {len(results)} configs")
    print(f"  Best pipeline: {best['pipeline_empirical']:.2f}% @ intercept={best['interception']}%, l1_acc={best['l1_accuracy']}% (mt={best['merge_t']}, rt={best['reject_t']})")
    print(f"  Configs with pipeline > pure_L2 ({pure_l2_acc_resolved:.1f}%): {len(over)}")
    if over:
        over_best = max(over, key=lambda x: x['pipeline_empirical'])
        print(f"    Max excess: {over_best['pipeline_empirical']:.2f}% (delta +{over_best['pipeline_empirical']-pure_l2_acc_resolved:.2f}pp)")

# Save
output = {
    'pure_l2_acc_resolved': round(pure_l2_acc_resolved, 2),
    'pure_l2_acc_strict': round(pure_l2_acc, 2),
    'n_pairs': N,
    'fixed_results': fixed_results,
    'lr_results': lr_results,
    'rf_results': rf_results,
}
with open(OUT, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {OUT}")
