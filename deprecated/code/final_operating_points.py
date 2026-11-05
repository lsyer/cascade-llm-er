#!/usr/bin/env python3
"""Final operating points with 0.01-fine grid: Fixed/LR/RF.
L1-acc-max on full OOF (no leakage), intercept>=10%.
Pipeline accuracy uses real L2 verdicts."""
import json, os, re
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), '..')
CKPT = os.path.join(BASE, 'results', 'checkpoints_v3', 'glm-5-turbo_checkpoint.json')
SCORED = os.path.join(BASE, 'results', 'scored_pairs.json')
OOF = os.path.join(BASE, 'results', 'oof_probabilities.npz')
OUT = os.path.join(BASE, 'results', 'final_operating_points.json')

with open(SCORED) as f:
    scored = json.load(f)
pair_ids = [p['pair_id'] for p in scored]
y = np.array([p['label'] for p in scored])
fixed_scores = np.array([p['l1_score_v2'] for p in scored])
N = len(y)

def parse_verdict(v):
    if isinstance(v, dict):
        v = v.get('verdict', v.get('raw', ''))
    v = str(v).strip().lower()
    if re.search(r'\bsame\b|\bmatch\b|\bidentical\b|yes|是', v): return 1
    if re.search(r'\bdifferent\b|\bnot.*match\b|\bdistinct\b|no|否|不同', v): return 0
    return None

with open(CKPT) as f:
    ckpt = json.load(f)
l2_preds = np.full(N, -1)
for i, pid in enumerate(pair_ids):
    entry = ckpt.get(pid)
    if entry:
        l2_preds[i] = parse_verdict(entry.get('verdict', entry.get('raw', '')))
l2_correct_arr = np.where(l2_preds != -1, (l2_preds == y).astype(int), 0)
pure_l2_acc = l2_correct_arr.sum() / N * 100

d = np.load(OOF)
lr_p, rf_p = d['lr_proba'], d['rf_proba']

def select_and_pipeline(scores, grid_m, grid_r, name):
    best = None; best_acc = -1
    all_cands = []
    for mt in grid_m:
        for rt in grid_r:
            if rt >= mt: continue
            decided = (scores >= mt) | (scores <= rt)
            if decided.sum() < 5: continue
            intercept = decided.sum() / N * 100
            if intercept < 10.0: continue
            preds = np.where(scores >= mt, 1, np.where(scores <= rt, 0, -1))
            l1_correct = (preds[decided] == y[decided]).sum()
            acc = l1_correct / decided.sum() * 100
            all_cands.append((mt, rt, intercept, acc))
            if acc > best_acc:
                best_acc = acc
                best = (mt, rt, intercept, acc, l1_correct, int(decided.sum()))

    # pipeline at best point
    mt, rt, intercept, l1_acc, l1_correct, n_dec = best
    decided_mask = (scores >= mt) | (scores <= rt)
    l2_correct_esc = l2_correct_arr[~decided_mask].sum()
    pipeline = (l1_correct + l2_correct_esc) / N * 100

    all_cands.sort(key=lambda x: -x[3])
    print(f"\n{name}: mt={mt:.2f} rt={rt:.2f} intercept={intercept:.1f}% l1_acc={l1_acc:.1f}% pipeline={pipeline:.2f}%")
    print(f"  top-5 candidates:")
    for c in all_cands[:5]:
        print(f"    mt={c[0]:.2f} rt={c[1]:.2f} intercept={c[2]:.1f}% acc={c[3]:.1f}%")
    return {'merge_t': round(float(mt),2), 'reject_t': round(float(rt),2),
            'l1_accuracy': round(l1_acc,1), 'interception': round(intercept,1),
            'pipeline_empirical': round(pipeline,2), 'llm_cost': round(100-intercept,1),
            'top5': [{'mt':c[0],'rt':c[1],'intercept':c[2],'acc':c[3]} for c in all_cands[:5]]}

# 0.01 grid
grid_m = [round(x, 2) for x in np.arange(0.50, 1.001, 0.01)]
grid_r_proba = [round(x, 2) for x in np.arange(0.00, 0.501, 0.01)]
grid_m_score = [round(x, 2) for x in np.arange(0.00, 1.001, 0.01)]
grid_r_score = [round(x, 2) for x in np.arange(-1.00, 0.001, 0.01)]

print(f"Pure L2 ceiling: {pure_l2_acc:.2f}%")

results = {
    'Fixed L1': select_and_pipeline(fixed_scores, grid_m_score, grid_r_score, 'Fixed L1'),
    'Unified LR': select_and_pipeline(lr_p, grid_m, grid_r_proba, 'Unified LR'),
    'Random Forest': select_and_pipeline(rf_p, grid_m, grid_r_proba, 'Random Forest'),
}
results['Pure L2'] = {'merge_t': None, 'reject_t': None, 'l1_accuracy': None,
                      'interception': 0, 'pipeline_empirical': round(pure_l2_acc,2), 'llm_cost': 100}

print(f"\n{'='*70}")
print(f"FINAL TABLE (0.01 grid, OOF, intercept>=10%)")
print(f"{'='*70}")
print(f"{'Strategy':<16} {'mt':>6} {'rt':>6} {'L1Acc':>7} {'Interc':>8} {'Pipe':>8} {'Cost':>7}")
for name, p in results.items():
    mt = f"{p['merge_t']:.2f}" if p['merge_t'] is not None else '—'
    rt = f"{p['reject_t']:.2f}" if p['reject_t'] is not None else '—'
    la = f"{p['l1_accuracy']:.1f}" if p['l1_accuracy'] is not None else '—'
    print(f"{name:<16} {mt:>6} {rt:>6} {la:>7} {p['interception']:>7.1f}% {p['pipeline_empirical']:>7.2f}% {p['llm_cost']:>6.1f}%")

with open(OUT, 'w') as f:
    json.dump({'grid_step': 0.01, 'selection_rule': 'L1 accuracy maximized on full OOF, interception>=10%',
               'pure_l2_ceiling': round(pure_l2_acc,2), **results}, f, indent=2)
print(f"\nSaved: {OUT}")
