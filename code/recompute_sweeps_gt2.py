#!/usr/bin/env python3
"""recompute_sweeps_gt2.py — Recompute the fixed dual-threshold sweep and the
LR/RF Pareto envelope under ground-truth-v2 labels (final_label), replacing the
old-label values behind fig:threshold-sensitivity and fig:pareto-frontier.

Inputs: scored_pairs.json (l1_score_v2), oof_probabilities.npz (lr/rf),
checkpoints_v3/glm-5-turbo_checkpoint.json, minec_ground_truth_v2.json.
Outputs: results/dual_threshold_sweep_gt2.json, results/pareto_frontier_data_gt2.json
Pipeline convention: strict, unresolved escalated verdicts count incorrect
(EMPIRICAL replay, not the old estimation formula).
"""
import json, os
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(BASE, 'results')

scored = json.load(open(os.path.join(RES, 'scored_pairs.json')))
pids = [p['pair_id'] for p in scored]
gt = json.load(open(os.path.join(BASE, 'data', 'minec_ground_truth_v2.json')))
recs = gt['records'] if isinstance(gt, dict) else gt
gold = {r['pair_id']: 1 if r['final_label'] == 'same' else 0 for r in recs}
ck = json.load(open(os.path.join(RES, 'checkpoints_v3', 'glm-5-turbo_checkpoint.json')))

def glm(p):
    v = str(ck.get(p, {}).get('verdict', '')).strip().lower()
    return 1 if v == 'same' else (0 if v == 'different' else -1)

y = np.array([gold[p] for p in pids])
g = np.array([glm(p) for p in pids])
g_ok = (g == y).astype(int)          # unresolved (-1) -> incorrect (strict)
l2_acc = g_ok.mean() * 100

# ---------- 1) fixed dual-threshold sweep (same 99 configs as archived) ----------
old_sweep = json.load(open(os.path.join(RES, 'dual_threshold_sweep.json')))
s = np.array([p['l1_score_v2'] for p in scored], dtype=float)
rows = []
for r in old_sweep:
    mt, rt = r['merge_threshold'], r['reject_threshold']
    esc = (s < mt) & (s > rt)
    itc = ~esc
    lab = np.where(s >= mt, 1, 0)                       # intercepted label
    l1_ok = int(((lab == y) & itc).sum())
    n_itc = int(itc.sum())
    l2_ok = int(g_ok[esc].sum())
    n = len(y)
    rows.append({
        'merge_threshold': mt, 'reject_threshold': rt,
        'l1_accuracy': round(l1_ok / n_itc * 100, 1) if n_itc else 0.0,
        'interception': round(n_itc / n * 100, 1),
        'merge_n': int((s >= mt).sum()), 'merge_correct': int(((lab == y) & (s >= mt)).sum()),
        'reject_n': int((s <= rt).sum()), 'reject_correct': int(((lab == y) & (s <= rt)).sum()),
        'escalate_n': int(esc.sum()),
        'l2_cost': round(esc.mean() * 100, 1),
        'pipeline': round((l1_ok + l2_ok) / n * 100, 1),
    })
json.dump({'l2_acc_strict_gt2': round(l2_acc, 2), 'rows': rows},
          open(os.path.join(RES, 'dual_threshold_sweep_gt2.json'), 'w'), indent=1)
print(f'sweep: {len(rows)} configs | pure GLM strict GT2 = {l2_acc:.2f}%')
d = {(r['merge_threshold'], r['reject_threshold']): r for r in rows}
dep = d.get((0.6, -0.4))
print('deployed point (0.6,-0.4):', dep)

# ---------- 2) Pareto envelope for LR / RF under GT v2 ----------
d2 = np.load(os.path.join(RES, 'oof_probabilities.npz'))
laba = {'lr': d2['lr_proba'], 'rf': d2['rf_proba']}
out = {'l2_accuracy': round(l2_acc, 1),
       'note': 'GT v2 labels; l1_accuracy on intercepted pairs (argmax label); empirical pipeline'}
for key in ('lr', 'rf'):
    p = laba[key]
    pts = []
    for mt in np.arange(0.50, 1.001, 0.05):
        for rt in np.arange(0.00, 0.501, 0.05):
            esc = (p < mt) & (p > rt)
            itc = ~esc
            lab = (p >= 0.5).astype(int)          # argmax label when intercepting
            l1_ok = int(((lab == y) & itc).sum())
            n_itc = int(itc.sum())
            l2_ok = int(g_ok[esc].sum())
            pts.append({'merge_t': round(float(mt), 2), 'reject_t': round(float(rt), 2),
                        'interception': round(n_itc / len(y) * 100, 1),
                        'l1_accuracy': round(l1_ok / n_itc * 100, 1) if n_itc else 0.0,
                        'pipeline': round((l1_ok + l2_ok) / len(y) * 100, 1)})
    # envelope: per 2pp interception bin, max l1_accuracy
    env = []
    for b in np.arange(0, 100, 2.0):
        inbin = [q for q in pts if abs(q['interception'] - b) < 2.0]
        if inbin:
            env.append(max(inbin, key=lambda q: q['l1_accuracy']))
    out[f'{key}_all_points'] = pts
    out[f'{key}_frontier'] = sorted(env, key=lambda q: q['interception'])
    print(f'{key}: {len(pts)} pts, frontier {len(env)}; best pipeline={max(q["pipeline"] for q in pts)}')
json.dump(out, open(os.path.join(RES, 'pareto_frontier_data_gt2.json'), 'w'), indent=1)
print('saved dual_threshold_sweep_gt2.json + pareto_frontier_data_gt2.json')
