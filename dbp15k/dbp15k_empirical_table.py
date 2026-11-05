#!/usr/bin/env python3
"""dbp15k_empirical_table.py — Final empirical pipeline table for DBP15K (Table 5 fix).

Merges L2 verdicts: original checkpoint (dbp15k_glm5turbo_results.json, 4,876 pairs)
+ rebuild verdicts (dbp15k_rebuild_l2.json). Replays fixed and adaptive cascades:
  intercepted -> L1 label, escalated -> L2 verdict, invalid/unresolved -> strict error.
Reports: strict pipeline accuracy, P/R/F1 (same-class), LLM cost.
Output: results/dbp15k_empirical_pipeline.json
"""
import json, os
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, 'results')

state = json.load(open(os.path.join(RES, 'dbp15k_rebuild_state.json')))
npz = np.load(os.path.join(RES, 'dbp15k_rebuild_state.npz'))
y = npz['y']; fpred = npz['fpred']; apred = npz['apred']
all_pairs = [tuple(p) for p in state['all_pairs']]
N = len(y)

# merged L2 verdict map
old = json.load(open(os.path.join(RES, 'dbp15k_glm5turbo_results.json')))['predictions']
l2 = {}
for p in old:
    v = p.get('l2_prediction', -1)
    try:
        v = int(v)
    except (TypeError, ValueError):
        v = -1
    l2[(p['z'], p['e'])] = v
try:
    new = json.load(open(os.path.join(RES, 'dbp15k_rebuild_l2.json')))
except FileNotFoundError:
    new = {}
def parse_raw(raw):
    r = str(raw).lower().strip()
    if 'match' in r and 'different' not in r:
        return 1
    if 'different' in r:
        return 0
    return -1
for k, v in new.items():
    if '||' in k:
        z, e = k.split('||', 1)
        l2[(z, e)] = parse_raw(v['raw'])

def replay(pred, name):
    final = np.full(N, -2)
    l1_dec = l1_cor = l2_use = l2_cor = unres = 0
    for i in range(N):
        if pred[i] != -1:
            l1_dec += 1
            final[i] = pred[i]
            if pred[i] == y[i]:
                l1_cor += 1
        else:
            z, e, _ = all_pairs[i]
            v = l2.get((z, e), -1)
            l2_use += 1
            if v == -1:
                unres += 1
            else:
                final[i] = v
                if v == y[i]:
                    l2_cor += 1
    f = final.copy()
    for i in np.where(f == -2)[0]:
        f[i] = 1 - y[i]
    tp = int(((f == 1) & (y == 1)).sum()); fp_ = int(((f == 1) & (y == 0)).sum())
    tn = int(((f == 0) & (y == 0)).sum()); fn = int(((f == 0) & (y == 1)).sum())
    prec = tp / (tp + fp_) * 100 if tp + fp_ else 0
    rec = tp / (tp + fn) * 100 if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    out = {
        'l1_decided': l1_dec, 'l1_correct': l1_cor,
        'l1_acc_pct': round(l1_cor / l1_dec * 100, 2) if l1_dec else 0,
        'intercept_pct': round(l1_dec / N * 100, 2),
        'l2_used': l2_use, 'l2_correct': l2_cor, 'unresolved': unres,
        'strict_pipeline_acc_pct': round((l1_cor + l2_cor) / N * 100, 2),
        'llm_cost_pct': round(l2_use / N * 100, 2),
        'tp': tp, 'fp': fp_, 'tn': tn, 'fn': fn,
        'precision_pct': round(prec, 2), 'recall_pct': round(rec, 2), 'f1_pct': round(f1, 2),
    }
    print(name, json.dumps(out))
    return out

results = {
    'fixed': replay(fpred, 'fixed   '),
    'adaptive': replay(apred, 'adaptive'),
}
# pure L2 on the union escalation set is NOT the same as all-pairs L2; report
# all-pairs L2 using every available verdict as reference ceiling
l2_all_cor = sum(1 for i in range(N) if l2.get(all_pairs[i][:2], -1) == y[i])
l2_all_have = sum(1 for i in range(N) if l2.get(all_pairs[i][:2], -1) != -1)
print(f"L2 verdicts available for {l2_all_have}/{N} pairs (ceiling row stays as published: checkpoint all-escalated run)")
results['_note'] = 'Pure-L2 row unchanged (4,843-pair all-escalated run in dbp15k_glm5turbo_results.json). Fixed/adaptive are now empirical replays.'
json.dump(results, open(os.path.join(RES, 'dbp15k_empirical_pipeline.json'), 'w'), indent=1)
print('written results/dbp15k_empirical_pipeline.json')
