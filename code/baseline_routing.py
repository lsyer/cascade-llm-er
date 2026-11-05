#!/usr/bin/env python3
"""baseline_routing.py — External routing-policy baselines with equal-budget
comparison (R2 #4 / R3 D4).

Baselines (all on Unified-LR OOF probabilities, same checkpoint L2 verdicts,
strict convention — identical replay engine as compute_prf1.py):
  B1 random escalation      (L1 argmax on intercepted; seed-swept)
  B2 confidence gating      (escalate if max class prob < c; L1 argmax otherwise)
  B3 entropy gating         (escalate if entropy > h)
  B4 two-threshold pipeline-aware routing (ours; from dual-threshold sweep)
Equal-budget comparison: pipeline accuracy at matched LLM-call budgets.
Output: results/baseline_routing.json
"""
import json, os, re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(BASE, 'results')

scored = json.load(open(os.path.join(RES, 'scored_pairs.json')))
pair_ids = [p['pair_id'] for p in scored]
y = np.array([p['label'] for p in scored])
N = len(y)

ckpt = json.load(open(os.path.join(RES, 'checkpoints_v3', 'glm-5-turbo_checkpoint.json')))
def l2(pid):
    v = str(ckpt.get(pid, {}).get('verdict', '')).strip().lower()
    return 1 if v == 'same' else (0 if v == 'different' else -1)
l2lab = np.array([l2(p) for p in pair_ids])

d = np.load(os.path.join(RES, 'oof_probabilities.npz'))
p = d['lr_proba']
l1_argmax = (p >= 0.5).astype(int)  # L1 decision when intercepting
l2_correct = np.where(l2lab != -1, (l2lab == y).astype(int), 0)
l2_valid = l2lab != -1

def pipeline(escalate_mask):
    """escalate_mask: True -> L2. Returns strict acc, cost, counts."""
    esc = escalate_mask
    int_ = ~esc
    l1_ok = (l1_argmax[int_] == y[int_]).sum()
    l2_ok = l2_correct[esc & l2_valid].sum()
    # strict: unresolved escalated pairs count wrong (no correct verdict)
    return {
        'strict_acc_pct': round((l1_ok + l2_ok) / N * 100, 2),
        'llm_cost_pct': round(esc.mean() * 100, 2),
        'l1_decided': int(int_.sum()), 'l1_correct': int(l1_ok),
        'l2_used': int(esc.sum()), 'l2_correct': int(l2_ok),
        'unresolved': int((esc & ~l2_valid).sum()),
    }

rng = np.random.default_rng(42)

# ---- B1 random escalation: sweep budget ----
b1 = []
perm = rng.permutation(N)
for budget in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
    k = int(N * budget / 100)
    m = np.zeros(N, bool)
    m[perm[:k]] = True
    s = pipeline(m); s.update({'budget_pct': budget, 'baseline': 'random'})
    b1.append(s)

# ---- B2 confidence gating: sweep c (fine, fills every budget) ----
conf = np.maximum(p, 1 - p)
b2 = []
for c in [round(0.5 + 0.002 * i, 4) for i in range(251)]:
    m = conf < c
    if m.sum() == 0:
        continue
    s = pipeline(m); s.update({'confidence_threshold': c, 'baseline': 'confidence'})
    b2.append(s)

# ---- B3 entropy gating: sweep h (fine, fills every budget) ----
eps = 1e-12
ent = -p * np.log2(p + eps) - (1 - p) * np.log2(1 - p + eps)
b3 = []
for h in [round(0.002 * i, 4) for i in range(501)]:
    m = ent > h
    if m.sum() == 0:
        continue
    s = pipeline(m); s.update({'entropy_threshold': h, 'baseline': 'entropy'})
    b3.append(s)

# ---- B4 ours: two-threshold routing, selected points from dual-threshold sweep ----
sw = json.load(open(os.path.join(RES, 'dual_threshold_sweep_gt2.json')))
b4 = []
def thr_pipeline(mt, rt):
    m = (p < mt) & (p > rt)
    return pipeline(m)
# our canonical operating points
for name, mt, rt in [('l1-acc-max (classifier-ablation LR)', 0.91, 0.03),
                     ('pipeline-aware (Table 2 mode3)', None, None)]:
    if mt is not None:
        s = thr_pipeline(mt, rt); s.update({'baseline': f'ours:{name}'})
        b4.append(s)
# plus a budget-matched sweep of ours: pick thresholds achieving various interception
if isinstance(sw, dict):
    pass  # sweep format inspected below if needed

out = {
    'B1_random': b1, 'B2_confidence': b2, 'B3_entropy': b3, 'B4_ours': b4,
    'replay_reference': {'t4_lr_strict': 91.47, 't4_lr_cost': 89.7,
                         'mode3_strict': 91.51, 'mode3_cost': 69.9},
}
json.dump(out, open(os.path.join(RES, 'baseline_routing_gt2.json'), 'w'), indent=1, ensure_ascii=False)

def show(rows, key):
    print(f"--- {key} ---")
    for r in rows:
        print(f"  cost={r['llm_cost_pct']:5.1f}%  strict={r['strict_acc_pct']:6.2f}%  unres={r['unresolved']}")
show(b1, 'B1 random')
show(b2, 'B2 confidence')
show(b3, 'B3 entropy')
show(b4, 'B4 ours')

# ---- Equal-budget table (GT v2): per-budget best under hard cost cap ----
# Rule: for each budget b, pick the configuration with cost <= b that maximizes
# strict accuracy. Gates are swept finely (0.002 steps) so each budget can be
# filled; Ours = best point of the LR two-threshold frontier (pareto gt2).
# Ours: fine-grid (0.01) two-threshold sweep, hard cost cap — same treatment
# as the finely-swept gates (results/ours_fine_grid_gt2.json)
ours_pts = [{'cost': q['cost'], 's': q['s'], 'mt': q['mt'], 'rt': q['rt']}
            for q in json.load(open(os.path.join(RES, 'ours_fine_grid_gt2.json')))]
budget_table = {}
for b in (20, 30, 50, 70, 90):
    row = {}
    cands = [i for i in b1 if i['budget_pct'] <= b]
    row['random'] = max(cands, key=lambda i: i['strict_acc_pct'])['strict_acc_pct'] if cands else None
    for name, sweep in (('confidence', b2), ('entropy', b3)):
        cands = [i for i in sweep if i['llm_cost_pct'] <= b]
        row[name] = max(cands, key=lambda i: i['strict_acc_pct'])['strict_acc_pct'] if cands else None
    cands = [i for i in ours_pts if i['cost'] <= b]
    best = max(cands, key=lambda i: i['s'])
    row['ours'] = best['s']
    row['ours_point'] = {'mt': best['mt'], 'rt': best['rt'], 'cost': round(best['cost'], 1)}
    budget_table[b] = row
    print(f"budget {b}%: rand={row['random']} conf={row['confidence']} "
          f"ent={row['entropy']} ours={row['ours']} @cost {row['ours_point']['cost']}")
json.dump(budget_table, open(os.path.join(RES, 'budget_table_gt2.json'), 'w'), indent=1)
