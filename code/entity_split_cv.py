#!/usr/bin/env python3
"""entity_split_cv.py — Entity-disjoint cross-validation (R2 detailed comment #3).

Protocol: connected components of the candidate-pair entity co-occurrence graph
are assigned whole to one of 5 folds (zero entity overlap between train/test).
Per fold: LR trained on the other folds (42 features, same config as
run_experiments.py); thresholds selected ON TRAIN via the same empirical
pipeline-aware objective (grid 0.05..0.95 step 0.05, min_l1_acc=75, maximize
strict pipeline with interception tiebreak); applied to the held-out fold;
escalated pairs resolved by the GLM-5-Turbo checkpoint (strict convention).

Reports side-by-side with the random-split OOF results.
Output: results/entity_split_cv.json
"""
import json, os, re
import numpy as np
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(BASE, 'results')

scored = json.load(open(os.path.join(RES, 'scored_pairs.json')))
pair_ids = [p['pair_id'] for p in scored]
y = np.array([p['label'] for p in scored])
types = np.array([p['type_a'].replace('activity', 'event') for p in scored])
N = len(y)

all_feature_names = set()
for p in scored:
    all_feature_names.update(p['features_v2'].keys())
feature_names = sorted(all_feature_names)
X = np.array([[p['features_v2'].get(f, 0) for f in feature_names] for p in scored])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ckpt = json.load(open(os.path.join(RES, 'checkpoints_v3', 'glm-5-turbo_checkpoint.json')))
def l2_label(pid):
    v = ckpt.get(pid, {}).get('verdict', '')
    v = str(v).strip().lower()
    if v == 'same': return 1
    if v == 'different': return 0
    return -1
l2lab = np.array([l2_label(p) for p in pair_ids])

# ---- entity components ----
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[parent[x]] = parent[parent[x]]
        x = parent[x]
    return x
vids = [p['vid_a'] for p in scored] + [p['vid_b'] for p in scored]
for p in scored:
    ra, rb = find(p['vid_a']), find(p['vid_b'])
    if ra != rb:
        parent[ra] = rb
comp_of = {}
for v in set(vids):
    comp_of[v] = find(v)
pair_comp = np.array([comp_of[p['vid_a']] for p in scored])  # vid_a's comp == vid_b's comp by construction

# sanity: no pair crosses components
assert all(comp_of[p['vid_a']] == comp_of[p['vid_b']] for p in scored)

# ---- assign components to 5 folds, greedy balance by pair count ----
# deterministic tie-break by component id (set iteration order varies per process)
comps = sorted(set(pair_comp), key=lambda c: (-int(np.sum(pair_comp == c)), str(c)))
fold_of_comp = {}
fold_pairs = [0] * 5
fold_same = [0] * 5
for c in comps:
    mask = pair_comp == c
    f = int(np.argmin(fold_pairs))
    fold_of_comp[c] = f
    fold_pairs[f] += mask.sum()
    fold_same[f] += y[mask].sum()
fold_of_pair = np.array([fold_of_comp[c] for c in pair_comp])
print("fold pair counts:", fold_pairs, "| same-label:", fold_same)

def apply_thresholds(proba, mt, rt):
    p = np.full(len(proba), -1)
    p[proba >= mt] = 1
    p[proba <= rt] = 0
    return p

def replay_stats(preds, idx):
    """strict pipeline stats on index subset idx."""
    l1d = l1c = l2u = l2c = unres = 0
    for i in idx:
        if preds[i] != -1:
            l1d += 1
            if preds[i] == y[i]: l1c += 1
        else:
            l2u += 1
            if l2lab[i] == -1: unres += 1
            elif l2lab[i] == y[i]: l2c += 1
    n = len(idx)
    return {'n': n, 'l1_decided': l1d, 'l1_correct': l1c,
            'l1_acc_pct': round(l1c / l1d * 100, 2) if l1d else 0,
            'intercept_pct': round(l1d / n * 100, 2),
            'l2_used': l2u, 'l2_correct': l2c, 'unresolved': unres,
            'strict_pipeline_acc_pct': round((l1c + l2c) / n * 100, 2),
            'llm_cost_pct': round(l2u / n * 100, 2)}

def select_thresholds_train(proba_tr, idx_tr):
    """pipeline-aware threshold selection on train (mirrors select_thresholds_empirical).
    proba_tr: probabilities for the train pairs (local order == idx_tr order)."""
    grid = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    y_tr = y[idx_tr]
    l2_tr = l2lab[idx_tr]
    best, best_key = None, None
    for mt in grid:
        for rt in grid:
            if rt >= mt:
                continue
            preds = apply_thresholds(proba_tr, mt, rt)
            l1d = preds != -1
            if l1d.sum() < 5:
                continue
            l1c = int((preds[l1d] == y_tr[l1d]).sum())
            acc = l1c / l1d.sum() * 100
            if acc < 75:
                continue
            esc = ~l1d
            l2c = int(sum(1 for j in np.where(esc)[0] if l2_tr[j] == y_tr[j]))
            strict = (l1c + l2c) / len(idx_tr) * 100
            intercept = l1d.sum() / len(idx_tr) * 100
            key = (round(strict, 6), round(intercept, 6), round(acc, 6),
                   -abs(mt - 0.8) - abs(rt - 0.3))
            if best_key is None or key > best_key:
                best_key = key
                best = (mt, rt)
    return best if best else (0.8, 0.3)

# ---- 5-fold entity-split CV ----
oof = np.full(N, np.nan)
th_per_fold = []
for f in range(5):
    te = np.where(fold_of_pair == f)[0]
    tr = np.where(fold_of_pair != f)[0]
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_scaled[tr], y[tr])
    proba_tr = lr.predict_proba(X_scaled[tr])[:, 1]
    mt, rt = select_thresholds_train(proba_tr, tr)
    oof[te] = lr.predict_proba(X_scaled[te])[:, 1]
    th_per_fold.append({'fold': f, 'merge_t': mt, 'reject_t': rt})
    print(f"fold {f}: train={len(tr)} test={len(te)} thresholds=({mt},{rt})")

preds = np.full(N, -1)
for i in range(N):
    th = th_per_fold[fold_of_pair[i]]
    if oof[i] >= th['merge_t']: preds[i] = 1
    elif oof[i] <= th['reject_t']: preds[i] = 0

# confusion matrix + P/R/F1 (strict: unresolved forced wrong)
final = preds.copy()
esc_idx = np.where(preds == -1)[0]
final[esc_idx] = l2lab[esc_idx]
unres_mask = final == -1
final[unres_mask] = 1 - y[unres_mask]
tp = int(((final == 1) & (y == 1)).sum()); fp = int(((final == 1) & (y == 0)).sum())
tn = int(((final == 0) & (y == 0)).sum()); fn = int(((final == 0) & (y == 1)).sum())
prec = tp / (tp + fp) * 100; rec = tp / (tp + fn) * 100
f1 = 2 * prec * rec / (prec + rec)

overall = replay_stats(preds, np.arange(N))
per_fold = {f: replay_stats(preds, np.where(fold_of_pair == f)[0]) for f in range(5)}
# Matched-budget random-split reference: best pipeline-aware unified operating
# point (fine grid over the random-split full-OOF frontier) whose LLM cost does
# not exceed this entity-disjoint run's cost. Computed, not hand-entered.
# Source: results/ours_fine_grid_gt2.json (random-split, GT v2 labels).
_fg_rows = json.load(open(os.path.join(RES, 'ours_fine_grid_gt2.json')))
_es_cost = overall['llm_cost_pct']
_matched = max((r for r in _fg_rows if r['cost'] <= _es_cost),
               key=lambda r: (r['s'], -r['cost']))
_matched_ref = {
    'accuracy_pct': _matched['s'],
    'llm_cost_pct': _matched['cost'],
    'merge_threshold': _matched['mt'],
    'reject_threshold': _matched['rt'],
    'source': 'ours_fine_grid_gt2.json (best row with cost <= entity-split cost %.2f%%)' % _es_cost,
}
out = {
    'protocol': 'entity-disjoint 5-fold CV via connected components (645 comps, largest 35 entities)',
    'fold_pair_counts': fold_pairs,
    'thresholds_per_fold': th_per_fold,
    'overall': overall,
    'per_fold': per_fold,
    'prf1_same_class': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                        'precision_pct': round(prec, 2), 'recall_pct': round(rec, 2), 'f1_pct': round(f1, 2)},
    'reference_random_split_matched_budget': _matched_ref,
}
def _san(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    raise TypeError(type(o))
json.dump(out, open(os.path.join(RES, 'entity_split_cv.json'), 'w'), indent=1, ensure_ascii=False, default=_san)
print("\nOVERALL:", json.dumps(overall))
print("P/R/F1:", json.dumps(out['prf1_same_class']))
print("random-split matched-budget reference:", json.dumps(_matched_ref))
