#!/usr/bin/env python3
"""Classifier operating points (Table: classifier-ablation), GT v2 canonical.

Protocol (paper Section threshold-protocols): thresholds selected on FULL
out-of-fold probabilities/scores over a 0.01-step grid to maximize L1
accuracy subject to interception >= 10%, ties broken toward higher
interception. Pipeline accuracy is strict (unresolved L2 verdicts count as
incorrect, denominator = 2591). P/R/F1 treat unresolved verdicts as errors.

Writes results/classifier_operating_points_gt2.json with embedded OOF
probabilities so every table number replays from this artifact alone.
"""
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

BASE = os.path.join(os.path.dirname(__file__), '..')
R = lambda n: json.load(open(os.path.join(BASE, 'results', n)))

sp = R('scored_pairs.json')
gt = {e['pair_id']: (1 if e['final_label'] == 'same' else 0)
      for e in R('../data/minec_ground_truth_v2.json')['records']}
ck = R('checkpoints_v3/glm-5-turbo_checkpoint.json')

pids = [p['pair_id'] for p in sp]
gold = np.array([gt[p] for p in pids])          # GT v2 labels
fs = np.array([p['l1_score_v2'] for p in sp])
L2 = np.array([{'same': 1, 'different': 0}.get(ck.get(p, {}).get('verdict', ''), -1)
               for p in pids])
N = len(pids)
assert N == 2591

feats = R('classifier_ablation.json')['feature_names']
X = np.array([[p['features_v2'].get(f, 0) for f in feats] for p in sp])
Xs = StandardScaler().fit_transform(X)

# ── 5-fold OOF probabilities, GT v2 labels ──
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_p = np.full(N, np.nan); rf_p = np.full(N, np.nan)
for tr, te in skf.split(Xs, gold):
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(Xs[tr], gold[tr]); lr_p[te] = lr.predict_proba(Xs[te])[:, 1]
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(Xs[tr], gold[tr]); rf_p[te] = rf.predict_proba(Xs[te])[:, 1]

def evaluate(scores, mt, rt):
    pred = np.full(N, -1); pred[scores >= mt] = 1; pred[scores <= rt] = 0
    it = pred != -1; esc = ~it
    l1c = int((pred[it] == gold[it]).sum()); l1n = int(it.sum())
    l2ok = int(((L2 == gold) & esc).sum())
    final = np.where(it, pred, np.where(L2 >= 0, L2, -1))
    tp = int(((final == 1) & (gold == 1)).sum())
    fp = int(((final == 1) & (gold == 0)).sum()) + int(((final == -1) & (gold == 0)).sum())
    fn = int(((final != 1) & (gold == 1)).sum())
    P = tp / (tp + fp) * 100; Rc = tp / (tp + fn) * 100
    return {'merge_t': round(mt, 2), 'reject_t': round(rt, 2),
            'l1_accuracy': round(l1c / l1n * 100, 1),
            'interception': round(it.mean() * 100, 1),
            'pipeline_strict': round((l1c + l2ok) / N * 100, 1),
            'llm_cost': round(esc.mean() * 100, 1),
            'precision': round(P, 1), 'recall': round(Rc, 1),
            'f1': round(2 * P * Rc / (P + Rc), 1),
            'n_decided': l1n}

def sweep(scores, lo=0.0, hi=1.0):
    grid = [round(x * 0.01, 2) for x in range(int(lo * 100), int(hi * 100) + 1)]
    best_key, best = None, None
    for mt in grid:
        for rt in grid:
            if rt >= mt:
                continue
            pred = np.full(N, -1); pred[scores >= mt] = 1; pred[scores <= rt] = 0
            it = pred != -1
            if it.sum() < 5 or it.mean() < 0.10:
                continue
            acc = (pred[it] == gold[it]).mean() * 100
            key = (round(acc, 6), round(it.mean(), 6))
            if best_key is None or key > best_key:
                best_key, best = key, (mt, rt)
    return best

out = {'protocol': 'L1-accuracy-max on full OOF, 0.01 grid, interception>=10%, ties->higher interception',
       'labels': 'GT v2 (minec_ground_truth_v2.json final_label)',
       'n_pairs': N}
for name, scores, lo, hi in (('unified_lr', lr_p, 0.0, 1.0),
                             ('random_forest', rf_p, 0.0, 1.0),
                             ('fixed_l1', fs, -1.0, 1.0)):
    mt, rt = sweep(scores, lo, hi)
    out[name] = evaluate(scores, mt, rt)

l2ok = int((L2 == gold).sum()); unres = int((L2 == -1).sum())
tp = int(((L2 == 1) & (gold == 1)).sum())
fp = int(((L2 == 1) & (gold == 0)).sum()) + int(((L2 == -1) & (gold == 0)).sum())
fn = int(((L2 != 1) & (gold == 1)).sum())
P = tp / (tp + fp) * 100; Rc = tp / (tp + fn) * 100
out['pure_l2'] = {'pipeline_strict': round(l2ok / N * 100, 1), 'unresolved': unres,
                  'precision': round(P, 1), 'recall': round(Rc, 1),
                  'f1': round(2 * P * Rc / (P + Rc), 1)}
out['rf_zero_fraction_pct'] = round((rf_p == 0).mean() * 100, 1)
out['oof_probabilities'] = {'lr': [round(v, 6) for v in lr_p],
                            'rf': [round(v, 6) for v in rf_p]}

path = os.path.join(BASE, 'results', 'classifier_operating_points_gt2.json')
json.dump(out, open(path, 'w'), indent=1)
for k in ('unified_lr', 'random_forest', 'fixed_l1', 'pure_l2'):
    print(k, out[k])
print('rf_zero_fraction_pct:', out['rf_zero_fraction_pct'])
print('saved', path)
