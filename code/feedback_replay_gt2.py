#!/usr/bin/env python3
"""feedback_replay_gt2.py — Rebuild the feedback-convergence table under GT v2.

Protocol (matches the published table and run_experiments.py Step 7):
  round 0  = fixed-weight L1 baseline on all pairs (from run_experiments archives);
  round r (1..4) = train unified LR on the first r folds (cumulative 519/1037/
  1555/2073 pseudo-feedback pairs), apply the DEPLOYED L1-only-utility
  thresholds (0.55/0.40), evaluate on the held-out complement; pipeline =
  strict empirical accuracy (L1-correct on intercepted + GLM-correct on
  escalated) / eval size.
"""
import json, os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(BASE, 'results')
scored = json.load(open(os.path.join(RES, 'scored_pairs.json')))
pids = [p['pair_id'] for p in scored]
y = np.array([p['label'] for p in scored])          # GT v2
feats = [p['features_v2'] for p in scored]
fnames = sorted({k for f in feats for k in f})
X = StandardScaler().fit_transform(
    np.array([[float(f.get(k, 0) or 0) for k in fnames] for f in feats]))
ck = json.load(open(os.path.join(RES, 'checkpoints_v3', 'glm-5-turbo_checkpoint.json')))
def lab(v):
    v = str(v).strip().lower()
    return 1 if v == 'same' else (0 if v == 'different' else -1)
glm = np.array([lab(ck.get(q, {}).get('verdict', '')) for q in pids])

skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(X, y))
train_idx = [te for tr, te in folds]      # fold test sets are the feedback batches

# round 0: fixed baseline from archive
ts = json.load(open(os.path.join(RES, 'threshold_selection_experiments.json')))
fx = ts['fixed_reference']['empirical']
rounds = [{'round': 0, 'n_feedback': 0,
           'l1_acc': fx['l1_acc_pct'], 'interception': fx['intercept_pct'],
           'l2_pct': round(100 - fx['intercept_pct'], 1),
           'pipeline': fx['strict_pipeline_acc_pct']}]
for r in range(1, 5):
    tr = np.concatenate(train_idx[:r])
    ev = np.concatenate(train_idx[r:])
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X[tr], y[tr])
    # deployed L1-only-utility thresholds (same as run_experiments.py Step 7)
    mt, rt = 0.55, 0.40
    p_ev = lr.predict_proba(X[ev])[:, 1]
    dec = np.full(len(ev), -1)
    dec[p_ev >= mt] = 1
    dec[p_ev <= rt] = 0
    itc = dec != -1
    esc = ~itc
    l1ok = int((dec[itc] == y[ev][itc]).sum())
    l2ok = int((glm[ev][esc] == y[ev][esc]).sum())
    rounds.append({'round': r, 'n_feedback': int(len(tr)),
                   'l1_acc': round(l1ok / itc.sum() * 100, 1),
                   'interception': round(itc.mean() * 100, 1),
                   'l2_pct': round(esc.mean() * 100, 1),
                   'pipeline': round((l1ok + l2ok) / len(ev) * 100, 1)})
json.dump(rounds, open(os.path.join(RES, 'feedback_convergence_gt2.json'), 'w'), indent=1)
for r in rounds:
    print(r)
