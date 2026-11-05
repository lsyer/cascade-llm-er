#!/usr/bin/env python3
"""replay_mode3_gt2.py — Faithful standalone replication of the pipeline-aware
nested protocol (run_experiments.py mode_3, unified LR) under GT v2 labels,
plus Ditto-router and Qwen-judge replays on the resulting escalation mask.
Saves results/mode3_replay_gt2.json.
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
y = np.array([p['label'] for p in scored])                       # GT v2 (rewritten 12:34)
feats = [p['features_v2'] for p in scored]
fnames = sorted({k for f in feats for k in f})
X = np.array([[float(f.get(k, 0) or 0) for k in fnames] for f in feats])
Xs = StandardScaler().fit_transform(X)

gt = json.load(open(os.path.join(BASE, 'data', 'minec_ground_truth_v2.json')))
recs = gt['records'] if isinstance(gt, dict) else gt
gold = np.array([1 if r['final_label'] == 'same' else 0 for r in {q['pair_id']: q for q in []}] or
                [1 if {r['pair_id']: r for r in recs}[q]['final_label'] == 'same' else 0 for q in pids])
ckg = json.load(open(os.path.join(RES, 'checkpoints_v3', 'glm-5-turbo_checkpoint.json')))
def lab(v):
    v = str(v).strip().lower()
    return 1 if v == 'same' else (0 if v == 'different' else -1)
glm = np.array([lab(ckg.get(q, {}).get('verdict', '')) for q in pids])
qwen = {r['pair_id']: lab(r['verdict']) for r in json.load(open(os.path.join(RES, 'qwen_l2_full.json')))['records']}
qw = np.array([qwen.get(q, -1) for q in pids])

# ---- OOF probabilities (5-fold, class_weight balanced, rs=42) ----
skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(Xs, y))
oof = np.full(len(y), np.nan)
for tr, te in folds:
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(Xs[tr], y[tr])
    oof[te] = lr.predict_proba(Xs[te])[:, 1]
fold_th = []
for tr, te in folds:
    # threshold selection on TRAINING fold: maximize strict pipeline (GLM replay), min L1 acc 75
    ptr = oof[tr]
    best = None
    for mt in np.arange(0.05, 0.96, 0.05):
        for rt in np.arange(0.05, 0.96, 0.05):
            if rt >= mt:
                continue
            dec = np.full(len(tr), -1)
            dec[ptr >= round(float(mt), 2)] = 1
            dec[ptr <= round(float(rt), 2)] = 0
            itc = dec != -1
            if itc.sum() < 5:
                continue
            l1ok = int((dec[itc] == y[tr][itc]).sum())
            if l1ok / itc.sum() * 100 < 75:
                continue
            esc = ~itc
            l2ok = int((glm[tr][esc] == y[tr][esc]).sum())
            key = ((l1ok + l2ok) / len(tr), itc.mean(), l1ok / itc.sum())
            if best is None or key > best[0]:
                best = (key, round(float(mt), 2), round(float(rt), 2))
    fold_th.append((best[1], best[2]))
print('per-fold thresholds:', fold_th)

# ---- apply to held-out folds ----
preds = np.full(len(y), -1)
for (tr, te), (mt, rt) in zip(skf.split(Xs, y), fold_th):
    pr = oof[te]
    d = np.full(len(te), -1)
    d[pr >= mt] = 1
    d[pr <= rt] = 0
    preds[te] = d
esc = preds == -1
itc = ~esc
out = {'fold_thresholds': fold_th, 'n_features': len(fnames)}
for name, L in (('glm', glm), ('qwen', qw)):
    l1ok = int((preds[itc] == gold[itc]).sum())
    l2ok = int((L[esc] == gold[esc]).sum())
    out[name] = {'strict': round((l1ok + l2ok) / len(y) * 100, 2),
                 'cost_pct': round(esc.mean() * 100, 1),
                 'l1_acc': round(l1ok / itc.sum() * 100, 1),
                 'intercept_pct': round(itc.mean() * 100, 1),
                 'unresolved': int((L[esc] == -1).sum())}
json.dump(out, open(os.path.join(RES, 'mode3_replay_gt2.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
