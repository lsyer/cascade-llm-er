#!/usr/bin/env python3
"""replay_ditto_gt2.py — Nested per-fold threshold selection on the archived
Ditto OOF probabilities (results/ditto_oof.npz, trained pre-GT-v2; 8/2591 label
revisions make retraining immaterial), replayed under GT v2 final_label with
GLM-5-Turbo and Qwen3.6-35B L2 verdicts (strict convention).
Protocol mirrors run_experiments.py mode_3: per fold, thresholds selected on
training indices using OOF probs (grid .05-.95, min L1 acc 75, maximize strict
pipeline), applied to the held-out fold.
"""
import json, os
import numpy as np
from sklearn.model_selection import StratifiedKFold

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(BASE, 'results')
d = np.load(os.path.join(RES, 'ditto_oof.npz'))
oof, y_tr = d['proba'], d['y']          # y_tr = labels at training time (fold split source)
scored = json.load(open(os.path.join(RES, 'scored_pairs.json')))
pids = [p['pair_id'] for p in scored]
assert len(pids) == len(oof)
gt = json.load(open(os.path.join(BASE, 'data', 'minec_ground_truth_v2.json')))
recs = gt['records'] if isinstance(gt, dict) else gt
gold = np.array([1 if {r['pair_id']: r for r in recs}[q]['final_label'] == 'same' else 0 for q in pids])
def lab(v):
    v = str(v).strip().lower()
    return 1 if v == 'same' else (0 if v == 'different' else -1)
ck = json.load(open(os.path.join(RES, 'checkpoints_v3', 'glm-5-turbo_checkpoint.json')))
glm = np.array([lab(ck.get(q, {}).get('verdict', '')) for q in pids])
qv = {r['pair_id']: lab(r['verdict']) for r in json.load(open(os.path.join(RES, 'qwen_l2_full.json')))['records']}
qw = np.array([qv.get(q, -1) for q in pids])

skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(np.zeros(len(y_tr)), y_tr))
preds = np.full(len(y_tr), -1)
ths = []
for tr, te in folds:
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
            l1ok = int((dec[itc] == gold[tr][itc]).sum())
            if l1ok / itc.sum() * 100 < 75:
                continue
            esc = ~itc
            l2ok = int((glm[tr][esc] == gold[tr][esc]).sum())
            key = ((l1ok + l2ok) / len(tr), itc.mean())
            if best is None or key > best[0]:
                best = (key, round(float(mt), 2), round(float(rt), 2))
    mt, rt = best[1], best[2]
    ths.append((mt, rt))
    pr = oof[te]
    dd = np.full(len(te), -1)
    dd[pr >= mt] = 1
    dd[pr <= rt] = 0
    preds[te] = dd
esc = preds == -1
itc = ~esc
standalone = float(((oof >= 0.5).astype(int) == gold).mean() * 100)
out = {'fold_thresholds': ths, 'standalone_oof_acc_pct_gt2': round(standalone, 2)}
for name, L in (('glm', glm), ('qwen', qw)):
    l1ok = int((preds[itc] == gold[itc]).sum())
    l2ok = int((L[esc] == gold[esc]).sum())
    out[name] = {'strict': round((l1ok + l2ok) / len(gold) * 100, 2),
                 'cost_pct': round(esc.mean() * 100, 1),
                 'l1_acc': round(l1ok / itc.sum() * 100, 1),
                 'unresolved': int((L[esc] == -1).sum())}
json.dump(out, open(os.path.join(RES, 'ditto_replay_gt2.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
