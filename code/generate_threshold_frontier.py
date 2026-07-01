#!/usr/bin/env python3
import json, os
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

BASE = Path('/root/workspace/tanshu_docs/experiments/release')
DATA_DIR = BASE / 'data'
RESULTS_DIR = BASE / 'results'
CODE_DIR = BASE / 'code'

import sys
sys.path.insert(0, str(CODE_DIR))
from l1_scorer import l1_score_with_name, extract_features

with open(DATA_DIR / 'dataset_v3_cleaned.json') as f:
    data = json.load(f)
pairs = data['same_type_pairs']
with open(DATA_DIR / 'minec_ground_truth_v2.json') as f:
    gt = json.load(f)
gt_map = {r['pair_id']: r for r in gt['records']}
with open(RESULTS_DIR / 'checkpoints_v3' / 'glm-5_checkpoint.json') as f:
    glm5 = json.load(f)

for p in pairs:
    rec = gt_map.get(p['pair_id'])
    if rec and rec.get('final_label') in ('same', 'different'):
        p['label'] = 1 if rec['final_label'] == 'same' else 0
    else:
        p['label'] = None

labeled = [p for p in pairs if p['label'] is not None]
for p in labeled:
    props_a = p.get('props_a', {})
    props_b = p.get('props_b', {})
    if isinstance(props_a, str): props_a = json.loads(props_a)
    if isinstance(props_b, str): props_b = json.loads(props_b)
    etype = 'event' if p['type_a'] == 'activity' else p['type_a']
    score, decision, detail = l1_score_with_name(p['name_a'], p['name_b'], props_a, props_b, etype)
    p['l1_score_v2'] = score
    p['l1_decision_v2'] = decision
    p['features_v2'] = extract_features(props_a, props_b, etype, p['name_a'], p['name_b'])

feature_names = sorted({k for p in labeled for k in p['features_v2'].keys()})
X = np.array([[p['features_v2'].get(f, 0) for f in feature_names] for p in labeled])
y = np.array([p['label'] for p in labeled])
types = np.array([('event' if p['type_a'] == 'activity' else p['type_a']) for p in labeled])
pair_ids = np.array([p['pair_id'] for p in labeled])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def verdict_to_label(v):
    return 1 if v == 'same' else 0 if v == 'different' else -1

def apply_prob(proba, mt, rt):
    pred = np.full(len(proba), -1)
    pred[proba >= mt] = 1
    pred[proba <= rt] = 0
    return pred

def empirical_stats(preds, ids, gold):
    total = len(gold)
    l1_decided=l1_correct=l2_used=l2_correct=unresolved=0
    for pred, pid, g in zip(preds, ids, gold):
        if pred != -1:
            l1_decided += 1
            if pred == g: l1_correct += 1
        else:
            l2_used += 1
            lv = verdict_to_label(glm5.get(pid, {}).get('verdict', ''))
            if lv == -1: unresolved += 1
            elif lv == g: l2_correct += 1
    resolved = l1_decided + l2_used - unresolved
    correct = l1_correct + l2_correct
    return {
        'total': int(total),
        'resolved': int(resolved),
        'unresolved': int(unresolved),
        'l1_decided': int(l1_decided),
        'l1_correct': int(l1_correct),
        'l2_used': int(l2_used),
        'l2_correct': int(l2_correct),
        'intercept_pct': round(l1_decided / total * 100, 3),
        'l1_acc_pct': round(l1_correct / l1_decided * 100, 3) if l1_decided else 0.0,
        'strict_pipeline_acc_pct': round(correct / total * 100, 3),
        'resolved_pipeline_acc_pct': round(correct / resolved * 100, 3) if resolved else 0.0,
    }

def select_legacy(proba, y_true, min_acc=75):
    best_score=-1
    best=(0.7,0.3)
    for mt in np.arange(0.55,0.95,0.05):
        for rt in np.arange(0.05,0.45,0.05):
            if rt >= mt: continue
            preds = apply_prob(proba, mt, rt)
            mask = preds != -1
            if mask.sum() < 5: continue
            acc = (preds[mask] == y_true[mask]).mean()*100
            interc = mask.mean()*100
            if acc < min_acc: continue
            score = acc * interc
            if score > best_score:
                best_score = score
                best = (float(mt), float(rt))
    return best

def select_empirical(proba, y_true, ids, mode='maximize_pipeline', target=None, min_l1_acc=75, fixed=(0.8,0.3)):
    cands=[]
    for mt in np.arange(0.05,0.96,0.05):
        for rt in np.arange(0.05,0.96,0.05):
            mt=round(float(mt),2); rt=round(float(rt),2)
            if rt >= mt: continue
            preds = apply_prob(proba, mt, rt)
            stats = empirical_stats(preds, ids, y_true)
            if stats['l1_decided'] < 5: continue
            if stats['l1_acc_pct'] < min_l1_acc: continue
            cands.append((mt,rt,stats))
    assert cands
    best=None; best_key=None
    if mode == 'match_fixed_pipeline_max_intercept':
        feasible=[c for c in cands if c[2]['strict_pipeline_acc_pct'] >= target]
        pool=feasible if feasible else cands
        for mt,rt,st in pool:
            if feasible:
                key=(st['intercept_pct'], st['strict_pipeline_acc_pct'], st['l1_acc_pct'], -(abs(mt-fixed[0])+abs(rt-fixed[1])))
            else:
                key=(-abs(st['strict_pipeline_acc_pct']-target), st['intercept_pct'], st['l1_acc_pct'])
            if best_key is None or key > best_key:
                best_key=key; best=(mt,rt,st)
    else:
        for mt,rt,st in cands:
            key=(st['strict_pipeline_acc_pct'], st['intercept_pct'], st['l1_acc_pct'], -(abs(mt-fixed[0])+abs(rt-fixed[1])))
            if best_key is None or key > best_key:
                best_key=key; best=(mt,rt,st)
    return best

# fixed reference
fixed_preds = np.array([1 if p['l1_decision_v2']=='merge' else 0 if p['l1_decision_v2']=='reject' else -1 for p in labeled])
fixed_stats = empirical_stats(fixed_preds, pair_ids, y)

# OOF probabilities for unified and hybrid
unified_proba = np.zeros(len(y))
hybrid_proba = np.zeros(len(y))
for train_idx, test_idx in skf.split(X_scaled, y):
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_scaled[train_idx], y[train_idx])
    unified_proba[test_idx] = lr.predict_proba(X_scaled[test_idx])[:,1]

    lr_u = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_u.fit(X_scaled[train_idx], y[train_idx])
    for t in sorted(set(types)):
        train_mask = types[train_idx] == t
        test_mask = types[test_idx] == t
        if not test_mask.any():
            continue
        X_test_t = X_scaled[test_idx][test_mask]
        if train_mask.sum() >= 50:
            lr_t = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
            lr_t.fit(X_scaled[train_idx][train_mask], y[train_idx][train_mask])
            hybrid_proba[test_idx[test_mask]] = lr_t.predict_proba(X_test_t)[:,1]
        else:
            hybrid_proba[test_idx[test_mask]] = lr_u.predict_proba(X_test_t)[:,1]

# frontier sweeps
frontier = {'fixed_reference': fixed_stats, 'unified': [], 'hybrid': []}
for label, proba in [('unified', unified_proba), ('hybrid', hybrid_proba)]:
    pts=[]
    for mt in np.arange(0.05,0.96,0.05):
        for rt in np.arange(0.05,0.96,0.05):
            mt=round(float(mt),2); rt=round(float(rt),2)
            if rt >= mt: continue
            st = empirical_stats(apply_prob(proba, mt, rt), pair_ids, y)
            pts.append({'merge_t': mt, 'reject_t': rt, **st})
    # simple Pareto on strict_pipeline vs intercept
    pareto=[]
    for p in pts:
        dominated=False
        for q in pts:
            if q is p: continue
            if q['strict_pipeline_acc_pct'] >= p['strict_pipeline_acc_pct'] and q['intercept_pct'] >= p['intercept_pct'] and (q['strict_pipeline_acc_pct'] > p['strict_pipeline_acc_pct'] or q['intercept_pct'] > p['intercept_pct']):
                dominated=True; break
        if not dominated:
            pareto.append(p)
    pareto=sorted(pareto, key=lambda x:(x['intercept_pct'], x['strict_pipeline_acc_pct']))
    frontier[label] = pareto

# same-model different-objective ablation using OOF probs
abl = {'fixed': fixed_stats}
for label, proba in [('unified', unified_proba), ('hybrid', hybrid_proba)]:
    legacy_mt, legacy_rt = select_legacy(proba, y)
    legacy_stats = empirical_stats(apply_prob(proba, legacy_mt, legacy_rt), pair_ids, y)
    norm_stats = empirical_stats(apply_prob(proba, 0.8, 0.3), pair_ids, y)
    emp_mt, emp_rt, emp_stats = select_empirical(proba, y, pair_ids, mode='maximize_pipeline', target=None)
    match_mt, match_rt, match_stats = select_empirical(proba, y, pair_ids, mode='match_fixed_pipeline_max_intercept', target=fixed_stats['strict_pipeline_acc_pct'])
    abl[label] = {
        'legacy_acc_times_interception': {'merge_t': legacy_mt, 'reject_t': legacy_rt, **legacy_stats},
        'normalized_to_fixed': {'merge_t': 0.8, 'reject_t': 0.3, **norm_stats},
        'maximize_empirical_pipeline': {'merge_t': emp_mt, 'reject_t': emp_rt, **emp_stats},
        'match_fixed_pipeline_max_intercept': {'merge_t': match_mt, 'reject_t': match_rt, **match_stats},
    }

out1 = RESULTS_DIR / 'threshold_frontier.json'
out2 = RESULTS_DIR / 'threshold_objective_ablation.json'
out1.write_text(json.dumps(frontier, indent=2))
out2.write_text(json.dumps(abl, indent=2))
print(json.dumps({'frontier': str(out1), 'ablation': str(out2), 'fixed': fixed_stats, 'unified_emp': abl['unified']['maximize_empirical_pipeline'], 'hybrid_emp': abl['hybrid']['maximize_empirical_pipeline']}, indent=2))
