#!/usr/bin/env python3
"""Compute empirical strict pipeline accuracy for L1-only utility thresholds using GLM-5-Turbo."""
import json, os, sys, math
import numpy as np
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

with open(os.path.join(RESULTS_DIR, 'scored_pairs.json')) as f:
    scored = json.load(f)
with open(os.path.join(RESULTS_DIR, 'checkpoints_v3', 'glm-5-turbo_checkpoint.json')) as f:
    turbo = json.load(f)

labeled = scored
all_feature_names = set()
for p in labeled:
    all_feature_names.update(p['features_v2'].keys())
feature_names = sorted(all_feature_names)

X = np.array([[p['features_v2'].get(f, 0) for f in feature_names] for p in labeled])
y = np.array([p['label'] for p in labeled])
types = np.array([p['type_a'].replace('activity', 'event') for p in labeled])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fixed_pair_ids = np.array([p['pair_id'] for p in labeled])


def apply_probability_thresholds(proba, merge_t, reject_t):
    preds = np.full(len(proba), -1)
    preds[proba >= merge_t] = 1
    preds[proba <= reject_t] = 0
    return preds


def select_thresholds(proba, y_true, min_acc=75):
    best_score = 0
    best = (0.7, 0.3)
    for mt in np.arange(0.55, 0.95, 0.05):
        for rt in np.arange(0.05, 0.45, 0.05):
            decisions = np.full(len(y_true), -1)
            decisions[proba >= mt] = 1
            decisions[proba <= rt] = 0
            decided = decisions != -1
            if decided.sum() < 5:
                continue
            correct = (decisions[decided] == y_true[decided]).sum()
            acc = correct / decided.sum() * 100
            interc = decided.sum() / len(y_true) * 100
            if acc < min_acc:
                continue
            if acc * interc > best_score:
                best_score = acc * interc
                best = (mt, rt)
    return best


def empirical_full(preds, ckpt, gold_labels, pair_ids):
    total = len(gold_labels)
    l1_decided = l1_correct = 0
    l2_used = l2_correct = 0
    unresolved = 0
    for pred, pid, gold in zip(preds, pair_ids, gold_labels):
        if pred != -1:
            l1_decided += 1
            if pred == gold:
                l1_correct += 1
        else:
            l2_used += 1
            v = ckpt.get(pid, {})
            verdict = v.get('verdict', '') if isinstance(v, dict) else str(v)
            if verdict == 'unknown':
                unresolved += 1
            elif (verdict == 'same') == (gold == 1):
                l2_correct += 1
    resolved = l1_decided + l2_used - unresolved
    resolved_correct = l1_correct + l2_correct
    return {
        'l1_decided': l1_decided, 'l1_correct': l1_correct,
        'l1_acc_pct': round(l1_correct / l1_decided * 100, 1) if l1_decided else 0,
        'intercept_pct': round(l1_decided / total * 100, 1),
        'l2_used': l2_used, 'l2_correct': l2_correct,
        'unresolved': unresolved,
        'strict_pipeline_acc_pct': round(resolved_correct / total * 100, 1),
    }


# Unified legacy
unified_preds = np.full(len(y), -1)
for train_idx, test_idx in skf.split(X_scaled, y):
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_scaled[train_idx], y[train_idx])
    proba_train = lr.predict_proba(X_scaled[train_idx])[:, 1]
    merge_t, reject_t = select_thresholds(proba_train, y[train_idx])
    proba = lr.predict_proba(X_scaled[test_idx])[:, 1]
    unified_preds[test_idx] = apply_probability_thresholds(proba, merge_t, reject_t)

u_stats = empirical_full(unified_preds, turbo, y, fixed_pair_ids)

# Hybrid legacy
TYPE_ORDER = sorted(set(types))
hybrid_preds = np.full(len(y), -1)
for train_idx, test_idx in skf.split(X_scaled, y):
    lr_u = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_u.fit(X_scaled[train_idx], y[train_idx])
    for t in TYPE_ORDER:
        train_mask = types[train_idx] == t
        test_mask = types[test_idx] == t
        n_train_t = train_mask.sum()
        X_test_t = X_scaled[test_idx][test_mask]
        global_test = test_idx[test_mask]
        if len(global_test) == 0:
            continue
        if n_train_t >= 50:
            X_tr = X_scaled[train_idx][train_mask]
            y_tr = y[train_idx][train_mask]
            lr_t = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
            lr_t.fit(X_tr, y_tr)
            proba_tr = lr_t.predict_proba(X_tr)[:, 1]
            mt, rt = select_thresholds(proba_tr, y_tr)
            proba = lr_t.predict_proba(X_test_t)[:, 1]
        else:
            proba_train_u = lr_u.predict_proba(X_scaled[train_idx])[:, 1]
            mt, rt = select_thresholds(proba_train_u, y[train_idx])
            proba = lr_u.predict_proba(X_test_t)[:, 1]
        hybrid_preds[global_test] = apply_probability_thresholds(proba, mt, rt)

h_stats = empirical_full(hybrid_preds, turbo, y, fixed_pair_ids)

result = {
    'l1_only_utility': {
        'unified': u_stats,
        'hybrid': h_stats,
    },
    'l2_model': 'GLM-5-Turbo',
}

print(json.dumps(result, indent=2))

with open(os.path.join(RESULTS_DIR, 'l1_only_empirical_turbo.json'), 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to l1_only_empirical_turbo.json")
