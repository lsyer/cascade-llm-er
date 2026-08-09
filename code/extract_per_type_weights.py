#!/usr/bin/env python3
"""
Extract per-type LR coefficients using the same data pipeline as run_experiments.py.
Uses corrected MINEC ground truth (minec_ground_truth_v2.json) + 42-feature space.
Trains a single LR per entity type on ALL labeled pairs of that type (no CV split —
this is the same protocol as the unified all_lr_coefficients.json, which trains lr_full
on the full dataset).

Outputs:
  results/per_type_lr_coefficients.json
"""
import json, os, sys
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from l1_scorer import l1_score_with_name, extract_features

BASE = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE, 'data')
RESULTS_DIR = os.path.join(BASE, 'results')

# ============================================================
# 1. Load data — same as run_experiments.py
# ============================================================
with open(os.path.join(DATA_DIR, 'dataset_v3_cleaned.json')) as f:
    data = json.load(f)
pairs = data['same_type_pairs']

with open(os.path.join(DATA_DIR, 'minec_ground_truth_v2.json')) as f:
    gt = json.load(f)
gt_map = {r['pair_id']: r for r in gt['records']}

# Merge labels
for p in pairs:
    rec = gt_map.get(p['pair_id'])
    if rec and rec.get('final_label') in ('same', 'different'):
        p['label'] = 1 if rec['final_label'] == 'same' else 0
    else:
        p['label'] = None

labeled = [p for p in pairs if p['label'] is not None]
print(f"Total pairs: {len(pairs)}, Labeled: {len(labeled)}")

# Re-score + extract features
for p in labeled:
    props_a = p.get('props_a', {})
    props_b = p.get('props_b', {})
    if isinstance(props_a, str):
        props_a = json.loads(props_a)
    if isinstance(props_b, str):
        props_b = json.loads(props_b)
    etype = p['type_a']
    if etype == 'activity':
        etype = 'event'
    p['features_v2'] = extract_features(props_a, props_b, etype, p['name_a'], p['name_b'])
    p['_etype'] = etype

# Build feature matrix — use global feature_names (same 42 as unified model)
all_feature_names = set()
for p in labeled:
    all_feature_names.update(p['features_v2'].keys())
feature_names = sorted(all_feature_names)
print(f"Feature space: {len(feature_names)} features")

TYPE_ORDER = ['organization', 'person', 'location', 'equipment', 'event']

# ============================================================
# 2. Train per-type LR on all pairs of that type
# ============================================================
result = {'feature_names': feature_names, 'n_features': len(feature_names), 'per_type': {}}

for t in TYPE_ORDER:
    t_pairs = [p for p in labeled if p['_etype'] == t]
    if len(t_pairs) < 20:
        print(f"  {t}: only {len(t_pairs)} pairs, skipping LR")
        result['per_type'][t] = {'n_pairs': len(t_pairs), 'note': 'insufficient samples for per-type LR'}
        continue

    X_t = np.array([[p['features_v2'].get(f, 0) for f in feature_names] for p in t_pairs])
    y_t = np.array([p['label'] for p in t_pairs])

    # Check class balance
    pos = y_t.sum()
    neg = len(y_t) - pos
    print(f"  {t}: {len(t_pairs)} pairs (pos={pos}, neg={neg})")

    # Standardize
    scaler_t = StandardScaler()
    X_scaled_t = scaler_t.fit_transform(X_t)

    # Train LR (same params as run_experiments.py)
    lr_t = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_t.fit(X_scaled_t, y_t)

    # Extract coefficients
    coefs = list(zip(feature_names, lr_t.coef_[0]))
    coefs_sorted = sorted(coefs, key=lambda x: abs(x[1]), reverse=True)

    result['per_type'][t] = {
        'n_pairs': len(t_pairs),
        'n_positive': int(pos),
        'n_negative': int(neg),
        'top5_features': [(f, round(w, 4)) for f, w in coefs_sorted[:5]],
        'all_coefficients': [(f, round(w, 4)) for f, w in coefs_sorted],
    }

    print(f"    Top-5: {[(f, round(w,2)) for f, w in coefs_sorted[:5]]}")

# ============================================================
# 3. Save
# ============================================================
out_path = os.path.join(RESULTS_DIR, 'per_type_lr_coefficients.json')
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_path}")
