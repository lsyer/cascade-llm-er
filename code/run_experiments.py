#!/usr/bin/env python3
"""
rerun_all_v2.py — Full experiment rerun with l1_scorer (universal property overlap).

Reads the 2639-pair candidate set and the current labels from minec_ground_truth_v2.json, re-scores all pairs with v2 scorer,
then regenerates all 8 affected experiment outputs.

Outputs to results/ directory.
"""
import json, os, sys, math
import numpy as np
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from l1_scorer import l1_score_with_name, extract_features, l1_score

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 1. Load data + re-score with v2
# ============================================================

print("=" * 60)
print("Step 1: Load dataset and re-score with l1_scorer")
print("=" * 60)

with open(os.path.join(DATA_DIR, 'dataset_v3_cleaned.json')) as f:
    data = json.load(f)

pairs = data['same_type_pairs']

with open(os.path.join(DATA_DIR, 'minec_ground_truth_v2.json')) as f:
    gt = json.load(f)
gt_map = {r['pair_id']: r for r in gt['records']}

with open(os.path.join(RESULTS_DIR, 'checkpoints_v3', 'glm-5_checkpoint.json')) as f:
    glm5_ckpt = json.load(f)

# Merge labels from unified ground truth
for p in pairs:
    key = p['pair_id']
    rec = gt_map.get(key)
    if rec and rec.get('final_label') in ('same', 'different'):
        p['label'] = 1 if rec['final_label'] == 'same' else 0
    else:
        p['label'] = None

labeled = [p for p in pairs if p['label'] is not None]
print(f"Total pairs: {len(pairs)}, Labeled: {len(labeled)}")

# Re-score all pairs with v2
for p in labeled:
    props_a = p.get('props_a', {})
    props_b = p.get('props_b', {})
    if isinstance(props_a, str):
        props_a = json.loads(props_a)
    if isinstance(props_b, str):
        props_b = json.loads(props_b)
    
    etype = p['type_a']
    # Normalize type name
    if etype == 'activity':
        etype = 'event'
    
    score, decision, detail = l1_score_with_name(
        p['name_a'], p['name_b'], props_a, props_b, etype
    )
    p['l1_score_v2'] = score
    p['l1_decision_v2'] = decision
    p['l1_detail_v2'] = detail
    p['features_v2'] = extract_features(props_a, props_b, etype, p['name_a'], p['name_b'])

# Save scored dataset
with open(os.path.join(RESULTS_DIR, 'scored_pairs.json'), 'w') as f:
    json.dump([{k: v for k, v in p.items() if k in (
        'pair_id', 'vid_a', 'vid_b', 'name_a', 'name_b', 'type_a', 'type_b',
        'label', 'l1_score_v2', 'l1_decision_v2', 'features_v2'
    )} for p in labeled], f, indent=2, ensure_ascii=False)

# ============================================================
# 2. Table III: L1 decision distribution
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Table III — L1 decision distribution")
print("=" * 60)

decisions = Counter(p['l1_decision_v2'] for p in labeled)
total = len(labeled)
dist = {
    'merge':     {'count': decisions.get('merge', 0), 'pct': decisions.get('merge', 0) / total * 100},
    'escalate':  {'count': decisions.get('escalate', 0), 'pct': decisions.get('escalate', 0) / total * 100},
    'reject':    {'count': decisions.get('reject', 0), 'pct': decisions.get('reject', 0) / total * 100},
    'interception_rate': (decisions.get('merge', 0) + decisions.get('reject', 0)) / total * 100,
    'total': total,
}
print(f"Merge: {dist['merge']['count']} ({dist['merge']['pct']:.1f}%)")
print(f"Escalate: {dist['escalate']['count']} ({dist['escalate']['pct']:.1f}%)")
print(f"Reject: {dist['reject']['count']} ({dist['reject']['pct']:.1f}%)")
print(f"Interception: {dist['interception_rate']:.1f}%")
with open(os.path.join(RESULTS_DIR, 'l1_distribution.json'), 'w') as f:
    json.dump(dist, f, indent=2)

# ============================================================
# 3. Table IV: L1 accuracy (fixed weights)
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Table IV — L1 accuracy (fixed weights, Level 0)")
print("=" * 60)

# L1 accuracy: for intercepted pairs (merge/reject), how many are correct?
intercepted = [p for p in labeled if p['l1_decision_v2'] in ('merge', 'reject')]
correct = sum(1 for p in intercepted if (
    (p['l1_decision_v2'] == 'merge' and p['label'] == 1) or
    (p['l1_decision_v2'] == 'reject' and p['label'] == 0)
))
l1_acc = correct / len(intercepted) * 100 if intercepted else 0
merge_correct = sum(1 for p in intercepted if p['l1_decision_v2'] == 'merge' and p['label'] == 1)
merge_total = sum(1 for p in intercepted if p['l1_decision_v2'] == 'merge')
reject_correct = sum(1 for p in intercepted if p['l1_decision_v2'] == 'reject' and p['label'] == 0)
reject_total = sum(1 for p in intercepted if p['l1_decision_v2'] == 'reject')

l1_acc_result = {
    'l1_accuracy': round(l1_acc, 1),
    'intercepted': len(intercepted),
    'interception_rate': round(len(intercepted) / total * 100, 1),
    'merge_precision': round(merge_correct / merge_total * 100, 1) if merge_total else 0,
    'reject_precision': round(reject_correct / reject_total * 100, 1) if reject_total else 0,
    'merge_total': merge_total,
    'reject_total': reject_total,
    'merge_correct': merge_correct,
    'reject_correct': reject_correct,
    'false_merge': merge_total - merge_correct,
    'false_reject': reject_total - reject_correct,
}
print(f"L1 accuracy: {l1_acc:.1f}% ({correct}/{len(intercepted)})")
print(f"Merge precision: {l1_acc_result['merge_precision']:.1f}% ({merge_correct}/{merge_total})")
print(f"Reject precision: {l1_acc_result['reject_precision']:.1f}% ({reject_correct}/{reject_total})")
with open(os.path.join(RESULTS_DIR, 'l1_accuracy.json'), 'w') as f:
    json.dump(l1_acc_result, f, indent=2)

# ============================================================
# 4. Table V: Expert label distribution per L1 bucket
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Table V — Expert labels per L1 bucket")
print("=" * 60)

buckets = {'merge': [], 'escalate': [], 'reject': []}
for p in labeled:
    buckets[p['l1_decision_v2']].append(p)

bucket_dist = {}
for bucket, plist in buckets.items():
    n = len(plist)
    same = sum(1 for p in plist if p['label'] == 1)
    diff = n - same
    bucket_dist[bucket] = {
        'total': n,
        'same': same,
        'different': diff,
        'same_pct': round(same / n * 100, 1) if n else 0,
    }
    print(f"{bucket}: {n} pairs, {same} same ({bucket_dist[bucket]['same_pct']:.1f}%)")
with open(os.path.join(RESULTS_DIR, 'expert_bucket.json'), 'w') as f:
    json.dump(bucket_dist, f, indent=2)

# ============================================================
# 5. Prepare features for LR experiments
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Prepare feature matrix")
print("=" * 60)

# Get all feature names
all_feature_names = set()
for p in labeled:
    all_feature_names.update(p['features_v2'].keys())
feature_names = sorted(all_feature_names)

X = np.array([[p['features_v2'].get(f, 0) for f in feature_names] for p in labeled])
y = np.array([p['label'] for p in labeled])
types = np.array([p['type_a'].replace('activity', 'event') for p in labeled])

print(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
print(f"Label distribution: {Counter(y)}")
print(f"Type distribution: {Counter(types)}")


def select_thresholds(proba, y_true, min_acc=75):
    """Tune merge/reject thresholds on training data to maximize acc × interception."""
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


def apply_probability_thresholds(proba, merge_t, reject_t):
    preds = np.full(len(proba), -1)
    preds[proba >= merge_t] = 1
    preds[proba <= reject_t] = 0
    return preds


def verdict_to_label(verdict):
    if verdict == 'same':
        return 1
    if verdict == 'different':
        return 0
    return -1


def empirical_pipeline_stats(preds, pair_ids, gold_labels):
    total = len(gold_labels)
    l1_decided = 0
    l1_correct = 0
    l2_used = 0
    l2_correct = 0
    unresolved = 0

    for pred, pid, gold in zip(preds, pair_ids, gold_labels):
        if pred != -1:
            l1_decided += 1
            if pred == gold:
                l1_correct += 1
        else:
            l2_used += 1
            verdict = glm5_ckpt.get(pid, {}).get('verdict', '')
            lbl = verdict_to_label(verdict)
            if lbl == -1:
                unresolved += 1
            elif lbl == gold:
                l2_correct += 1

    resolved = l1_decided + l2_used - unresolved
    resolved_correct = l1_correct + l2_correct
    return {
        'total': int(total),
        'resolved': int(resolved),
        'unresolved': int(unresolved),
        'l1_decided': int(l1_decided),
        'l1_correct': int(l1_correct),
        'l2_used': int(l2_used),
        'l2_correct': int(l2_correct),
        'l1_acc_pct': (l1_correct / l1_decided * 100) if l1_decided else 0.0,
        'strict_pipeline_acc_pct': (resolved_correct / total * 100) if total else 0.0,
        'resolved_pipeline_acc_pct': (resolved_correct / resolved * 100) if resolved else 0.0,
        'intercept_pct': (l1_decided / total * 100) if total else 0.0,
    }


def select_thresholds_empirical(proba, y_true, pair_ids, target_mode='maximize_pipeline',
                                min_l1_acc=None, target_pipeline=None,
                                fixed_merge_t=0.6, fixed_reject_t=-0.4):
    """Tune probability thresholds against empirical pipeline behavior."""
    fixed_merge_p = (fixed_merge_t + 1.0) / 2.0
    fixed_reject_p = (fixed_reject_t + 1.0) / 2.0
    best = None
    best_key = None
    grid = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]

    candidates = []
    for mt in grid:
        for rt in grid:
            if rt >= mt:
                continue
            preds = apply_probability_thresholds(proba, mt, rt)
            stats = empirical_pipeline_stats(preds, pair_ids, y_true)
            if stats['l1_decided'] < 5:
                continue
            if min_l1_acc is not None and stats['l1_acc_pct'] < min_l1_acc:
                continue
            candidates.append({'merge_t': mt, 'reject_t': rt, 'stats': stats})

    if not candidates:
        return fixed_merge_p, fixed_reject_p, {
            'selection_mode': target_mode,
            'fallback': True,
            'reason': 'no candidate passed filters',
        }

    if target_mode == 'match_fixed_pipeline_max_intercept':
        if target_pipeline is None:
            raise ValueError('target_pipeline is required for match_fixed_pipeline_max_intercept')
        feasible = [c for c in candidates if c['stats']['strict_pipeline_acc_pct'] >= target_pipeline]
        pool = feasible if feasible else candidates
        for c in pool:
            s = c['stats']
            if feasible:
                key = (
                    round(s['intercept_pct'], 6),
                    round(s['strict_pipeline_acc_pct'], 6),
                    round(s['l1_acc_pct'], 6),
                    -abs(c['merge_t'] - fixed_merge_p) - abs(c['reject_t'] - fixed_reject_p),
                )
            else:
                key = (
                    -abs(s['strict_pipeline_acc_pct'] - target_pipeline),
                    round(s['intercept_pct'], 6),
                    round(s['l1_acc_pct'], 6),
                )
            if best_key is None or key > best_key:
                best_key = key
                best = c
    else:
        for c in candidates:
            s = c['stats']
            key = (
                round(s['strict_pipeline_acc_pct'], 6),
                round(s['intercept_pct'], 6),
                round(s['l1_acc_pct'], 6),
                -abs(c['merge_t'] - fixed_merge_p) - abs(c['reject_t'] - fixed_reject_p),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = c

    meta = {
        'selection_mode': target_mode,
        'fallback': False,
        'fixed_equivalent_probability_thresholds': {
            'merge_t': round(fixed_merge_p, 3),
            'reject_t': round(fixed_reject_p, 3),
        },
        'candidate_count': len(candidates),
    }
    if target_pipeline is not None:
        meta['target_pipeline'] = round(target_pipeline, 4)
    if min_l1_acc is not None:
        meta['min_l1_acc'] = min_l1_acc
    return best['merge_t'], best['reject_t'], {**meta, **best['stats']}

# ============================================================
# 6. Table IX: Classifier ablation (LR vs RF, 5-fold CV)
# ============================================================
print("\n" + "=" * 60)
print("Step 6: Table IX — Classifier ablation (5-fold CV)")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# LR
lr_preds = np.full(len(y), -1)
for train_idx, test_idx in skf.split(X_scaled, y):
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_scaled[train_idx], y[train_idx])
    lr_preds[test_idx] = lr.predict(X_scaled[test_idx])

lr_mask = lr_preds != -1
lr_correct = (lr_preds[lr_mask] == y[lr_mask]).sum()
lr_acc = lr_correct / lr_mask.sum() * 100

# RF
rf_preds = np.full(len(y), -1)
for train_idx, test_idx in skf.split(X_scaled, y):
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_scaled[train_idx], y[train_idx])
    rf_preds[test_idx] = rf.predict(X_scaled[test_idx])

rf_mask = rf_preds != -1
rf_correct = (rf_preds[rf_mask] == y[rf_mask]).sum()
rf_acc = rf_correct / rf_mask.sum() * 100

classifier_result = {
    'lr_accuracy': round(lr_acc, 1),
    'rf_accuracy': round(rf_acc, 1),
    'n_features': len(feature_names),
    'feature_names': feature_names,
}
print(f"LR: {lr_acc:.1f}%, RF: {rf_acc:.1f}%")
with open(os.path.join(RESULTS_DIR, 'classifier_ablation.json'), 'w') as f:
    json.dump(classifier_result, f, indent=2)

# ============================================================
# 7. Feedback loop: convergence over rounds
# ============================================================
print("\n" + "=" * 60)
print("Step 7: Feedback loop convergence (Table X, Figure 3)")
print("=" * 60)

# Simulate feedback: 5-fold CV, each round adds 20% as feedback
n = len(y)
rounds_data = []

# Round 0 = fixed weights (already computed above)
rounds_data.append({
    'round': 0, 'feedback': 0, 'l1_acc': round(l1_acc, 1),
    'interception': round(len(intercepted) / total * 100, 1),
    'l2_esc': round(100 - len(intercepted) / total * 100, 1),
})

for round_num in range(1, 5):
    train_size = round_num * n // 5
    # Use first train_size as feedback, rest as test
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    if len(test_idx) == 0:
        continue
    
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_scaled[train_idx], y[train_idx])
    preds = lr.predict(X_scaled[test_idx])
    
    # Compute accuracy and interception on test set
    correct = (preds == y[test_idx]).sum()
    acc = correct / len(test_idx) * 100
    interception = 100.0  # LR predicts 0/1 for all, so 100% intercepted
    
    rounds_data.append({
        'round': round_num, 'feedback': train_size, 'l1_acc': round(acc, 1),
        'interception': round(interception, 1), 'l2_esc': 0.0,
    })
    print(f"Round {round_num}: feedback={train_size}, acc={acc:.1f}%, intercept={interception:.1f}%")

with open(os.path.join(RESULTS_DIR, 'feedback_convergence.json'), 'w') as f:
    json.dump(rounds_data, f, indent=2)

# ============================================================
# 8. Hierarchy: Fixed vs Unified vs Per-type Hybrid
# ============================================================
print("\n" + "=" * 60)
print("Step 8: Hierarchy comparison (Table XI)")
print("=" * 60)

# Level 0: Fixed (already have)
level0_acc = l1_acc
level0_intercept = len(intercepted) / total * 100
fixed_pair_ids = np.array([p['pair_id'] for p in labeled])
fixed_preds = np.array([
    1 if p['l1_decision_v2'] == 'merge' else 0 if p['l1_decision_v2'] == 'reject' else -1
    for p in labeled
])
fixed_empirical = empirical_pipeline_stats(fixed_preds, fixed_pair_ids, y)

# Level 1: Unified LR with tuned thresholds (5-fold CV)
unified_preds = np.full(len(y), -1)
unified_thresholds_legacy = []
unified_thresholds_empirical = []
unified_thresholds_match_fixed = []
unified_preds_empirical = np.full(len(y), -1)
unified_preds_match_fixed = np.full(len(y), -1)
unified_preds_aligned = np.full(len(y), -1)
for train_idx, test_idx in skf.split(X_scaled, y):
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_scaled[train_idx], y[train_idx])
    proba_train = lr.predict_proba(X_scaled[train_idx])[:, 1]
    train_pair_ids = np.array([labeled[i]['pair_id'] for i in train_idx])

    # Legacy tuning: acc × interception
    merge_t, reject_t = select_thresholds(proba_train, y[train_idx])
    unified_thresholds_legacy.append({'merge_t': float(merge_t), 'reject_t': float(reject_t)})

    # New tuning A: maximize empirical pipeline
    em_mt, em_rt, em_meta = select_thresholds_empirical(
        proba_train, y[train_idx], train_pair_ids,
        target_mode='maximize_pipeline',
        min_l1_acc=75,
    )
    unified_thresholds_empirical.append({'merge_t': float(em_mt), 'reject_t': float(em_rt), **em_meta})

    # New tuning B: match fixed pipeline, maximize interception
    mf_mt, mf_rt, mf_meta = select_thresholds_empirical(
        proba_train, y[train_idx], train_pair_ids,
        target_mode='match_fixed_pipeline_max_intercept',
        min_l1_acc=75,
        target_pipeline=fixed_empirical['strict_pipeline_acc_pct'],
    )
    unified_thresholds_match_fixed.append({'merge_t': float(mf_mt), 'reject_t': float(mf_rt), **mf_meta})

    # Apply to test
    proba = lr.predict_proba(X_scaled[test_idx])[:, 1]
    unified_preds[test_idx] = apply_probability_thresholds(proba, merge_t, reject_t)
    unified_preds_empirical[test_idx] = apply_probability_thresholds(proba, em_mt, em_rt)
    unified_preds_match_fixed[test_idx] = apply_probability_thresholds(proba, mf_mt, mf_rt)
    unified_preds_aligned[test_idx] = apply_probability_thresholds(proba, 0.8, 0.3)

u_mask = unified_preds != -1
u_correct = sum(1 for i in range(len(y)) if u_mask[i] and unified_preds[i] == y[i])
level1_acc = u_correct / u_mask.sum() * 100 if u_mask.sum() > 0 else 0
level1_intercept = u_mask.sum() / len(y) * 100
u_aligned_stats = empirical_pipeline_stats(unified_preds_aligned, fixed_pair_ids, y)
u_empirical_stats = empirical_pipeline_stats(unified_preds_empirical, fixed_pair_ids, y)
u_match_fixed_stats = empirical_pipeline_stats(unified_preds_match_fixed, fixed_pair_ids, y)

# Level 2: Per-type hybrid with tuned thresholds (min_samples=50)
TYPE_ORDER = sorted(set(types))
hybrid_preds = np.full(len(y), -1)
hybrid_preds_empirical = np.full(len(y), -1)
hybrid_preds_match_fixed = np.full(len(y), -1)
hybrid_preds_aligned = np.full(len(y), -1)
hybrid_thresholds_legacy = []
hybrid_thresholds_empirical = []
hybrid_thresholds_match_fixed = []

for train_idx, test_idx in skf.split(X_scaled, y):
    # Train unified for fallback
    lr_u = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_u.fit(X_scaled[train_idx], y[train_idx])
    proba_train_u = lr_u.predict_proba(X_scaled[train_idx])[:, 1]
    train_pair_ids = np.array([labeled[i]['pair_id'] for i in train_idx])
    u_mt, u_rt = select_thresholds(proba_train_u, y[train_idx])
    u_em_mt, u_em_rt, u_em_meta = select_thresholds_empirical(
        proba_train_u, y[train_idx], train_pair_ids,
        target_mode='maximize_pipeline', min_l1_acc=75,
    )
    u_mf_mt, u_mf_rt, u_mf_meta = select_thresholds_empirical(
        proba_train_u, y[train_idx], train_pair_ids,
        target_mode='match_fixed_pipeline_max_intercept',
        min_l1_acc=75,
        target_pipeline=fixed_empirical['strict_pipeline_acc_pct'],
    )

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
            pair_ids_tr = train_pair_ids[train_mask]
            lr_t = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
            lr_t.fit(X_tr, y_tr)
            proba_tr = lr_t.predict_proba(X_tr)[:, 1]
            mt, rt = select_thresholds(proba_tr, y_tr)
            em_mt, em_rt, em_meta = select_thresholds_empirical(
                proba_tr, y_tr, pair_ids_tr,
                target_mode='maximize_pipeline', min_l1_acc=75,
            )
            mf_mt, mf_rt, mf_meta = select_thresholds_empirical(
                proba_tr, y_tr, pair_ids_tr,
                target_mode='match_fixed_pipeline_max_intercept',
                min_l1_acc=75,
                target_pipeline=fixed_empirical['strict_pipeline_acc_pct'],
            )
            proba = lr_t.predict_proba(X_test_t)[:, 1]
            source = 'per_type'
        else:
            proba = lr_u.predict_proba(X_test_t)[:, 1]
            mt, rt = u_mt, u_rt
            em_mt, em_rt, em_meta = u_em_mt, u_em_rt, u_em_meta
            mf_mt, mf_rt, mf_meta = u_mf_mt, u_mf_rt, u_mf_meta
            source = 'unified_fallback'

        hybrid_thresholds_legacy.append({'type': t, 'source': source, 'merge_t': float(mt), 'reject_t': float(rt)})
        hybrid_thresholds_empirical.append({'type': t, 'source': source, 'merge_t': float(em_mt), 'reject_t': float(em_rt), **em_meta})
        hybrid_thresholds_match_fixed.append({'type': t, 'source': source, 'merge_t': float(mf_mt), 'reject_t': float(mf_rt), **mf_meta})

        hybrid_preds[global_test] = apply_probability_thresholds(proba, mt, rt)
        hybrid_preds_empirical[global_test] = apply_probability_thresholds(proba, em_mt, em_rt)
        hybrid_preds_match_fixed[global_test] = apply_probability_thresholds(proba, mf_mt, mf_rt)
        hybrid_preds_aligned[global_test] = apply_probability_thresholds(proba, 0.8, 0.3)

h_mask = hybrid_preds != -1
h_correct = sum(1 for i in range(len(y)) if h_mask[i] and hybrid_preds[i] == y[i])
level2_acc = h_correct / h_mask.sum() * 100 if h_mask.sum() > 0 else 0
level2_intercept = h_mask.sum() / len(y) * 100
h_aligned_stats = empirical_pipeline_stats(hybrid_preds_aligned, fixed_pair_ids, y)
h_empirical_stats = empirical_pipeline_stats(hybrid_preds_empirical, fixed_pair_ids, y)
h_match_fixed_stats = empirical_pipeline_stats(hybrid_preds_match_fixed, fixed_pair_ids, y)

# Pipeline accuracy (using L2 acc = 97.1% from GLM-5)
L2_ACC = 0.971
p0 = level0_intercept / 100 * level0_acc / 100 + (1 - level0_intercept / 100) * L2_ACC
p1 = level1_intercept / 100 * level1_acc / 100 + (1 - level1_intercept / 100) * L2_ACC
p2 = level2_intercept / 100 * level2_acc / 100 + (1 - level2_intercept / 100) * L2_ACC

hierarchy_result = {
    'fixed': {'l1_acc': round(level0_acc, 1), 'intercept': round(level0_intercept, 1),
              'pipeline': round(p0 * 100, 1), 'l2_calls': round(100 - level0_intercept, 1)},
    'unified': {'l1_acc': round(level1_acc, 1), 'intercept': round(level1_intercept, 1),
                'pipeline': round(p1 * 100, 1), 'l2_calls': round(100 - level1_intercept, 1)},
    'hybrid': {'l1_acc': round(level2_acc, 1), 'intercept': round(level2_intercept, 1),
               'pipeline': round(p2 * 100, 1), 'l2_calls': round(100 - level2_intercept, 1)},
    'feature_names': feature_names,
    'n_features': len(feature_names),
    'per_type_sample_counts': {t: int((types == t).sum()) for t in TYPE_ORDER},
}
print(f"Fixed:  L1={level0_acc:.1f}%, intercept={level0_intercept:.1f}%, pipeline={p0*100:.1f}%")
print(f"Unified: L1={level1_acc:.1f}%, intercept={level1_intercept:.1f}%, pipeline={p1*100:.1f}%")
print(f"Hybrid:  L1={level2_acc:.1f}%, intercept={level2_intercept:.1f}%, pipeline={p2*100:.1f}%")
with open(os.path.join(RESULTS_DIR, 'hierarchy_comparison.json'), 'w') as f:
    json.dump(hierarchy_result, f, indent=2)

threshold_experiments = {
    'fixed_reference': {
        'score_range': [-1.0, 1.0],
        'merge_t': 0.6,
        'reject_t': -0.4,
        'probability_equivalent': {'merge_t': 0.8, 'reject_t': 0.3},
        'legacy_l1_acc_pct': round(level0_acc, 1),
        'legacy_intercept_pct': round(level0_intercept, 1),
        'empirical': {k: round(v, 1) if isinstance(v, float) else v for k, v in fixed_empirical.items()},
    },
    'mode_1_probability_aligned_to_fixed': {
        'description': 'Use Fixed-equivalent probability thresholds (0.3, 0.8) on Unified/Hybrid for normalized-range comparison.',
        'unified': {
            'merge_t': 0.8,
            'reject_t': 0.3,
            **{k: round(v, 1) if isinstance(v, float) else v for k, v in u_aligned_stats.items()}
        },
        'hybrid': {
            'merge_t': 0.8,
            'reject_t': 0.3,
            **{k: round(v, 1) if isinstance(v, float) else v for k, v in h_aligned_stats.items()}
        }
    },
    'mode_2_match_fixed_pipeline_max_intercept': {
        'target_fixed_strict_pipeline_pct': round(fixed_empirical['strict_pipeline_acc_pct'], 1),
        'unified': {k: round(v, 1) if isinstance(v, float) else v for k, v in u_match_fixed_stats.items()},
        'hybrid': {k: round(v, 1) if isinstance(v, float) else v for k, v in h_match_fixed_stats.items()},
        'unified_thresholds_by_fold': unified_thresholds_match_fixed,
        'hybrid_thresholds_by_fold': hybrid_thresholds_match_fixed,
    },
    'mode_3_maximize_empirical_pipeline': {
        'unified': {k: round(v, 1) if isinstance(v, float) else v for k, v in u_empirical_stats.items()},
        'hybrid': {k: round(v, 1) if isinstance(v, float) else v for k, v in h_empirical_stats.items()},
        'unified_thresholds_by_fold': unified_thresholds_empirical,
        'hybrid_thresholds_by_fold': hybrid_thresholds_empirical,
    },
    'legacy_acc_times_interception_thresholds': {
        'unified_thresholds_by_fold': unified_thresholds_legacy,
        'hybrid_thresholds_by_fold': hybrid_thresholds_legacy,
    },
}
with open(os.path.join(RESULTS_DIR, 'threshold_selection_experiments.json'), 'w') as f:
    json.dump(threshold_experiments, f, indent=2)

# ============================================================
# 9. Per-type accuracy + feature importance
# ============================================================
print("\n" + "=" * 60)
print("Step 9: Per-type accuracy + features (Table XII, Figure 4)")
print("=" * 60)

per_type = {}
for t in TYPE_ORDER:
    mask = types == t
    n_t = mask.sum()
    
    # Fixed L1 accuracy for this type
    t_intercepted = [i for i in range(len(y)) if mask[i] and labeled[i]['l1_decision_v2'] in ('merge', 'reject')]
    t_correct = sum(1 for i in t_intercepted if (
        (labeled[i]['l1_decision_v2'] == 'merge' and y[i] == 1) or
        (labeled[i]['l1_decision_v2'] == 'reject' and y[i] == 0)
    ))
    fixed_acc = t_correct / len(t_intercepted) * 100 if t_intercepted else 0
    fixed_intercept = len(t_intercepted) / n_t * 100 if n_t else 0
    
    # Unified LR for this type (from hybrid preds)
    u_mask_t = mask & (unified_preds != -1)
    u_correct_t = sum(1 for i in range(len(y)) if u_mask_t[i] and unified_preds[i] == y[i])
    unified_acc = u_correct_t / u_mask_t.sum() * 100 if u_mask_t.sum() > 0 else 0
    
    # Hybrid LR for this type
    h_mask_t = mask & (hybrid_preds != -1)
    h_correct_t = sum(1 for i in range(len(y)) if h_mask_t[i] and hybrid_preds[i] == y[i])
    hybrid_acc = h_correct_t / h_mask_t.sum() * 100 if h_mask_t.sum() > 0 else 0
    
    per_type[t] = {
        'n_pairs': int(n_t),
        'fixed_acc': round(fixed_acc, 1),
        'fixed_intercept': round(fixed_intercept, 1),
        'unified_acc': round(unified_acc, 1),
        'hybrid_acc': round(hybrid_acc, 1),
        'used_per_type': bool(n_t >= 50),
    }
    print(f"{t:15s}: n={n_t:4d}, fixed={fixed_acc:.1f}%, unified={unified_acc:.1f}%, hybrid={hybrid_acc:.1f}%")

with open(os.path.join(RESULTS_DIR, 'per_type_accuracy.json'), 'w') as f:
    json.dump(per_type, f, indent=2)

# Top features from unified LR (full train)
lr_full = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_full.fit(X_scaled, y)
top_features = sorted(zip(feature_names, lr_full.coef_[0]), key=lambda x: abs(x[1]), reverse=True)[:10]
print("\nTop-10 features (unified LR):")
for fname, weight in top_features:
    print(f"  {fname:35s}: {weight:+.3f}")
feature_importance = {'top_features': [(f, round(w, 3)) for f, w in top_features]}
with open(os.path.join(RESULTS_DIR, 'feature_importance.json'), 'w') as f:
    json.dump(feature_importance, f, indent=2)

# ============================================================
# 10. Pipeline accuracy (Table VIII)
# ============================================================
print("\n" + "=" * 60)
print("Step 10: Pipeline accuracy")
print("=" * 60)

pipeline = {
    'l1_alone': round(l1_acc, 1),
    'pipeline_fixed': round(p0 * 100, 1),
    'pipeline_unified': round(p1 * 100, 1),
    'pipeline_hybrid': round(p2 * 100, 1),
    'l2_acc': 97.1,
}
print(f"L1 alone: {l1_acc:.1f}%")
print(f"Pipeline (fixed): {p0*100:.1f}%")
print(f"Pipeline (hybrid): {p2*100:.1f}%")
with open(os.path.join(RESULTS_DIR, 'pipeline_accuracy.json'), 'w') as f:
    json.dump(pipeline, f, indent=2)

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("ALL DONE — Results saved to results/")
print("=" * 60)
print(f"Output directory: {RESULTS_DIR}")
for f in sorted(os.listdir(RESULTS_DIR)):
    print(f"  {f}")
