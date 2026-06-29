#!/usr/bin/env python3
"""
DBP15K Hierarchical Weight Learning Experiment.

Mirrors MINEC paper Section 5.5 (Adaptive Weight Learning):
- Level 0: Fixed L1 weights (baseline)
- Level 1: Unified logistic regression trained on feedback
- (Level 2 per-type skipped: DBP15K has no entity types)

Uses 5-fold CV: each round adds one fold as "feedback" training data.
Gold labels serve as ground truth (superior to L2-as-feedback in MINEC).

Also computes:
- Feature importance (LR coefficients)
- Pipeline accuracy estimate (L1+L2)
"""

import json, random, re, os, sys, math
from collections import defaultdict
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "JAPE" / "data" / "dbp15k" / "zh_en"
RESULTS_DIR = Path(__file__).parent

# ─── Data loading (shared with L1 script) ───

def load_labels(filepath):
    labels = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                labels[parts[0]] = {'label': parts[1], 'trans': parts[2]}
            elif len(parts) == 2:
                labels[parts[0]] = {'label': parts[1], 'trans': parts[1]}
    return labels

def load_ILLs(filepath):
    pairs = set()
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                pairs.add((parts[0], parts[1]))
    return pairs

def load_attrs(filepath):
    attrs = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                attrs[parts[0]] = set(p.split('/')[-1].lower() for p in parts[1:])
    return attrs

def load_rel_triples(filepath):
    triples = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                triples[parts[0]].append((parts[1].split('/')[-1], parts[2].split('/')[-1]))
    return triples

def normalize_name(name):
    name = name.lower().strip()
    name = re.sub(r'[_\-\(\)]', ' ', name)
    name = re.sub(r'\s+', ' ', name)
    return name

print("Loading DBP15K ZH-EN...")
zh_labels = load_labels(DATA_DIR / "s_labels")
en_labels = load_labels(DATA_DIR / "t_labels")
gold_set = load_ILLs(DATA_DIR / "ent_ILLs")
zh_attrs = load_attrs(DATA_DIR / "training_attrs_1")
en_attrs = load_attrs(DATA_DIR / "training_attrs_2")
zh_triples = load_rel_triples(DATA_DIR / "s_triples")
en_triples = load_rel_triples(DATA_DIR / "t_triples")

# ─── Build evaluation set (same as L1 script) ───
random.seed(42)
gold_list = list(gold_set)
pos_sample = random.sample(gold_list, min(2000, len(gold_list)))

zh_uris = list(zh_labels.keys())
en_uris = list(en_labels.keys())
en_name_index = defaultdict(list)
for uri in en_uris:
    label = en_labels.get(uri, {}).get('label', '')
    norm = normalize_name(label)
    if norm:
        first = norm.split()[0][:3] if norm.split() else norm[:3]
        en_name_index[first].append(uri)

neg_pairs = []
attempts = 0
while len(neg_pairs) < 2000 and attempts < 50000:
    attempts += 1
    zh_uri = random.choice(zh_uris)
    if zh_uri not in zh_labels:
        continue
    trans = zh_labels[zh_uri].get('trans', '')
    norm = normalize_name(trans)
    if not norm:
        continue
    prefix = norm.split()[0][:3] if norm.split() else norm[:3]
    candidates = en_name_index.get(prefix, [])
    if not candidates:
        continue
    en_uri = random.choice(candidates)
    if (zh_uri, en_uri) not in gold_set:
        neg_pairs.append((zh_uri, en_uri))

all_pairs = [(zh, en, 1) for zh, en in pos_sample] + [(zh, en, 0) for zh, en in neg_pairs]
print(f"Evaluation set: {len(all_pairs)} pairs ({len(pos_sample)} pos + {len(neg_pairs)} neg)")

# ─── Feature extraction ───

def extract_features(zh_uri, en_uri):
    """Extract feature vector for a pair. Returns (features_dict, score)."""
    f = {}

    # F1: Name token overlap (Jaccard)
    zh_trans = zh_labels.get(zh_uri, {}).get('trans', '')
    en_label = en_labels.get(en_uri, {}).get('label', '')
    n1 = normalize_name(zh_trans)
    n2 = normalize_name(en_label)
    t1, t2 = set(n1.split()), set(n2.split())
    f['name_overlap'] = len(t1 & t2) / max(len(t1 | t2), 1) if t1 and t2 else 0.0
    if n1 == n2 and n1:
        f['name_exact'] = 1.0
    else:
        f['name_exact'] = 0.0

    # F2: Name substring (contains)
    if n1 and n2:
        f['name_contains'] = 1.0 if (n1 in n2 or n2 in n1) else 0.0
    else:
        f['name_contains'] = 0.0

    # F3: Attribute overlap (Jaccard)
    zh_a = zh_attrs.get(zh_uri, set())
    en_a = en_attrs.get(en_uri, set())
    f['attr_overlap'] = len(zh_a & en_a) / max(len(zh_a | en_a), 1) if zh_a and en_a else 0.0
    f['attr_count_diff'] = abs(len(zh_a) - len(en_a))

    # F4: Relational degree
    zh_d = len(zh_triples.get(zh_uri, []))
    en_d = len(en_triples.get(en_uri, []))
    f['degree_ratio'] = min(zh_d, en_d) / max(zh_d, en_d) if zh_d > 0 and en_d > 0 else 0.0
    f['degree_diff'] = abs(zh_d - en_d)
    f['zh_degree'] = zh_d
    f['en_degree'] = en_d

    # F5: URI local name similarity
    zh_local = zh_uri.split('/')[-1].lower()
    en_local = en_uri.split('/')[-1].lower()
    zh_local = re.sub(r'%[0-9a-fA-F]{2}', '_', zh_local)
    ut1 = set(re.split(r'[_\-]', zh_local))
    ut2 = set(re.split(r'[_\-]', en_local))
    f['uri_overlap'] = len(ut1 & ut2) / max(len(ut1 | ut2), 1) if ut1 and ut2 else 0.0

    # F6: Attribute count (absolute)
    f['zh_attr_count'] = len(zh_a)
    f['en_attr_count'] = len(en_a)

    # F7: Shared attribute ratio (relative to smaller set)
    if zh_a and en_a:
        shared = len(zh_a & en_a)
        f['shared_attr_ratio'] = shared / min(len(zh_a), len(en_a))
    else:
        f['shared_attr_ratio'] = 0.0

    # F8: Combined score (original L1)
    f['l1_score'] = (
        f['name_overlap'] * 0.40 +
        f['attr_overlap'] * 0.35 +
        f['degree_ratio'] * 0.15 +
        f['uri_overlap'] * 0.10
    )
    if f['attr_overlap'] == 0 and (zh_d > 5 or en_d > 5):
        f['l1_score'] *= 0.7

    return f

print("\nExtracting features...")
features_list = []
labels_list = []
for zh_uri, en_uri, label in all_pairs:
    feat = extract_features(zh_uri, en_uri)
    features_list.append(feat)
    labels_list.append(label)

FEATURE_NAMES = ['name_overlap', 'name_exact', 'name_contains', 'attr_overlap',
                 'attr_count_diff', 'degree_ratio', 'degree_diff',
                 'zh_degree', 'en_degree', 'uri_overlap',
                 'zh_attr_count', 'en_attr_count', 'shared_attr_ratio', 'l1_score']

X = np.array([[f[k] for k in FEATURE_NAMES] for f in features_list])
y = np.array(labels_list)

print(f"Feature matrix: {X.shape}, labels: {y.shape}, positive rate: {y.mean():.1%}")

# ─── Level 0: Fixed L1 weights ───
MERGE_TH = 0.5
REJECT_TH = 0.3

def eval_fixed_l1(X_data, y_data):
    """Evaluate fixed L1 using l1_score feature (index 13)."""
    scores = X_data[:, FEATURE_NAMES.index('l1_score')]
    tp = fp = tn = fn = 0
    escalated = 0
    for i in range(len(scores)):
        s = scores[i]
        l = y_data[i]
        if s >= MERGE_TH:
            if l == 1: tp += 1
            else: fp += 1
        elif s < REJECT_TH:
            if l == 0: tn += 1
            else: fn += 1
        else:
            escalated += 1
    intercepted = len(scores) - escalated
    accuracy = (tp + tn) / max(intercepted, 1)
    interception_rate = intercepted / len(scores)
    return {'accuracy': accuracy, 'interception': interception_rate,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'escalated': escalated}

print("\n=== Level 0: Fixed L1 Weights ===")
l0 = eval_fixed_l1(X, y)
print(f"  Accuracy: {l0['accuracy']:.1%}, Interception: {l0['interception']:.1%}")
print(f"  TP={l0['tp']} FP={l0['fp']} TN={l0['tn']} FN={l0['fn']} Esc={l0['escalated']}")

# ─── Level 1: Unified LR with 5-fold feedback simulation ───
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

print("\n=== Level 1: Unified LR (5-fold feedback simulation) ===")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train LR (excluding l1_score from training features to avoid leakage)
    TRAIN_FEATURES = [f for f in FEATURE_NAMES if f != 'l1_score']
    X_train_lr = X_train[:, [FEATURE_NAMES.index(f) for f in TRAIN_FEATURES]]
    X_test_lr = X_test[:, [FEATURE_NAMES.index(f) for f in TRAIN_FEATURES]]

    lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    lr.fit(X_train_lr, y_train)

    # Predict probabilities on test set
    probs = lr.predict_proba(X_test_lr)[:, 1]

    # Apply thresholds (merge >= 0.5, reject < 0.3)
    tp = fp = tn = fn = 0
    escalated = 0
    for i in range(len(probs)):
        p = probs[i]
        l = y_test[i]
        if p >= MERGE_TH:
            if l == 1: tp += 1
            else: fp += 1
        elif p < REJECT_TH:
            if l == 0: tn += 1
            else: fn += 1
        else:
            escalated += 1

    intercepted = len(probs) - escalated
    accuracy = (tp + tn) / max(intercepted, 1)
    interception_rate = intercepted / len(probs)

    fold_results.append({
        'fold': fold_idx + 1,
        'train_size': len(train_idx),
        'test_size': len(test_idx),
        'accuracy': round(accuracy, 4),
        'interception': round(interception_rate, 4),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'escalated': escalated,
    })
    print(f"  Fold {fold_idx+1}: acc={accuracy:.1%}, int={interception_rate:.1%}, esc={escalated}")

avg_l1_acc = np.mean([r['accuracy'] for r in fold_results])
avg_l1_int = np.mean([r['interception'] for r in fold_results])
print(f"\n  Average: accuracy={avg_l1_acc:.1%}, interception={avg_l1_int:.1%}")

# ─── Feature importance ───
print("\n=== Feature Importance (LR coefficients, trained on all data) ===")
TRAIN_FEATURES = [f for f in FEATURE_NAMES if f != 'l1_score']
lr_full = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
lr_full.fit(X[:, [FEATURE_NAMES.index(f) for f in TRAIN_FEATURES]], y)

coefs = sorted(zip(TRAIN_FEATURES, lr_full.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
for name, coef in coefs:
    print(f"  {name:25s}  w={coef:+.3f}")

# ─── Incremental feedback simulation (like MINEC Table XI) ───
print("\n=== Feedback Convergence (incremental, like Table XI) ===")

# Random shuffle
shuffled_idx = np.random.RandomState(42).permutation(len(X))
X_shuf = X[shuffled_idx]
y_shuf = y[shuffled_idx]
fold_size = len(X) // 5

feedback_results = []

# Round 0: Fixed weights
feedback_results.append({
    'round': 0, 'feedback_samples': 0,
    'accuracy': round(l0['accuracy'], 4),
    'interception': round(l0['interception'], 4),
    'escalation': round(1 - l0['interception'], 4),
})

for round_i in range(1, 6):
    train_end = fold_size * round_i
    train_idx = list(range(train_end))
    test_idx = list(range(train_end, len(X)))

    if len(test_idx) == 0:
        test_idx = list(range(len(X)))
        train_idx = list(range(len(X)))

    X_train = X_shuf[train_idx]
    y_train = y_shuf[train_idx]
    X_test = X_shuf[test_idx]
    y_test = y_shuf[test_idx]

    TRAIN_FEAT = [f for f in FEATURE_NAMES if f != 'l1_score']
    lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    lr.fit(X_train[:, [FEATURE_NAMES.index(f) for f in TRAIN_FEAT]], y_train)
    probs = lr.predict_proba(X_test[:, [FEATURE_NAMES.index(f) for f in TRAIN_FEAT]])[:, 1]

    tp = fp = tn = fn = 0
    escalated = 0
    for i in range(len(probs)):
        p = probs[i]
        l = y_test[i]
        if p >= MERGE_TH:
            if l == 1: tp += 1
            else: fp += 1
        elif p < REJECT_TH:
            if l == 0: tn += 1
            else: fn += 1
        else:
            escalated += 1

    intercepted = len(probs) - escalated
    acc = (tp + tn) / max(intercepted, 1)
    int_rate = intercepted / len(probs)

    feedback_results.append({
        'round': round_i,
        'feedback_samples': train_end,
        'accuracy': round(acc, 4),
        'interception': round(int_rate, 4),
        'escalation': round(1 - int_rate, 4),
    })
    print(f"  Round {round_i} ({train_end} samples): acc={acc:.1%}, int={int_rate:.1%}, esc={1-int_rate:.1%}")

# ─── Save ───

# Load L2 results if available
l2_accuracy = None
l2_path = RESULTS_DIR / "dbp15k_l2_results.json"
if l2_path.exists():
    with open(l2_path) as f:
        l2_data = json.load(f)
        if 'l2_accuracy' in l2_data and l2_data.get('l2_accuracy', 0) > 0.3:
            l2_accuracy = l2_data['l2_accuracy']

output = {
    'dataset': 'DBP15K ZH-EN',
    'total_pairs': len(all_pairs),
    'feature_names': FEATURE_NAMES,
    'level0_fixed': l0,
    'level1_unified_lr': {
        'fold_results': fold_results,
        'avg_accuracy': round(avg_l1_acc, 4),
        'avg_interception': round(avg_l1_int, 4),
    },
    'feedback_convergence': feedback_results,
    'feature_importance': [(name, round(coef, 4)) for name, coef in coefs],
}

if l2_accuracy:
    # Pipeline estimate
    best_l1 = feedback_results[-1]  # last round
    pipeline_fixed = l0['accuracy'] * l0['interception'] + l2_accuracy * (1 - l0['interception'])
    pipeline_adaptive = best_l1['accuracy'] * best_l1['interception'] + l2_accuracy * (1 - best_l1['interception'])
    output['pipeline_estimate'] = {
        'l2_accuracy': round(l2_accuracy, 4),
        'fixed_pipeline': round(pipeline_fixed, 4),
        'adaptive_pipeline': round(pipeline_adaptive, 4),
        'llm_cost_reduction': round(1 - (1 - best_l1['interception']) / (1 - l0['interception']), 4),
    }
    print(f"\n=== Pipeline Estimate (L2 acc={l2_accuracy:.1%}) ===")
    print(f"  Fixed pipeline:    {pipeline_fixed:.1%}")
    print(f"  Adaptive pipeline: {pipeline_adaptive:.1%}")
    print(f"  LLM cost reduction: {1 - (1-best_l1['interception'])/(1-l0['interception']):.1%}")

output_path = RESULTS_DIR / "dbp15k_hierarchy_results.json"
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to {output_path}")
