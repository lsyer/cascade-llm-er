#!/usr/bin/env python3
"""
Rerun two experiments that had bugs/stale data:

1. feedback_convergence: Fix the hardcoded interception=100% bug.
   Use select_thresholds (same as hierarchy_comparison) so that
   interception is realistic (~88%) and pairs can escalate to L2.

2. all_lr_coefficients: Extract all 42 LR coefficients (not just top-10)
   so that Figure 8 and Table IV can use the correct v2 features.
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from collections import Counter

RESULTS_DIR = "/root/workspace/tanshu_docs/experiments/release/results"

# === Load data ===
with open(f"{RESULTS_DIR}/scored_pairs.json") as f:
    pairs = json.load(f)

print(f"Loaded {len(pairs)} pairs")

# Feature names from hierarchy_comparison
with open(f"{RESULTS_DIR}/hierarchy_comparison.json") as f:
    hc = json.load(f)
feature_names = hc['feature_names']
print(f"Features: {len(feature_names)}")

# Build X, y
X = np.array([[p['features_v2'].get(f, 0) for f in feature_names] for p in pairs])
y = np.array([p['label'] for p in pairs])
types = np.array([p['type_a'].replace('activity', 'event') for p in pairs])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Label distribution: {Counter(y)}")
print(f"Type distribution: {Counter(types)}")

# === select_thresholds (same as run_experiments.py) ===
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

# ============================================================
# 1. Feedback convergence — FIXED with select_thresholds
# ============================================================
print("\n" + "=" * 60)
print("FIXED feedback convergence with select_thresholds")
print("=" * 60)

n = len(y)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rounds_data = []

# Round 0 = fixed weights
# From l1_distribution.json
with open(f"{RESULTS_DIR}/l1_accuracy.json") as f:
    l1_data = json.load(f)
rounds_data.append({
    'round': 0, 'feedback': 0,
    'l1_acc': l1_data['l1_accuracy'],
    'interception': l1_data['interception_rate'],
    'l2_esc': round(100 - l1_data['interception_rate'], 1),
    'pipeline': round(l1_data['interception_rate']/100 * l1_data['l1_accuracy'] +
                       (100 - l1_data['interception_rate'])/100 * 97.1, 1),
})
print(f"Round 0 (Fixed): acc={l1_data['l1_accuracy']:.1f}%, intercept={l1_data['interception_rate']:.1f}%")

# Rounds 1-4: simulate incremental feedback using k-fold CV.
# Each round uses k folds for training and evaluates on ALL n pairs
# (with thresholds tuned on training data only), so:
#   - Every round is evaluated on the same full test set (no shrinking test set)
#   - More training data → better model → monotonic improvement expected
#   - Consistent with hierarchy_comparison methodology
for round_num in range(1, 5):
    n_folds = round_num + 1  # Round 1 = 2-fold, Round 4 = 5-fold
    skf_round = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_decisions = np.full(n, -1)
    fold_size = n // n_folds
    train_size = (n_folds - 1) * fold_size  # approximate feedback size

    for train_idx, test_idx in skf_round.split(X_scaled, y):
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        lr.fit(X_scaled[train_idx], y[train_idx])

        # Tune thresholds on TRAINING data only
        proba_train = lr.predict_proba(X_scaled[train_idx])[:, 1]
        merge_t, reject_t = select_thresholds(proba_train, y[train_idx])

        # Apply thresholds to test data
        proba_test = lr.predict_proba(X_scaled[test_idx])[:, 1]
        for i, idx in enumerate(test_idx):
            if proba_test[i] >= merge_t:
                all_decisions[idx] = 1   # merge
            elif proba_test[i] <= reject_t:
                all_decisions[idx] = 0   # reject
            # else: -1 = escalate

    decided = all_decisions != -1
    correct = sum(1 for i in range(n) if decided[i] and all_decisions[i] == y[i])
    acc = correct / decided.sum() * 100 if decided.sum() > 0 else 0
    interception = decided.sum() / n * 100
    l2_esc = 100 - interception
    pipeline = interception / 100 * acc + l2_esc / 100 * 97.1

    rounds_data.append({
        'round': round_num, 'feedback': train_size,
        'l1_acc': round(acc, 1),
        'interception': round(interception, 1),
        'l2_esc': round(l2_esc, 1),
        'pipeline': round(pipeline, 1),
        'thresholds': {'merge': round(merge_t, 2), 'reject': round(reject_t, 2)},
    })
    print(f"Round {round_num}: feedback={train_size}, acc={acc:.1f}%, "
          f"intercept={interception:.1f}%, l2_esc={l2_esc:.1f}%, "
          f"pipeline={pipeline:.1f}%, "
          f"thresholds=(merge={merge_t:.2f}, reject={reject_t:.2f})")

with open(f"{RESULTS_DIR}/feedback_convergence.json", 'w') as f:
    json.dump(rounds_data, f, indent=2)
print(f"\n✓ Saved feedback_convergence.json")

# ============================================================
# 2. All 42 LR coefficients
# ============================================================
print("\n" + "=" * 60)
print("All 42 LR coefficients (unified model)")
print("=" * 60)

lr_full = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_full.fit(X_scaled, y)
coef = lr_full.coef_[0]

all_features = [(feature_names[i], round(coef[i], 4)) for i in range(len(feature_names))]
all_features.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"\nAll {len(all_features)} features (sorted by |weight|):")
for name, weight in all_features:
    print(f"  {name:<40}: {weight:+.4f}")

result = {
    'n_features': len(all_features),
    'features': all_features,
}
with open(f"{RESULTS_DIR}/all_lr_coefficients.json", 'w') as f:
    json.dump(result, f, indent=2)
print(f"\n✓ Saved all_lr_coefficients.json")

# Also update feature_importance.json (now with all 42)
feature_importance = {'top_features': all_features[:10]}
with open(f"{RESULTS_DIR}/feature_importance.json", 'w') as f:
    json.dump(feature_importance, f, indent=2)
print(f"✓ Updated feature_importance.json (top-10)")

print("\nDone.")
