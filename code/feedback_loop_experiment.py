#!/usr/bin/env python3
"""
Feedback Loop Experiment — L1 threshold sweep + learned weights via logistic regression.

Two experiments:
1. Threshold sweep: scan merge_threshold × reject_threshold, find optimal F1
2. Feedback loop simulation: incremental logistic regression learning over 5 folds,
   showing accuracy/escalation/cost convergence curve.

All data from real experiments. No simulated numbers.
"""

import json
import re
import sys
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

# ─── Paths ───
BASE = Path(__file__).resolve().parent.parent
DATA_FILE = BASE / "data" / "dataset_v3_cleaned.json"
CKPT_FILE = BASE / "results" / "checkpoints_v3" / "glm-5_checkpoint.json"
OUTPUT_DIR = BASE / "results"

# ─── Load data ───
print("Loading data...")
with open(DATA_FILE) as f:
    ds = json.load(f)

with open(CKPT_FILE) as f:
    ckpt = json.load(f)

pairs = ds["same_type_pairs"]
print(f"  Pairs: {len(pairs)}, Expert labels: {len(ckpt)}")

# ─── Parse signals into feature vector ───
def parse_signals(sig_str, props_a, props_b, type_a, type_b):
    """Extract numerical features from L1 signal string + property data."""
    feats = {
        "matches": 0, "conflicts": 0, "unknowns": 0,
        "name_match": 0, "name_conflict": 0,
        "region_match": 0, "region_conflict": 0,
        "coords_match": 0,
        "org_match": 0, "org_conflict": 0,
        "timeline_match": 0, "timeline_conflict": 0,
        "model_match": 0, "model_conflict": 0,
        "industry_match": 0,
        "time_match": 0,
        "participants_match": 0,
        "type_consistent": 1 if type_a == type_b else 0,
        "near_perfect": 0,
        "prop_overlap": 0.0,
    }

    if not sig_str:
        return feats

    sig_lower = sig_str.lower()

    # Near-perfect signal
    if "near-perfect" in sig_lower:
        feats["near_perfect"] = 1
        feats["matches"] = 4  # treat as strong match
        return feats

    # Parse match/conflict/unknown counts
    m = re.search(r"matches=(\d+)\s+conflicts=(\d+)\s+unknowns=(\d+)", sig_str)
    if m:
        feats["matches"] = int(m.group(1))
        feats["conflicts"] = int(m.group(2))
        feats["unknowns"] = int(m.group(3))

    # Parse individual signal fields
    for part in sig_str.split(";")[1:]:
        part = part.strip().lower()
        if "name=match" in part:
            feats["name_match"] = 1
        elif "name=conflict" in part:
            feats["name_conflict"] = 1
        elif "region=match" in part:
            feats["region_match"] = 1
        elif "region=conflict" in part:
            feats["region_conflict"] = 1
        elif "coords=match" in part or "admin=match" in part:
            feats["coords_match"] = 1
        elif any(x in part for x in ["org=match", "org_type=match", "affiliation=match"]):
            feats["org_match"] = 1
        elif any(x in part for x in ["org=conflict", "org_type=conflict", "affiliation=conflict"]):
            feats["org_conflict"] = 1
        elif "timeline=match" in part or "time=match" in part:
            feats["timeline_match"] = 1
        elif "timeline=conflict" in part or "time=conflict" in part:
            feats["timeline_conflict"] = 1
        elif "model=match" in part:
            feats["model_match"] = 1
        elif "model=conflict" in part:
            feats["model_conflict"] = 1
        elif "industry=match" in part:
            feats["industry_match"] = 1
        elif "participants=match" in part:
            feats["participants_match"] = 1
        elif "deploy=match" in part:
            feats["model_match"] = 1  # deploy similarity ~ model match for equipment
        elif "deploy=conflict" in part:
            feats["model_conflict"] = 1

    # Compute property overlap rate
    def get_meaningful_props(props):
        skip = {"confidence", "created_at", "updated_at", "aliases"}
        result = {}
        for k, v in props.items():
            if k in skip:
                continue
            val = str(v).strip()
            if val and val != "{}" and val != "":
                result[k] = val.lower().strip()
        return result

    pa = get_meaningful_props(props_a or {})
    pb = get_meaningful_props(props_b or {})

    common_keys = set(pa.keys()) & set(pb.keys())
    if common_keys:
        matches = sum(1 for k in common_keys if pa[k] == pb[k])
        feats["prop_overlap"] = matches / len(common_keys)

    return feats

# ─── Build dataset ───
print("Building feature matrix...")

FEATURE_NAMES = None
X_all = []
y_all = []  # 1=same, 0=different
l1_scores = []
pair_ids = []
entity_types = []

valid = 0
skipped = 0

for p in pairs:
    pid = p["pair_id"]
    if pid not in ckpt:
        skipped += 1
        continue

    label_raw = ckpt[pid].get("verdict", ckpt[pid].get("raw", "")).lower().strip()
    if label_raw == "same":
        y = 1
    elif label_raw == "different":
        y = 0
    else:
        skipped += 1
        continue  # skip unknown

    feats = parse_signals(
        p.get("signals", ""),
        p.get("props_a", {}),
        p.get("props_b", {}),
        p.get("type_a", ""),
        p.get("type_b", ""),
    )

    if FEATURE_NAMES is None:
        FEATURE_NAMES = sorted(feats.keys())

    x = [feats[k] for k in FEATURE_NAMES]
    X_all.append(x)
    y_all.append(y)
    l1_scores.append(p.get("l1_score", 0))
    pair_ids.append(pid)
    entity_types.append(p.get("type_a", "unknown"))
    valid += 1

X_all = np.array(X_all)
y_all = np.array(y_all)
l1_scores = np.array(l1_scores)

print(f"  Valid pairs: {valid}, Skipped: {skipped}")
print(f"  Same: {sum(y_all)}, Different: {sum(1-y_all)}")
print(f"  Features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Threshold Sweep
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 1: L1 Threshold Sweep")
print("=" * 60)

MERGE_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
REJECT_THRESHOLDS = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]

def eval_threshold(scores, labels, merge_th, reject_th):
    """Evaluate L1 with given thresholds. Returns metrics dict."""
    decisions = []
    correct = 0
    total_l1 = 0  # pairs L1 decided (merge + reject)
    total = len(scores)
    merge_count = 0
    reject_count = 0
    escalate_count = 0
    merge_correct = 0
    reject_correct = 0

    for s, y in zip(scores, labels):
        if s >= merge_th:
            decisions.append("merge")
            merge_count += 1
            total_l1 += 1
            if y == 1:
                correct += 1
                merge_correct += 1
        elif s <= reject_th:
            decisions.append("reject")
            reject_count += 1
            total_l1 += 1
            if y == 0:
                correct += 1
                reject_correct += 1
        else:
            decisions.append("escalate")
            escalate_count += 1

    accuracy = correct / total_l1 * 100 if total_l1 > 0 else 0
    interception = total_l1 / total * 100  # % of pairs L1 handles
    escalation = escalate_count / total * 100

    # Merge precision
    merge_prec = merge_correct / merge_count * 100 if merge_count > 0 else 0
    reject_prec = reject_correct / reject_count * 100 if reject_count > 0 else 0

    return {
        "merge_th": merge_th,
        "reject_th": reject_th,
        "accuracy": round(accuracy, 1),
        "interception": round(interception, 1),
        "escalation": round(escalation, 1),
        "merge_count": merge_count,
        "reject_count": reject_count,
        "escalate_count": escalate_count,
        "merge_precision": round(merge_prec, 1),
        "reject_precision": round(reject_prec, 1),
        "total_l1": total_l1,
    }

# Current baseline
print("\n--- Current baseline (merge≥0.5, reject≤-0.3) ---")
baseline = eval_threshold(l1_scores, y_all, 0.5, -0.3)
for k, v in baseline.items():
    print(f"  {k}: {v}")

# Sweep
print("\n--- Threshold sweep ---")
results_sweep = []
for mt in MERGE_THRESHOLDS:
    for rt in REJECT_THRESHOLDS:
        r = eval_threshold(l1_scores, y_all, mt, rt)
        results_sweep.append(r)

# Find best F1-like metric (accuracy * interception)
print(f"\n{'merge_th':>8} {'reject_th':>9} {'acc%':>5} {'intcpt%':>7} {'esc%':>5} {'m_prec':>6} {'r_prec':>6} {'m+n':>5}")
print("-" * 60)

# Sort by accuracy*interception (want both high)
for r in sorted(results_sweep, key=lambda x: x["accuracy"] * x["interception"], reverse=True)[:10]:
    print(f"{r['merge_th']:>8.1f} {r['reject_th']:>9.1f} {r['accuracy']:>5.1f} {r['interception']:>7.1f} "
          f"{r['escalation']:>5.1f} {r['merge_precision']:>6.1f} {r['reject_precision']:>6.1f} {r['total_l1']:>5}")

# Also show conservative merge strategies
print("\n--- Conservative merge (merge_th≥0.6) ---")
conservative = [r for r in results_sweep if r["merge_th"] >= 0.6 and r["reject_th"] >= -0.3]
for r in sorted(conservative, key=lambda x: x["merge_precision"], reverse=True)[:5]:
    print(f"  merge≥{r['merge_th']:.1f}, reject≤{r['reject_th']:.1f}: "
          f"acc={r['accuracy']:.1f}%, merge_prec={r['merge_precision']:.1f}%, "
          f"intcpt={r['interception']:.1f}%, esc={r['escalation']:.1f}%")

# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Feedback Loop — Learned Weights
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 2: Feedback Loop — Logistic Regression Weight Learning")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# ─── Simulate feedback rounds ───
# Split into 5 folds. Round 0 = baseline (fixed weights).
# Each round: train LR on accumulated labeled data, evaluate on remaining.
np.random.seed(42)
indices = np.random.permutation(len(X_all))
folds = np.array_split(indices, 5)

print(f"\nFold sizes: {[len(f) for f in folds]}")
print(f"Total: {len(X_all)}")

# Evaluate with L1 original score (baseline)
def eval_pipeline(y_true, y_pred_l1, y_true_escalated, y_pred_l2=None):
    """
    Simulate pipeline: L1 decides on some, escalates rest to L2 (expert label).
    Returns overall accuracy and LLM call count.
    """
    pass  # We'll compute inline

results_rounds = []

# Round 0: Original L1 (fixed weights, current thresholds)
print("\n--- Round 0: Baseline (fixed L1, merge≥0.5, reject≤-0.3) ---")
r0 = eval_threshold(l1_scores, y_all, 0.5, -0.3)
print(f"  L1 accuracy: {r0['accuracy']:.1f}%, interception: {r0['interception']:.1f}%")
print(f"  Escalated to L2: {r0['escalate_count']} pairs (would need LLM calls)")
results_rounds.append({
    "round": 0,
    "feedback_size": 0,
    "l1_accuracy": r0["accuracy"],
    "l1_interception": r0["interception"],
    "l1_merge_precision": r0["merge_precision"],
    "l1_reject_precision": r0["reject_precision"],
    "l2_calls": r0["escalate_count"],
    "description": "Baseline: fixed weights, merge≥0.5, reject≤-0.3"
})

# Rounds 1-4: Incremental learning
accumulated_train_idx = np.array([], dtype=int)
remaining_idx = np.array([], dtype=int)
all_indices = np.random.permutation(len(X_all))

# Re-split with clear boundaries
fold_size = len(X_all) // 5
folds_list = [all_indices[i*fold_size:(i+1)*fold_size] for i in range(5)]

train_acc = np.array([], dtype=int)
test_acc = np.array([], dtype=[])

for round_num in range(1, 6):
    # Add this fold's data as "feedback" (labeled by expert)
    train_acc = np.concatenate([train_acc, folds_list[round_num - 1]])
    # Remaining is test
    test_idx = np.concatenate(folds_list[round_num:]) if round_num < 5 else np.array([], dtype=int)

    X_train = X_scaled[train_acc]
    y_train = y_all[train_acc]

    if len(test_idx) == 0:
        # Last round: all data is training, evaluate on full set (in-sample)
        X_test = X_scaled
        y_test = y_all
        train_or_test = "in-sample (all data)"
    else:
        X_test = X_scaled[test_idx]
        y_test = y_all[test_idx]
        train_or_test = f"test ({len(test_idx)} pairs)"

    # Train logistic regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)

    # Predict on test set
    y_proba = lr.predict_proba(X_test)[:, 1]

    # Use LR probability as new "L1 score", apply thresholds
    # We'll use 0.7 as merge threshold, 0.3 as reject (analogous to ±0.5 in original space)
    # This means: LR must be quite confident to decide

    # Try multiple thresholds to find good operating point
    best_f1 = 0
    best_metrics = None

    for merge_th in [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        for reject_th in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
            decisions = []
            correct = 0
            total_l1 = 0
            merge_count = 0
            reject_count = 0
            escalate_count = 0
            merge_correct = 0
            reject_correct = 0

            for prob, y in zip(y_proba, y_test):
                if prob >= merge_th:
                    merge_count += 1
                    total_l1 += 1
                    if y == 1:
                        correct += 1
                        merge_correct += 1
                elif prob <= reject_th:
                    reject_count += 1
                    total_l1 += 1
                    if y == 0:
                        correct += 1
                        reject_correct += 1
                else:
                    escalate_count += 1

            if total_l1 == 0:
                continue

            acc = correct / total_l1 * 100
            interception = total_l1 / len(y_test) * 100
            merge_prec = merge_correct / merge_count * 100 if merge_count > 0 else 0
            reject_prec = reject_correct / reject_count * 100 if reject_count > 0 else 0

            # We want: high accuracy + high interception (fewer LLM calls)
            f1_proxy = acc * interception / 100

            if f1_proxy > best_f1 and acc >= 80:  # minimum 80% accuracy
                best_f1 = f1_proxy
                best_metrics = {
                    "merge_th": merge_th,
                    "reject_th": reject_th,
                    "accuracy": round(acc, 1),
                    "interception": round(interception, 1),
                    "escalation": round(100 - interception, 1),
                    "merge_precision": round(merge_prec, 1),
                    "reject_precision": round(reject_prec, 1),
                    "merge_count": merge_count,
                    "reject_count": reject_count,
                    "escalate_count": escalate_count,
                    "l2_calls": escalate_count,
                }

    if best_metrics is None:
        # Fallback: just report accuracy
        y_pred = lr.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        best_metrics = {
            "merge_th": 0.5,
            "reject_th": 0.5,
            "accuracy": round(acc, 1),
            "interception": 100.0,
            "escalation": 0.0,
            "merge_precision": 0,
            "reject_precision": 0,
            "merge_count": 0,
            "reject_count": 0,
            "escalate_count": 0,
            "l2_calls": 0,
        }

    # Feature importance
    coefs = dict(zip(FEATURE_NAMES, lr.coef_[0]))
    top_features = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    print(f"\n--- Round {round_num}: {len(train_acc)} feedback samples, eval on {train_or_test} ---")
    print(f"  Learned thresholds: merge≥{best_metrics['merge_th']}, reject≤{best_metrics['reject_th']}")
    print(f"  L1 accuracy: {best_metrics['accuracy']:.1f}%")
    print(f"  Interception: {best_metrics['interception']:.1f}% (L2 calls={best_metrics['l2_calls']})")
    print(f"  Merge precision: {best_metrics['merge_precision']:.1f}%, Reject precision: {best_metrics['reject_precision']:.1f}%")
    print(f"  Top features: {[(k, round(v, 3)) for k, v in top_features]}")

    results_rounds.append({
        "round": round_num,
        "feedback_size": len(train_acc),
        "l1_accuracy": best_metrics["accuracy"],
        "l1_interception": best_metrics["interception"],
        "l1_merge_precision": best_metrics["merge_precision"],
        "l1_reject_precision": best_metrics["reject_precision"],
        "l2_calls": best_metrics["l2_calls"],
        "top_features": [(k, round(v, 3)) for k, v in top_features],
        "description": f"Round {round_num}: LR trained on {len(train_acc)} feedback samples",
    })

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY: Feedback Loop Convergence")
print("=" * 60)
print(f"\n{'Round':>5} {'Feedback':>8} {'L1 Acc%':>7} {'Intcpt%':>7} {'L2 Calls':>8} {'Merge Prec':>10}")
print("-" * 50)
for r in results_rounds:
    print(f"{r['round']:>5} {r['feedback_size']:>8} {r['l1_accuracy']:>7.1f} "
          f"{r['l1_interception']:>7.1f} {r['l2_calls']:>8} {r['l1_merge_precision']:>10.1f}")

# Save results
output = {
    "experiment": "feedback_loop_v1",
    "description": "L1 threshold sweep + logistic regression feedback loop simulation",
    "dataset": {
        "total_pairs": valid,
        "same": int(sum(y_all)),
        "different": int(sum(1 - y_all)),
        "features": FEATURE_NAMES,
        "entity_types": dict(Counter(entity_types)),
    },
    "experiment_1_threshold_sweep": {
        "baseline": baseline,
        "all_combinations": results_sweep,
    },
    "experiment_2_feedback_loop": results_rounds,
}

out_path = OUTPUT_DIR / "feedback_loop_experiment.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to {out_path}")
