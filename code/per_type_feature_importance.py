#!/usr/bin/env python3
"""
Experiment 25: Per-type feature importance + complete results for paper rewrite.

Outputs:
1. Per-type top-5 features (Table)
2. Full hybrid vs unified vs fixed comparison table (per-type breakdown)
3. Pipeline accuracy with different L2 models
4. Convergence: hybrid strategy over feedback rounds
"""
import json, re, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from pathlib import Path

BASE = Path("/root/workspace/tanshu_docs/experiments")
with open(BASE / "data/dataset_v3_cleaned.json") as f:
    ds = json.load(f)
with open(BASE / "results/checkpoints_v3/glm-5_checkpoint.json") as f:
    ckpt = json.load(f)
pairs = ds["same_type_pairs"]

def parse_signals(sig_str, props_a, props_b, type_a, type_b):
    feats = {"matches":0,"conflicts":0,"unknowns":0,"name_match":0,"name_conflict":0,
        "region_match":0,"region_conflict":0,"coords_match":0,"org_match":0,"org_conflict":0,
        "timeline_match":0,"timeline_conflict":0,"model_match":0,"model_conflict":0,
        "industry_match":0,"time_match":0,"participants_match":0,
        "type_consistent":1 if type_a==type_b else 0,"near_perfect":0,"prop_overlap":0.0}
    if not sig_str: return feats
    sig_lower = sig_str.lower()
    if "near-perfect" in sig_lower:
        feats["near_perfect"]=1; feats["matches"]=4; return feats
    m = re.search(r"matches=(\d+)\s+conflicts=(\d+)\s+unknowns=(\d+)", sig_str)
    if m:
        feats["matches"]=int(m.group(1)); feats["conflicts"]=int(m.group(2)); feats["unknowns"]=int(m.group(3))
    for part in sig_str.split(";")[1:]:
        part=part.strip().lower()
        if "name=match" in part: feats["name_match"]=1
        elif "name=conflict" in part: feats["name_conflict"]=1
        elif "region=match" in part: feats["region_match"]=1
        elif "region=conflict" in part: feats["region_conflict"]=1
        elif "coords=match" in part or "admin=match" in part: feats["coords_match"]=1
        elif any(x in part for x in ["org=match","org_type=match","affiliation=match"]): feats["org_match"]=1
        elif any(x in part for x in ["org=conflict","org_type=conflict","affiliation=conflict"]): feats["org_conflict"]=1
        elif "timeline=match" in part or "time=match" in part: feats["timeline_match"]=1
        elif "timeline=conflict" in part or "time=conflict" in part: feats["timeline_conflict"]=1
        elif "model=match" in part: feats["model_match"]=1
        elif "model=conflict" in part: feats["model_conflict"]=1
        elif "industry=match" in part: feats["industry_match"]=1
        elif "participants=match" in part: feats["participants_match"]=1
        elif "deploy=match" in part: feats["model_match"]=1
        elif "deploy=conflict" in part: feats["model_conflict"]=1
    def gmp(props):
        skip={"confidence","created_at","updated_at","aliases"}
        result={}
        for k,v in props.items():
            if k in skip: continue
            val=str(v).strip()
            if val and val!="{}" and val!="": result[k]=val.lower().strip()
        return result
    pa=gmp(props_a or {}); pb=gmp(props_b or {})
    ck=set(pa.keys())&set(pb.keys())
    if ck:
        matches=sum(1 for k in ck if pa[k]==pb[k])
        feats["prop_overlap"]=matches/len(ck)
    return feats

# Build dataset
records = []
feature_names = None
for p in pairs:
    pid = p["pair_id"]
    if pid not in ckpt: continue
    lr_raw = ckpt[pid].get("verdict", ckpt[pid].get("raw", "")).lower().strip()
    if lr_raw == "same": y = 1
    elif lr_raw == "different": y = 0
    else: continue
    feats = parse_signals(p.get("signals",""), p.get("props_a",{}), p.get("props_b",{}),
                          p.get("type_a",""), p.get("type_b",""))
    if feature_names is None: feature_names = sorted(feats.keys())
    records.append({"x": [feats[k] for k in feature_names], "y": y,
                    "type": p.get("type_a","unknown")})

X_all = np.array([r["x"] for r in records])
y_all = np.array([r["y"] for r in records])
types = np.array([r["type"] for r in records])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)
TYPE_ORDER = ["organization", "person", "location", "equipment", "event"]

# =====================================================
# 1. PER-TYPE FEATURE IMPORTANCE (train on all data)
# =====================================================
print("=" * 80)
print("1. PER-TYPE FEATURE IMPORTANCE (top-5 by |weight|)")
print("=" * 80)

# Also unified for comparison
lr_uni = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr_uni.fit(X_scaled, y_all)
print(f"\n{'Unified':<15}", end="")
weights_uni = list(zip(feature_names, lr_uni.coef_[0]))
for f, w in sorted(weights_uni, key=lambda x: -abs(x[1]))[:5]:
    print(f"  {f}({w:+.2f})", end="")
print()

for t in TYPE_ORDER:
    mask = types == t
    X_t = X_scaled[mask]
    y_t = y_all[mask]
    if len(y_t) < 30:
        print(f"{t:<15}  (skipped, only {len(y_t)} pairs)")
        continue
    lr_t = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_t.fit(X_t, y_t)
    weights = list(zip(feature_names, lr_t.coef_[0]))
    top5 = sorted(weights, key=lambda x: -abs(x[1]))[:5]
    print(f"{t:<15}", end="")
    for f, w in top5:
        print(f"  {f}({w:+.2f})", end="")
    print(f"  (n={len(y_t)})")

# =====================================================
# 2. HYBRID CONVERGENCE OVER FEEDBACK ROUNDS
# =====================================================
print("\n" + "=" * 80)
print("2. HYBRID STRATEGY CONVERGENCE (5-fold CV, incremental feedback)")
print("=" * 80)

def select_thresholds(proba, y_true, min_acc=75):
    best_score = 0
    best = (0.7, 0.3)
    for mt in np.arange(0.55, 0.95, 0.05):
        for rt in np.arange(0.05, 0.45, 0.05):
            decisions = np.full(len(y_true), -1)
            decisions[proba >= mt] = 1
            decisions[proba <= rt] = 0
            decided = decisions != -1
            if decided.sum() < 5: continue
            correct = (decisions[decided] == y_true[decided]).sum()
            acc = correct / decided.sum() * 100
            interc = decided.sum() / len(y_true) * 100
            if acc < min_acc: continue
            if acc * interc > best_score:
                best_score = acc * interc
                best = (mt, rt)
    return best

def run_hybrid_fold(train_idx, test_idx, min_samples=50):
    """Run hybrid strategy on given train/test split. Returns predictions."""
    preds = np.full(len(y_all), -1)
    
    # Train unified for fallback
    lr_u = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_u.fit(X_scaled[train_idx], y_all[train_idx])
    proba_train_u = lr_u.predict_proba(X_scaled[train_idx])[:, 1]
    u_mt, u_rt = select_thresholds(proba_train_u, y_all[train_idx])
    
    for t in TYPE_ORDER:
        train_mask = types[train_idx] == t
        test_mask = types[test_idx] == t
        n_train = train_mask.sum()
        X_test_t = X_scaled[test_idx][test_mask]
        global_test = test_idx[test_mask]
        
        if n_train >= min_samples:
            X_tr = X_scaled[train_idx][train_mask]
            y_tr = y_all[train_idx][train_mask]
            lr_t = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
            lr_t.fit(X_tr, y_tr)
            proba_tr = lr_t.predict_proba(X_tr)[:, 1]
            mt, rt = select_thresholds(proba_tr, y_tr)
            proba_te = lr_t.predict_proba(X_test_t)[:, 1]
        else:
            proba_te = lr_u.predict_proba(X_test_t)[:, 1]
            mt, rt = u_mt, u_rt
        
        tp = np.full(len(X_test_t), -1)
        tp[proba_te >= mt] = 1
        tp[proba_te <= rt] = 0
        preds[global_test] = tp
    return preds

def run_unified_fold(train_idx, test_idx):
    """Run unified strategy."""
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_scaled[train_idx], y_all[train_idx])
    proba_tr = lr.predict_proba(X_scaled[train_idx])[:, 1]
    mt, rt = select_thresholds(proba_tr, y_all[train_idx])
    
    preds = np.full(len(y_all), -1)
    proba_te = lr.predict_proba(X_scaled[test_idx])[:, 1]
    tp = np.full(len(test_idx), -1)
    tp[proba_te >= mt] = 1
    tp[proba_te <= rt] = 0
    preds[test_idx] = tp
    return preds

def compute_metrics(preds):
    decided = preds != -1
    correct = (preds[decided] == y_all[decided]).sum()
    n = decided.sum()
    acc = correct / n * 100 if n > 0 else 0
    interc = n / len(y_all) * 100
    # merge precision
    merge_mask = (preds == 1) & decided
    mp = (y_all[merge_mask] == 1).sum() / max(1, merge_mask.sum()) * 100
    return acc, interc, mp

np.random.seed(42)
indices = np.random.permutation(len(X_all))
folds = np.array_split(indices, 5)

print(f"\n{'Round':<7} {'FB':>5} {'--- Unified ---':<25} {'--- Hybrid ---':<25}")
print(f"{'':7} {'':5} {'Acc':>7} {'Interc':>7} {'MP':>6}   {'Acc':>7} {'Interc':>7} {'MP':>6}")
print("-" * 70)

for round_num in range(6):
    if round_num == 0:
        # Baseline: fixed L1
        from collections import Counter
        print(f"{'0':<7} {'0':>5} {'65.8':>6}% {'39.6':>6}% {'67.3':>5}%   (fixed L1 baseline)")
        continue
    
    if round_num >= 5:
        break  # All data used as feedback, no test set left
    
    train_idx = np.concatenate([folds[j] for j in range(round_num)])
    test_idx = np.concatenate([folds[j] for j in range(round_num, 5)])
    
    # Unified
    preds_u = run_unified_fold(train_idx, test_idx)
    acc_u, int_u, mp_u = compute_metrics(preds_u)
    
    # Hybrid
    preds_h = run_hybrid_fold(train_idx, test_idx, min_samples=50)
    acc_h, int_h, mp_h = compute_metrics(preds_h)
    
    fb_size = len(train_idx) if round_num < 5 else len(indices)
    print(f"{round_num:<7} {fb_size:>5} {acc_u:>6.1f}% {int_u:>6.1f}% {mp_u:>5.1f}%   "
          f"{acc_h:>6.1f}% {int_h:>6.1f}% {mp_h:>5.1f}%")

# =====================================================
# 3. PIPELINE ACCURACY TABLE (with all 4 L2 models)
# =====================================================
print("\n" + "=" * 80)
print("3. PIPELINE ACCURACY: Fixed vs Unified vs Hybrid (with each L2 model)")
print("=" * 80)

l2_accs = {"GLM-5": 97.1, "GLM-4.5-Air": 94.4, "Qwen3.6": 91.8, "GLM-4.5": 90.7}

# Get best hybrid metrics (from min_samples=50, round=1)
np.random.seed(42)
indices2 = np.random.permutation(len(X_all))
folds2 = np.array_split(indices2, 5)
train_r1 = folds2[0]
test_r1 = np.concatenate(folds2[1:])

# Unified round 1
preds_u = run_unified_fold(train_r1, test_r1)
acc_u, int_u, mp_u = compute_metrics(preds_u)

# Hybrid round 1
preds_h = run_hybrid_fold(train_r1, test_r1, min_samples=50)
acc_h, int_h, mp_h = compute_metrics(preds_h)

print(f"\n{'Strategy':<20} {'L1 Acc':>7} {'Interc':>7}", end="")
for m in l2_accs:
    print(f" {m:>12}", end="")
print()
print("-" * 75)

print(f"{'Fixed L1':<20} {'65.8':>6}% {'39.6':>6}%", end="")
for m, l2a in l2_accs.items():
    pipe = 0.396 * 65.8 + 0.604 * l2a
    print(f" {pipe:>10.1f}%", end="")
print("  (baseline)")

print(f"{'Learned Unified LR':<20} {acc_u:>6.1f}% {int_u:>6.1f}%", end="")
for m, l2a in l2_accs.items():
    pipe = int_u/100 * acc_u + (1-int_u/100) * l2a
    print(f" {pipe:>10.1f}%", end="")
print()

print(f"{'Learned Hybrid LR':<20} {acc_h:>6.1f}% {int_h:>6.1f}%", end="")
for m, l2a in l2_accs.items():
    pipe = int_h/100 * acc_h + (1-int_h/100) * l2a
    print(f" {pipe:>10.1f}%", end="")
print()

# =====================================================
# 4. DETAILED PER-TYPE TABLE (hybrid, round 1)
# =====================================================
print("\n" + "=" * 80)
print("4. DETAILED PER-TYPE BREAKDOWN (Hybrid LR, 5-fold CV)")
print("=" * 80)

# Full 5-fold CV hybrid
preds_h_full = np.full(len(X_all), -1)
for i in range(5):
    train_idx = folds[i]
    test_idx = np.concatenate([folds[j] for j in range(5) if j != i])
    p = run_hybrid_fold(train_idx, test_idx, min_samples=50)
    # Only fill in test predictions (don't overwrite)
    mask_new = (p != -1) & (preds_h_full == -1)
    preds_h_full[mask_new] = p[mask_new]

print(f"\n{'Type':<13} {'Total':>6} {'Decided':>8} {'Correct':>8} {'Acc':>7} {'Interc':>7}")
print("-" * 55)
for t in TYPE_ORDER:
    mask = types == t
    d = mask & (preds_h_full != -1)
    c = (preds_h_full[d] == y_all[d]).sum()
    nd = d.sum()
    acc = c / nd * 100 if nd > 0 else 0
    interc = nd / mask.sum() * 100
    print(f"{t:<13} {mask.sum():>6} {nd:>8} {c:>8} {acc:>6.1f}% {interc:>6.1f}%")

d_all = preds_h_full != -1
c_all = (preds_h_full[d_all] == y_all[d_all]).sum()
nd_all = d_all.sum()
print("-" * 55)
print(f"{'OVERALL':<13} {len(y_all):>6} {nd_all:>8} {c_all:>8} "
      f"{c_all/nd_all*100:>6.1f}% {nd_all/len(y_all)*100:>6.1f}%")
