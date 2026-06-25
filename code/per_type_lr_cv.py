#!/usr/bin/env python3
"""
Experiment 23: Per-type LR + per-type thresholds (strict 5-fold CV)

Protocol:
  For each fold:
    For each type:
      1. Train LR on that type's training data
      2. Select optimal (merge_th, reject_th) ON TRAINING DATA
      3. Apply trained model + selected thresholds to TEST data
    Collect all test predictions → compute per-type + overall metrics

This is fully fair: thresholds are selected on train, evaluated on test.
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

def eval_thresholds(proba, y_true, merge_th, reject_th):
    decisions = np.full(len(y_true), -1)
    decisions[proba >= merge_th] = 1
    decisions[proba <= reject_th] = 0
    decided = decisions != -1
    if decided.sum() == 0:
        return {"acc": 0, "interc": 0, "n_decided": 0}
    correct = (decisions[decided] == y_true[decided]).sum()
    return {
        "acc": correct / decided.sum() * 100,
        "interc": decided.sum() / len(y_true) * 100,
        "n_decided": int(decided.sum()),
        "n_correct": int(correct),
    }

def select_thresholds_on_train(proba_train, y_train, min_acc=70):
    """Select thresholds on training data to maximize acc × interc, 
    with minimum accuracy constraint."""
    best_score = 0
    best = (0.7, 0.3)  # default
    for mt in np.arange(0.55, 0.95, 0.05):
        for rt in np.arange(0.05, 0.45, 0.05):
            r = eval_thresholds(proba_train, y_train, mt, rt)
            if r["n_decided"] < 5: continue
            if r["acc"] < min_acc: continue
            score = r["acc"] * r["interc"]
            if score > best_score:
                best_score = score
                best = (mt, rt)
    return best

# 5-fold split
np.random.seed(42)
all_indices = np.random.permutation(len(X_all))
fold_size = len(X_all) // 5
folds_list = [all_indices[i*fold_size:(i+1)*fold_size] for i in range(5)]

# ═══════════════════════════════════════════════════════════════
# Strategy A: Unified LR + unified thresholds (paper baseline)
# ═══════════════════════════════════════════════════════════════
preds_unified = np.full(len(X_all), -1)

for i in range(5):
    train_idx = folds_list[i]
    test_idx = np.concatenate([folds_list[j] for j in range(5) if j != i])
    
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_scaled[train_idx], y_all[train_idx])
    
    # Select thresholds on training data
    proba_train = lr.predict_proba(X_scaled[train_idx])[:, 1]
    best_mt, best_rt = select_thresholds_on_train(proba_train, y_all[train_idx], min_acc=75)
    
    # Apply to test
    proba_test = lr.predict_proba(X_scaled[test_idx])[:, 1]
    test_preds = np.full(len(test_idx), -1)
    test_preds[proba_test >= best_mt] = 1
    test_preds[proba_test <= best_rt] = 0
    preds_unified[test_idx] = test_preds

# ═══════════════════════════════════════════════════════════════
# Strategy B: Per-type LR + per-type thresholds
# ═══════════════════════════════════════════════════════════════
preds_per_type = np.full(len(X_all), -1)
selected_thresholds_log = []  # Track what thresholds were selected per fold per type

for i in range(5):
    train_idx = folds_list[i]
    test_idx = np.concatenate([folds_list[j] for j in range(5) if j != i])
    
    for t in TYPE_ORDER:
        # Split by type
        train_mask_t = (types[train_idx] == t)
        test_mask_t = (types[test_idx] == t)
        
        train_local = np.where(train_mask_t)[0]
        test_local = np.where(test_mask_t)[0]
        
        if len(train_local) < 20 or test_local.sum() == 0:
            continue
        
        X_train_t = X_scaled[train_idx][train_mask_t]
        y_train_t = y_all[train_idx][train_mask_t]
        
        # Train per-type LR
        lr_t = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        lr_t.fit(X_train_t, y_train_t)
        
        # Select thresholds on training data
        proba_train_t = lr_t.predict_proba(X_train_t)[:, 1]
        best_mt, best_rt = select_thresholds_on_train(proba_train_t, y_train_t, min_acc=75)
        
        # Apply to test
        X_test_t = X_scaled[test_idx][test_mask_t]
        proba_test_t = lr_t.predict_proba(X_test_t)[:, 1]
        
        test_preds_t = np.full(len(X_test_t), -1)
        test_preds_t[proba_test_t >= best_mt] = 1
        test_preds_t[proba_test_t <= best_rt] = 0
        
        global_test_idx = test_idx[test_mask_t]
        preds_per_type[global_test_idx] = test_preds_t
        
        selected_thresholds_log.append({
            "fold": i, "type": t, "merge_th": best_mt, "reject_th": best_rt,
            "n_train": len(train_local), "n_test": int(test_local.sum())
        })

# ═══════════════════════════════════════════════════════════════
# Results comparison
# ═══════════════════════════════════════════════════════════════
print("=" * 85)
print("PER-TYPE COMPARISON: Unified LR vs Per-type LR (5-fold CV)")
print("Thresholds selected on training fold, evaluated on test fold")
print("=" * 85)

print(f"\n{'Type':<13} │ {'--- Unified LR ---':<26} │ {'--- Per-type LR ---':<26} │ {'Delta Acc':>8}")
print(f"{'':13} │ {'Acc':>7} {'Interc':>7} {'n':>6}   │ {'Acc':>7} {'Interc':>7} {'n':>6}   │")
print("─" * 85)

for t in TYPE_ORDER:
    mask = types == t
    y_t = y_all[mask]
    
    # Unified
    ru = eval_thresholds(np.zeros(len(y_t)), y_t, 999, -999)  # dummy proba
    decided_u = mask & (preds_unified != -1)
    correct_u = (preds_unified[decided_u] == y_all[decided_u]).sum()
    n_decided_u = decided_u.sum()
    acc_u = correct_u / n_decided_u * 100 if n_decided_u > 0 else 0
    interc_u = n_decided_u / mask.sum() * 100
    
    # Per-type
    decided_p = mask & (preds_per_type != -1)
    correct_p = (preds_per_type[decided_p] == y_all[decided_p]).sum()
    n_decided_p = decided_p.sum()
    acc_p = correct_p / n_decided_p * 100 if n_decided_p > 0 else 0
    interc_p = n_decided_p / mask.sum() * 100
    
    delta = acc_p - acc_u
    print(f"{t:<13} │ {acc_u:>6.1f}% {interc_u:>6.1f}% {n_decided_u:>5}   │ "
          f"{acc_p:>6.1f}% {interc_p:>6.1f}% {n_decided_p:>5}   │ {delta:>+7.1f}pp")

# Overall
decided_u = preds_unified != -1
correct_u = (preds_unified[decided_u] == y_all[decided_u]).sum()
n_decided_u = decided_u.sum()
acc_u = correct_u / n_decided_u * 100
interc_u = n_decided_u / len(y_all) * 100

decided_p = preds_per_type != -1
correct_p = (preds_per_type[decided_p] == y_all[decided_p]).sum()
n_decided_p = decided_p.sum()
acc_p = correct_p / n_decided_p * 100
interc_p = n_decided_p / len(y_all) * 100

print("─" * 85)
print(f"{'OVERALL':<13} │ {acc_u:>6.1f}% {interc_u:>6.1f}% {n_decided_u:>5}   │ "
      f"{acc_p:>6.1f}% {interc_p:>6.1f}% {n_decided_p:>5}   │ {acc_p-acc_u:>+7.1f}pp")
print()

# Selected thresholds summary
print("=" * 85)
print("SELECTED THRESHOLDS (per fold, per type)")
print("=" * 85)
th_by_type = defaultdict(list)
for entry in selected_thresholds_log:
    th_by_type[entry["type"]].append((entry["merge_th"], entry["reject_th"]))

print(f"\n{'Type':<13} {'Merge≥ (5 folds)':<30} {'Reject≤ (5 folds)':<30}")
for t in TYPE_ORDER:
    mts = [f"{x[0]:.2f}" for x in th_by_type[t]]
    rts = [f"{x[1]:.2f}" for x in th_by_type[t]]
    avg_mt = np.mean([x[0] for x in th_by_type[t]])
    avg_rt = np.mean([x[1] for x in th_by_type[t]])
    print(f"{t:<13} {' '.join(mts):<30} {' '.join(rts):<30}")
    print(f"{'':13} avg_merge={avg_mt:.2f}, avg_reject={avg_rt:.2f}")

# Pipeline cost analysis
print("\n" + "=" * 85)
print("PIPELINE COST ANALYSIS")
print("=" * 85)

# With L2 at 97.1% (GLM-5), compute effective pipeline accuracy
# pipeline_acc = interc × L1_acc + (1-interc) × L2_acc
l2_acc = 97.1

for label, acc, interc in [
    ("Unified LR", acc_u, interc_u),
    ("Per-type LR", acc_p, interc_p),
]:
    pipeline = interc/100 * acc + (1-interc/100) * l2_acc
    print(f"  {label:<20}: L1 Acc={acc:.1f}%, Interc={interc:.1f}%, "
          f"Pipeline≈{pipeline:.1f}%, L2 calls={100-interc:.1f}%")
