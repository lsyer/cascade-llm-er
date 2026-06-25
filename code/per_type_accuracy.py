#!/usr/bin/env python3
"""
Per-type accuracy breakdown for:
1. Fixed-weight L1 (baseline)
2. Learned L1 (logistic regression, 5-fold CV)
3. L2 models (GLM-5, GLM-4.5-Air, Qwen3.6, GLM-4.5)
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

# Also load L2 results
l2_file = BASE / "results/l2_v3_comparison.json"
with open(l2_file) as f:
    l2_data = json.load(f)

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

# Build dataset with type info
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
    records.append({
        "pid": pid,
        "x": [feats[k] for k in feature_names],
        "y": y,
        "type": p.get("type_a","unknown"),
        "l1_score": p.get("l1_score", 0),
        "l1_decision": p.get("l1_decision",""),
    })

X_all = np.array([r["x"] for r in records])
y_all = np.array([r["y"] for r in records])
types = np.array([r["type"] for r in records])
l1_scores = np.array([r["l1_score"] for r in records])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

print(f"Total pairs: {len(records)}")
print()

# =====================================================
# 1. Fixed-weight L1 per-type accuracy
# =====================================================
print("=" * 70)
print("1. FIXED-WEIGHT L1 (baseline): per-type accuracy on INTERCEPTED pairs")
print("=" * 70)

type_stats_fixed = defaultdict(lambda: {"correct": 0, "total": 0, "intercepted": 0})
for r in records:
    t = r["type"]
    type_stats_fixed[t]["total"] += 1
    # L1 decision from original score
    score = r["l1_score"]
    if score >= 0.5:
        type_stats_fixed[t]["intercepted"] += 1
        if r["y"] == 1: type_stats_fixed[t]["correct"] += 1
    elif score <= -0.3:
        type_stats_fixed[t]["intercepted"] += 1
        if r["y"] == 0: type_stats_fixed[t]["correct"] += 1

print(f"\n{'Type':<15} {'Intercepted':>12} {'Correct':>8} {'Accuracy':>10}")
print("-" * 50)
for t in ["organization","person","location","equipment","event"]:
    s = type_stats_fixed[t]
    acc = s["correct"]/s["intercepted"]*100 if s["intercepted"]>0 else 0
    print(f"{t:<15} {s['intercepted']:>12} {s['correct']:>8} {acc:>9.1f}%")
total_int = sum(s["intercepted"] for s in type_stats_fixed.values())
total_cor = sum(s["correct"] for s in type_stats_fixed.values())
print(f"{'TOTAL':<15} {total_int:>12} {total_cor:>8} {total_cor/total_int*100:>9.1f}%")

# =====================================================
# 2. Learned L1 (LR) per-type accuracy — 5-fold CV
# =====================================================
print("\n" + "=" * 70)
print("2. LEARNED L1 (Logistic Regression, 5-fold CV): per-type accuracy")
print("=" * 70)

np.random.seed(42)
all_indices = np.random.permutation(len(X_all))
fold_size = len(X_all) // 5
folds_list = [all_indices[i*fold_size:(i+1)*fold_size] for i in range(5)]

# For each fold: train on fold i (518), predict on rest, collect predictions
all_preds = np.full(len(X_all), -1)  # -1 = escalated

for i in range(5):
    train_idx = folds_list[i]
    test_idx = np.concatenate([folds_list[j] for j in range(5) if j != i])
    
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_scaled[train_idx], y_all[train_idx])
    proba = lr.predict_proba(X_scaled[test_idx])[:, 1]
    
    # Use thresholds that maximize acc×interception
    preds = np.full(len(test_idx), -1)
    preds[proba >= 0.7] = 1
    preds[proba <= 0.3] = 0
    all_preds[test_idx] = preds

type_stats_learned = defaultdict(lambda: {"correct": 0, "total_decided": 0, "total": 0, "intercepted": 0})
for idx in range(len(records)):
    t = types[idx]
    type_stats_learned[t]["total"] += 1
    pred = all_preds[idx]
    if pred != -1:
        type_stats_learned[t]["intercepted"] += 1
        if pred == y_all[idx]:
            type_stats_learned[t]["correct"] += 1

print(f"\n{'Type':<15} {'Intercepted':>12} {'Correct':>8} {'Accuracy':>10} {'Interc%':>8}")
print("-" * 58)
for t in ["organization","person","location","equipment","event"]:
    s = type_stats_learned[t]
    acc = s["correct"]/s["intercepted"]*100 if s["intercepted"]>0 else 0
    interc = s["intercepted"]/s["total"]*100 if s["total"]>0 else 0
    print(f"{t:<15} {s['intercepted']:>12} {s['correct']:>8} {acc:>9.1f}% {interc:>7.1f}%")
total_int = sum(s["intercepted"] for s in type_stats_learned.values())
total_cor = sum(s["correct"] for s in type_stats_learned.values())
total_all = sum(s["total"] for s in type_stats_learned.values())
print(f"{'TOTAL':<15} {total_int:>12} {total_cor:>8} {total_cor/total_int*100:>9.1f}% {total_int/total_all*100:>7.1f}%")

# =====================================================
# 3. L2 per-type accuracy (from existing results)
# =====================================================
print("\n" + "=" * 70)
print("3. L2 MODELS: per-type accuracy on ESCALATED pairs")
print("=" * 70)

# Check what's in l2_data
print(f"\nL2 data keys: {list(l2_data.keys())[:10]}")
if isinstance(l2_data, dict):
    for k in l2_data:
        v = l2_data[k]
        if isinstance(v, dict):
            print(f"  {k}: {list(v.keys())[:5]}")
        elif isinstance(v, list):
            print(f"  {k}: list[{len(v)}]")
