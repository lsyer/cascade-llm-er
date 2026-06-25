#!/usr/bin/env python3
"""
Experiment 22: Per-type threshold + per-type model analysis

Three questions:
1. L1 threshold vs accuracy tradeoff (by type)
2. Per-type thresholds: better tradeoff than unified?
3. Per-type models (separate LR per type): better than unified LR?

Protocol: 5-fold CV (same as paper).
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
                    "type": p.get("type_a","unknown"),
                    "l1_score": p.get("l1_score", 0)})

X_all = np.array([r["x"] for r in records])
y_all = np.array([r["y"] for r in records])
types = np.array([r["type"] for r in records])
l1_scores = np.array([r["l1_score"] for r in records])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

TYPE_ORDER = ["organization", "person", "location", "equipment", "event"]

def eval_decisions(y_true, decisions):
    """decisions: -1=escalate, 0=reject, 1=merge. Returns dict."""
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

# ═══════════════════════════════════════════════════════════
# Q1: FIXED-WEIGHT L1 THRESHOLD SWEEP (per type)
# ═══════════════════════════════════════════════════════════
print("=" * 80)
print("Q1: FIXED-WEIGHT L1 — Threshold vs Accuracy tradeoff (per type)")
print("=" * 80)

MERGE_THS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
REJECT_THS = [-0.1, -0.2, -0.3, -0.4, -0.5]

print(f"\n{'Type':<13}", end="")
for mt in MERGE_THS:
    print(f" m≥{mt:.1f}", end="")
print()

for t in TYPE_ORDER:
    mask = types == t
    scores_t = l1_scores[mask]
    y_t = y_all[mask]
    print(f"{t:<13}", end="")
    for mt in MERGE_THS:
        # Use reject_th = -0.3 fixed, vary merge_th
        decisions = np.full(len(y_t), -1)
        decisions[scores_t >= mt] = 1
        decisions[scores_t <= -0.3] = 0
        r = eval_decisions(y_t, decisions)
        if r["n_decided"] > 0:
            print(f" {r['acc']:.0f}/{r['interc']:.0f}", end="")
        else:
            print(f"  -  ", end="")
    print(f"  (n={mask.sum()})")

print(f"\nFormat: accuracy%/interception%")
print(f"(reject threshold fixed at -0.3)")

# ═══════════════════════════════════════════════════════════
# Q2: UNIFIED vs PER-TYPE THRESHOLDS (on fixed-weight L1)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("Q2: UNIFIED vs PER-TYPE THRESHOLDS (fixed-weight L1)")
print("=" * 80)

# Unified: merge≥0.5, reject≤-0.3 (paper baseline)
decisions_unified = np.full(len(y_all), -1)
decisions_unified[l1_scores >= 0.5] = 1
decisions_unified[l1_scores <= -0.3] = 0
r_unified = eval_decisions(y_all, decisions_unified)
print(f"\nUnified (merge≥0.5, reject≤-0.3):  Acc={r_unified['acc']:.1f}%, Interc={r_unified['interc']:.1f}%")

# Per-type: for each type, find the threshold pair that maximizes acc × interc
best_per_type_decisions = np.full(len(y_all), -1)
print("\nPer-type optimal thresholds (maximize acc × interc):")
print(f"{'Type':<13} {'Merge≥':>7} {'Reject≤':>7} {'Acc':>7} {'Interc':>7} {'n':>5}")
print("-" * 50)

for t in TYPE_ORDER:
    mask = types == t
    scores_t = l1_scores[mask]
    y_t = y_all[mask]
    
    best_score = 0
    best = None
    for mt in np.arange(0.3, 1.01, 0.05):
        for rt in np.arange(-0.5, 0.01, 0.05):
            decisions = np.full(len(y_t), -1)
            decisions[scores_t >= mt] = 1
            decisions[scores_t <= rt] = 0
            r = eval_decisions(y_t, decisions)
            if r["n_decided"] < 5: continue
            score = r["acc"] * r["interc"]
            if score > best_score:
                best_score = score
                best = (mt, rt, r)
    
    if best:
        mt, rt, r = best
        print(f"{t:<13} {mt:>6.2f} {rt:>7.2f} {r['acc']:>6.1f}% {r['interc']:>6.1f}% {r['n_decided']:>5}")
        # Apply to global decisions
        for idx in np.where(mask)[0]:
            if l1_scores[idx] >= mt:
                best_per_type_decisions[idx] = 1
            elif l1_scores[idx] <= rt:
                best_per_type_decisions[idx] = 0

r_per_type = eval_decisions(y_all, best_per_type_decisions)
print(f"\nPer-type thresholds combined:     Acc={r_per_type['acc']:.1f}%, Interc={r_per_type['interc']:.1f}%")
print(f"Unified baseline:                 Acc={r_unified['acc']:.1f}%, Interc={r_unified['interc']:.1f}%")
print(f"Delta:                            Acc={r_per_type['acc']-r_unified['acc']:+.1f}pp, Interc={r_per_type['interc']-r_unified['interc']:+.1f}pp")

# ═══════════════════════════════════════════════════════════
# Q3: UNIFIED LR vs PER-TYPE LR (5-fold CV)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("Q3: UNIFIED LR vs PER-TYPE LR (5-fold CV)")
print("=" * 80)

np.random.seed(42)
all_indices = np.random.permutation(len(X_all))
fold_size = len(X_all) // 5
folds_list = [all_indices[i*fold_size:(i+1)*fold_size] for i in range(5)]

# --- Unified LR ---
all_preds_unified = np.full(len(X_all), -1)
for i in range(5):
    train_idx = folds_list[i]
    test_idx = np.concatenate([folds_list[j] for j in range(5) if j != i])
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_scaled[train_idx], y_all[train_idx])
    proba = lr.predict_proba(X_scaled[test_idx])[:, 1]
    preds = np.full(len(test_idx), -1)
    preds[proba >= 0.7] = 1
    preds[proba <= 0.3] = 0
    all_preds_unified[test_idx] = preds

r_unified_lr = eval_decisions(y_all, all_preds_unified)
print(f"\nUnified LR (merge≥0.7, reject≤0.3):")
print(f"  Overall: Acc={r_unified_lr['acc']:.1f}%, Interc={r_unified_lr['interc']:.1f}%")
for t in TYPE_ORDER:
    mask = types == t
    r = eval_decisions(y_all[mask], all_preds_unified[mask])
    print(f"  {t:<13}: Acc={r['acc']:.1f}%, Interc={r['interc']:.1f}%")

# --- Per-type LR ---
all_preds_per_type = np.full(len(X_all), -1)
for t in TYPE_ORDER:
    mask = types == t
    idx_t = np.where(mask)[0]
    X_t = X_scaled[mask]
    y_t = y_all[mask]
    
    if len(y_t) < 100:
        # Too few samples, skip per-type model
        print(f"\n{t}: skipped (only {len(y_t)} pairs)")
        continue
    
    # Per-type 5-fold CV
    idx_arr = np.random.permutation(len(y_t))
    ft = len(y_t) // 5
    
    for i in range(5):
        train_local = np.concatenate([idx_arr[j*ft:(j+1)*ft] for j in range(5) if j != i])
        test_local = idx_arr[i*ft:(i+1)*ft]
        
        if len(train_local) < 20:
            continue
        
        lr_t = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        lr_t.fit(X_t[train_local], y_t[train_local])
        proba = lr_t.predict_proba(X_t[test_local])[:, 1]
        
        # Per-type threshold scan
        best_f1 = 0
        best_mt, best_rt = 0.7, 0.3
        for mt in np.arange(0.55, 0.95, 0.05):
            for rt in np.arange(0.1, 0.45, 0.05):
                preds = np.full(len(test_local), -1)
                preds[proba >= mt] = 1
                preds[proba <= rt] = 0
                r = eval_decisions(y_t[test_local], preds)
                if r["n_decided"] < 3: continue
                if r["acc"] * r["interc"] > best_f1:
                    best_f1 = r["acc"] * r["interc"]
                    best_mt, best_rt = mt, rt
        
        preds = np.full(len(test_local), -1)
        preds[proba >= best_mt] = 1
        preds[proba <= best_rt] = 0
        
        global_idx = idx_t[test_local]
        all_preds_per_type[global_idx] = preds

print(f"\nPer-type LR (each type trained + thresholded separately):")
r_pt = eval_decisions(y_all, all_preds_per_type)
print(f"  Overall: Acc={r_pt['acc']:.1f}%, Interc={r_pt['interc']:.1f}%")
for t in TYPE_ORDER:
    mask = types == t
    r = eval_decisions(y_all[mask], all_preds_per_type[mask])
    print(f"  {t:<13}: Acc={r['acc']:.1f}%, Interc={r['interc']:.1f}%")

# Summary comparison
print("\n" + "=" * 80)
print("SUMMARY COMPARISON")
print("=" * 80)
print(f"\n{'Method':<40} {'Acc':>7} {'Interc':>7} {'Esc%':>7}")
print("-" * 65)
print(f"{'Fixed L1 unified (paper baseline)':<40} {65.4:>6.1f}% {39.4:>6.1f}% {60.6:>6.1f}%")
print(f"{'Fixed L1 per-type thresholds':<40} {r_per_type['acc']:>6.1f}% {r_per_type['interc']:>6.1f}% {100-r_per_type['interc']:>6.1f}%")
print(f"{'Learned L1 unified LR (5-fold CV)':<40} {r_unified_lr['acc']:>6.1f}% {r_unified_lr['interc']:>6.1f}% {100-r_unified_lr['interc']:>6.1f}%")
print(f"{'Learned L1 per-type LR (5-fold CV)':<40} {r_pt['acc']:>6.1f}% {r_pt['interc']:>6.1f}% {100-r_pt['interc']:>6.1f}%")
