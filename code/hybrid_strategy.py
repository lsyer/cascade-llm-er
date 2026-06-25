#!/usr/bin/env python3
"""
Experiment 24: Hybrid strategy — per-type LR for large types, unified LR fallback for small types.

Rule: if a type has ≥N training samples in this fold, train per-type LR.
      Otherwise, fall back to unified LR for that type.

Scan N (threshold) from 50 to 500 to find the sweet spot.
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
            score = acc * interc
            if score > best_score:
                best_score = score
                best = (mt, rt)
    return best

def run_strategy(min_samples_for_per_type, verbose=False):
    """
    Hybrid: per-type LR if training samples for that type ≥ min_samples,
    otherwise fall back to unified LR.
    """
    np.random.seed(42)
    all_indices = np.random.permutation(len(X_all))
    fold_size = len(X_all) // 5
    folds_list = [all_indices[i*fold_size:(i+1)*fold_size] for i in range(5)]
    
    all_preds = np.full(len(X_all), -1)
    
    for i in range(5):
        train_idx = folds_list[i]
        test_idx = np.concatenate([folds_list[j] for j in range(5) if j != i])
        
        # Always train unified LR (for fallback)
        lr_unified = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        lr_unified.fit(X_scaled[train_idx], y_all[train_idx])
        proba_train_uni = lr_unified.predict_proba(X_scaled[train_idx])[:, 1]
        uni_mt, uni_rt = select_thresholds(proba_train_uni, y_all[train_idx], min_acc=75)
        
        for t in TYPE_ORDER:
            train_mask_t = types[train_idx] == t
            test_mask_t = types[test_idx] == t
            n_train_t = train_mask_t.sum()
            
            X_test_t = X_scaled[test_idx][test_mask_t]
            global_test_idx = test_idx[test_mask_t]
            
            if n_train_t >= min_samples_for_per_type:
                # Per-type LR
                X_train_t = X_scaled[train_idx][train_mask_t]
                y_train_t = y_all[train_idx][train_mask_t]
                
                lr_t = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
                lr_t.fit(X_train_t, y_train_t)
                
                proba_train_t = lr_t.predict_proba(X_train_t)[:, 1]
                mt, rt = select_thresholds(proba_train_t, y_train_t, min_acc=75)
                
                proba_test_t = lr_t.predict_proba(X_test_t)[:, 1]
                source = "per-type"
            else:
                # Unified LR fallback
                proba_test_t = lr_unified.predict_proba(X_test_t)[:, 1]
                mt, rt = uni_mt, uni_rt
                source = "unified"
            
            test_preds = np.full(len(X_test_t), -1)
            test_preds[proba_test_t >= mt] = 1
            test_preds[proba_test_t <= rt] = 0
            all_preds[global_test_idx] = test_preds
            
            if verbose:
                print(f"  Fold {i} {t:<13}: {source} (n_train={n_train_t}) th=({mt:.2f},{rt:.2f})")
    
    # Compute metrics
    decided = all_preds != -1
    correct = (all_preds[decided] == y_all[decided]).sum()
    n_decided = decided.sum()
    acc = correct / n_decided * 100
    interc = n_decided / len(y_all) * 100
    pipeline = interc/100 * acc + (1-interc/100) * 97.1
    
    # Per-type breakdown
    per_type = {}
    for t in TYPE_ORDER:
        mask = types == t
        d = mask & (all_preds != -1)
        c = (all_preds[d] == y_all[d]).sum()
        nd = d.sum()
        per_type[t] = {
            "acc": c / nd * 100 if nd > 0 else 0,
            "interc": nd / mask.sum() * 100,
            "n_decided": int(nd),
            "n_total": int(mask.sum()),
        }
    
    return {
        "acc": acc, "interc": interc, "pipeline": pipeline,
        "per_type": per_type, "min_samples": min_samples_for_per_type
    }

# ═══════════════════════════════════════════════════════════════
# Scan min_samples threshold
# ═══════════════════════════════════════════════════════════════
print("=" * 90)
print("HYBRID STRATEGY SCAN: minimum training samples for per-type LR")
print("(below threshold → unified LR fallback)")
print("=" * 90)

# First show type counts per fold
print("\nType distribution (per fold, train side):")
np.random.seed(42)
all_indices = np.random.permutation(len(X_all))
fold_size = len(X_all) // 5
folds_list = [all_indices[i*fold_size:(i+1)*fold_size] for i in range(5)]
print(f"{'Type':<13}", end="")
for i in range(5):
    print(f"  F{i+1}_train", end="")
print(f"  {'Total':>6}")
for t in TYPE_ORDER:
    print(f"{t:<13}", end="")
    for i in range(5):
        train_idx = folds_list[i]
        n = (types[train_idx] == t).sum()
        print(f"  {n:>8}", end="")
    print(f"  {(types == t).sum():>6}")

# Run scan
print(f"\n{'Min_Samples':<12} {'Strategy':<35} {'L1_Acc':>7} {'Interc':>7} {'Pipeline':>8} {'L2_Calls':>8}")
print("-" * 85)

results_scan = []
for min_n in [0, 50, 100, 150, 200, 250, 300, 400, 500, 9999]:
    r = run_strategy(min_n)
    
    if min_n == 0:
        label = "All per-type"
    elif min_n == 9999:
        label = "All unified (paper baseline)"
    else:
        # Which types go per-type?
        big_types = [t for t in TYPE_ORDER if (types == t).sum() // 5 * 4 >= min_n]
        small_types = [t for t in TYPE_ORDER if t not in big_types]
        label = f"per-type: {','.join(big_types[:3])}"
        if small_types:
            label += f" | unified: {','.join(small_types[:2])}"
    
    print(f"{min_n:<12} {label:<35} {r['acc']:>6.1f}% {r['interc']:>6.1f}% {r['pipeline']:>7.1f}% {100-r['interc']:>7.1f}%")
    results_scan.append(r)

# Detailed breakdown for best result
best = max(results_scan, key=lambda x: x["pipeline"])
print(f"\n{'='*90}")
print(f"BEST RESULT: min_samples={best['min_samples']} → Pipeline={best['pipeline']:.1f}%")
print(f"{'='*90}")

print(f"\n{'Type':<13} {'Acc':>7} {'Interc':>7} {'n_decided':>10} {'n_total':>8}")
print("-" * 50)
for t in TYPE_ORDER:
    pt = best["per_type"][t]
    print(f"{t:<13} {pt['acc']:>6.1f}% {pt['interc']:>6.1f}% {pt['n_decided']:>10} {pt['n_total']:>8}")
print("-" * 50)
print(f"{'OVERALL':<13} {best['acc']:>6.1f}% {best['interc']:>6.1f}% {'':>10} {len(y_all):>8}")
print(f"{'Pipeline':<13} {best['pipeline']:>6.1f}%")

# Compare all three strategies
print(f"\n{'='*90}")
print(f"FINAL COMPARISON")
print(f"{'='*90}")
print(f"\n{'Strategy':<40} {'L1_Acc':>7} {'Interc':>7} {'Pipeline':>8} {'L2%':>6}")
print("-" * 72)

baseline = run_strategy(9999)
all_per_type = run_strategy(0)

for label, r in [
    ("Fixed L1 (paper baseline)", None),
    ("Learned unified LR", baseline),
    ("Learned per-type LR (all types)", all_per_type),
    (f"Learned hybrid (best: min={best['min_samples']})", best),
]:
    if r is None:
        print(f"{label:<40} {'65.8':>6}% {'39.6':>6}% {'~76':>7}% {'60.4':>5}%")
    else:
        print(f"{label:<40} {r['acc']:>6.1f}% {r['interc']:>6.1f}% {r['pipeline']:>7.1f}% {100-r['interc']:>5.1f}%")
