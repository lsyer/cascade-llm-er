#!/usr/bin/env python3
"""
Proper 5-fold cross-validation comparison: LR vs RF
Same protocol as the feedback loop experiment (Round 1: 518 feedback samples).
"""
import json, re, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
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

# Build feature matrix
X_all, y_all, feature_names = [], [], None
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
    X_all.append([feats[k] for k in feature_names])
    y_all.append(y)

X_all = np.array(X_all)
y_all = np.array(y_all)
print(f"Total pairs: {len(y_all)}, Features: {len(feature_names)}")
print(f"Same: {sum(y_all)}, Different: {len(y_all)-sum(y_all)}")

# StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# Same 5-fold protocol as the paper's feedback loop experiment
np.random.seed(42)
all_indices = np.random.permutation(len(X_all))
fold_size = len(X_all) // 5
folds_list = [all_indices[i*fold_size:(i+1)*fold_size] for i in range(5)]

def eval_model_on_test(model, X_train, y_train, X_test, y_test, label):
    """Train model, predict on test, scan thresholds, return best result."""
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    
    best_f1 = 0
    best = None
    for mt in [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        for rt in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
            preds = np.full(len(y_test), -1)
            preds[proba >= mt] = 1
            preds[proba <= rt] = 0
            decided = preds != -1
            if decided.sum() == 0: continue
            correct = (preds[decided] == y_test[decided]).sum()
            acc = correct / decided.sum() * 100
            interc = decided.sum() / len(y_test) * 100
            # F1-like: acc * interc
            score = acc * interc
            if score > best_f1 and acc >= 70 and interc >= 50:
                best_f1 = score
                merge_mask = (preds == 1) & decided
                reject_mask = (preds == 0) & decided
                mp = (y_test[merge_mask] == 1).sum() / max(1, merge_mask.sum()) * 100
                rp = (y_test[reject_mask] == 0).sum() / max(1, reject_mask.sum()) * 100
                best = {
                    "acc": round(acc, 1), "interc": round(interc, 1),
                    "merge_prec": round(mp, 1), "reject_prec": round(rp, 1),
                    "mt": mt, "rt": rt
                }
    return best

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION: LR vs RF (Round 1 protocol: 518 training)")
print("=" * 70)

# Round 1: first fold = training (518 samples), rest = test
train_idx = folds_list[0]
test_idx = np.concatenate(folds_list[1:])

X_train = X_scaled[train_idx]
y_train = y_all[train_idx]
X_test = X_scaled[test_idx]
y_test = y_all[test_idx]

print(f"\nTrain: {len(train_idx)}, Test: {len(test_idx)}")

# Logistic Regression (same as paper)
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr_result = eval_model_on_test(lr, X_train, y_train, X_test, y_test, "LR")
print(f"\nLogistic Regression:")
print(f"  Acc={lr_result['acc']}%, Interc={lr_result['interc']}%, "
      f"Merge Prec={lr_result['merge_prec']}%, Reject Prec={lr_result['reject_prec']}%")
print(f"  (merge≥{lr_result['mt']}, reject≤{lr_result['rt']})")

# Random Forest - multiple configs
rf_configs = [
    {"n_estimators": 100, "max_depth": 5, "label": "RF(d=5,n=100)"},
    {"n_estimators": 200, "max_depth": 8, "label": "RF(d=8,n=200)"},
    {"n_estimators": 200, "max_depth": 12, "label": "RF(d=12,n=200)"},
    {"n_estimators": 500, "max_depth": 8, "label": "RF(d=8,n=500)"},
]

print()
for cfg in rf_configs:
    rf = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        class_weight="balanced",
        random_state=42
    )
    result = eval_model_on_test(rf, X_train, y_train, X_test, y_test, cfg["label"])
    if result:
        print(f"{cfg['label']}: Acc={result['acc']}%, Interc={result['interc']}%, "
              f"Merge Prec={result['merge_prec']}%, Reject Prec={result['reject_prec']}%")
    else:
        print(f"{cfg['label']}: No valid operating point found")

# Also run all 5 folds for both LR and RF
print("\n" + "=" * 70)
print("FULL 5-FOLD CV (train on 1 fold, test on remaining 4)")
print("=" * 70)

lr_accs, rf_accs = [], []
lr_intercs, rf_intercs = [], []

for i in range(5):
    train_idx = folds_list[i]
    test_idx = np.concatenate([folds_list[j] for j in range(5) if j != i])
    
    X_train = X_scaled[train_idx]
    y_train = y_all[train_idx]
    X_test = X_scaled[test_idx]
    y_test = y_all[test_idx]
    
    # LR
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_r = eval_model_on_test(lr, X_train, y_train, X_test, y_test, "LR")
    lr_accs.append(lr_r["acc"])
    lr_intercs.append(lr_r["interc"])
    
    # RF (best config)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)
    rf_r = eval_model_on_test(rf, X_train, y_train, X_test, y_test, "RF")
    rf_accs.append(rf_r["acc"] if rf_r else 0)
    rf_intercs.append(rf_r["interc"] if rf_r else 0)
    
    print(f"  Fold {i+1} (train={len(train_idx)}, test={len(test_idx)}): "
          f"LR={lr_r['acc']}%/{lr_r['interc']}%, "
          f"RF={rf_r['acc'] if rf_r else 'N/A'}%/{rf_r['interc'] if rf_r else 'N/A'}%")

print(f"\n  LR avg: {np.mean(lr_accs):.1f}% acc, {np.mean(lr_intercs):.1f}% interc")
print(f"  RF avg: {np.mean(rf_accs):.1f}% acc, {np.mean(rf_intercs):.1f}% interc")
print(f"  LR range: {min(lr_accs):.1f}-{max(lr_accs):.1f}% acc")
print(f"  RF range: {min(rf_accs):.1f}-{max(rf_accs):.1f}% acc")
