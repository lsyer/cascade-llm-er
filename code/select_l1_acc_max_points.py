#!/usr/bin/env python3
"""Select L1-accuracy-maximizing thresholds (interception >= 10%) for Fixed/LR/RF.

For LR/RF: 5-fold CV, per-fold select (merge_t, reject_t) on training fold
           that maximizes L1 accuracy subject to that fold's interception >= 10%.
           Apply to test fold -> OOF predictions. Aggregate.
For Fixed: full-data score sweep (no training), select L1-acc-max with intercept >= 10%.

Compute empirical pipeline accuracy using REAL L2 verdicts from checkpoint.
"""
import json, os, re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

BASE = os.path.join(os.path.dirname(__file__), '..')
CKPT = os.path.join(BASE, 'results', 'checkpoints_v3', 'glm-5-turbo_checkpoint.json')
SCORED = os.path.join(BASE, 'results', 'scored_pairs.json')
FIXED_SWEEP = os.path.join(BASE, 'results', 'dual_threshold_sweep.json')
OUT = os.path.join(BASE, 'results', 'l1_acc_max_operating_points.json')

MIN_INTERCEPT = 10.0  # per-fold / per-config interception floor

with open(SCORED) as f:
    scored = json.load(f)
pair_ids = [p['pair_id'] for p in scored]
y = np.array([p['label'] for p in scored])
fixed_scores = np.array([p['l1_score_v2'] for p in scored])
N = len(y)

# Real L2 verdicts
def parse_verdict(v):
    if isinstance(v, dict):
        v = v.get('verdict', v.get('raw', ''))
    v = str(v).strip().lower()
    if re.search(r'\bsame\b|\bmatch\b|\bidentical\b|yes|是', v):
        return 1
    if re.search(r'\bdifferent\b|\bnot.*match\b|\bdistinct\b|no|否|不同', v):
        return 0
    return None

with open(CKPT) as f:
    ckpt = json.load(f)
l2_preds = np.full(N, -1)
for i, pid in enumerate(pair_ids):
    entry = ckpt.get(pid)
    if entry:
        l2_preds[i] = parse_verdict(entry.get('verdict', entry.get('raw', '')))
l2_correct_arr = np.where(l2_preds != -1, (l2_preds == y).astype(int), 0)
pure_l2_acc = l2_correct_arr.sum() / N * 100

FEATURE_NAMES = json.load(open(os.path.join(BASE, 'results', 'classifier_ablation.json')))['feature_names']
X = np.array([[p['features_v2'].get(f, 0) for f in FEATURE_NAMES] for p in scored])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- Per-fold threshold selection: maximize L1 accuracy, intercept >= MIN ----
def select_l1_acc_max(proba_train, y_train, grid_m, grid_r):
    best = None
    best_l1_acc = -1
    n_train = len(y_train)
    for mt in grid_m:
        for rt in grid_r:
            if rt >= mt:
                continue
            decided = (proba_train >= mt) | (proba_train <= rt)
            n_dec = decided.sum()
            if n_dec < 5:
                continue
            intercept = n_dec / n_train * 100
            if intercept < MIN_INTERCEPT:
                continue
            preds = np.where(proba_train >= mt, 1, np.where(proba_train <= rt, 0, -1))
            l1_acc = (preds[decided] == y_train[decided]).sum() / n_dec * 100
            if l1_acc > best_l1_acc:
                best_l1_acc = l1_acc
                best = (mt, rt)
    return best

def apply_thresh(proba, mt, rt):
    decided = (proba >= mt) | (proba <= rt)
    preds = np.where(proba >= mt, 1, np.where(proba <= rt, 0, -1))
    return preds, decided

grid_m = [round(x, 2) for x in np.arange(0.50, 1.001, 0.05)]
grid_r = [round(x, 2) for x in np.arange(0.00, 0.501, 0.05)]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}
for clf_name, clf_factory in [('LR', lambda: LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
                               ('RF', lambda: RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))]:
    oof_preds = np.full(N, -1)
    oof_decided = np.zeros(N, dtype=bool)
    chosen = []
    for fold, (tr, te) in enumerate(skf.split(X_scaled, y)):
        clf = clf_factory()
        clf.fit(X_scaled[tr], y[tr])
        proba_tr = clf.predict_proba(X_scaled[tr])[:, 1]
        proba_te = clf.predict_proba(X_scaled[te])[:, 1]
        mt, rt = select_l1_acc_max(proba_tr, y[tr], grid_m, grid_r)
        preds_te, dec_te = apply_thresh(proba_te, mt, rt)
        oof_preds[te[dec_te]] = preds_te[dec_te]
        oof_decided[te] = dec_te
        chosen.append({'fold': fold, 'merge_t': mt, 'reject_t': rt})

    n_dec = oof_decided.sum()
    l1_correct = (oof_preds[oof_decided] == y[oof_decided]).sum()
    l1_acc = l1_correct / n_dec * 100
    intercept = n_dec / N * 100
    escalated = ~oof_decided
    l2_correct_esc = l2_correct_arr[escalated].sum()
    pipeline = (l1_correct + l2_correct_esc) / N * 100
    results[clf_name] = {
        'l1_accuracy': round(l1_acc, 1),
        'interception': round(intercept, 1),
        'pipeline_empirical': round(pipeline, 2),
        'llm_cost': round(100 - intercept, 1),
        'l1_correct': int(l1_correct), 'n_decided': int(n_dec),
        'per_fold_thresholds': chosen,
    }
    print(f"{clf_name}: l1_acc={l1_acc:.1f}%, intercept={intercept:.1f}%, pipeline={pipeline:.2f}%, cost={100-intercept:.1f}%")
    for c in chosen:
        print(f"    fold {c['fold']}: mt={c['merge_t']}, rt={c['reject_t']}")

# ---- Fixed L1: full-data score sweep, L1-acc-max with intercept>=10% ----
with open(FIXED_SWEEP) as f:
    fixed_sweep = json.load(f)
best_fixed = None
best_l1 = -1
for r in fixed_sweep:
    if r['interception'] < MIN_INTERCEPT:
        continue
    if r['l1_accuracy'] > best_l1:
        best_l1 = r['l1_accuracy']
        best_fixed = r

# Empirical pipeline for Fixed L1 at that point (use real L2 on escalated)
mt_s = best_fixed['merge_threshold']
rt_s = best_fixed['reject_threshold']
decided_mask = (fixed_scores >= mt_s) | (fixed_scores <= rt_s)
fixed_l1_preds = np.where(fixed_scores >= mt_s, 1, np.where(fixed_scores <= rt_s, 0, -1))
l1_correct_fix = (fixed_l1_preds[decided_mask] == y[decided_mask]).sum()
l1_acc_fix = l1_correct_fix / decided_mask.sum() * 100
esc_mask = ~decided_mask
l2_correct_esc_fix = l2_correct_arr[esc_mask].sum()
pipeline_fix = (l1_correct_fix + l2_correct_esc_fix) / N * 100
intercept_fix = decided_mask.sum() / N * 100
results['Fixed'] = {
    'l1_accuracy': round(l1_acc_fix, 1),
    'interception': round(intercept_fix, 1),
    'pipeline_empirical': round(pipeline_fix, 2),
    'llm_cost': round(100 - intercept_fix, 1),
    'merge_t': mt_s, 'reject_t': rt_s,
}
print(f"\nFixed: mt={mt_s}, rt={rt_s}, l1_acc={l1_acc_fix:.1f}%, intercept={intercept_fix:.1f}%, pipeline={pipeline_fix:.2f}%")

# Also the hand-tuned point for reference
dec_ht = (fixed_scores >= 0.6) | (fixed_scores <= -0.4)
preds_ht = np.where(fixed_scores >= 0.6, 1, np.where(fixed_scores <= -0.4, 0, -1))
l1c_ht = (preds_ht[dec_ht] == y[dec_ht]).sum()
l2c_ht = l2_correct_arr[~dec_ht].sum()
pipe_ht = (l1c_ht + l2c_ht) / N * 100
results['Fixed_handtuned'] = {
    'l1_accuracy': round(l1c_ht / dec_ht.sum() * 100, 1),
    'interception': round(dec_ht.sum() / N * 100, 1),
    'pipeline_empirical': round(pipe_ht, 2),
    'llm_cost': round(100 - dec_ht.sum() / N * 100, 1),
    'merge_t': 0.6, 'reject_t': -0.4,
}
print(f"Fixed (hand-tuned 0.6/-0.4): l1_acc={l1c_ht/dec_ht.sum()*100:.1f}%, intercept={dec_ht.sum()/N*100:.1f}%, pipeline={pipe_ht:.2f}%")

print(f"\nPure L2 ceiling: {pure_l2_acc:.2f}%")

output = {
    'min_interception': MIN_INTERCEPT,
    'pure_l2_ceiling': round(pure_l2_acc, 2),
    'selection_rule': 'L1 accuracy maximized on training fold, subject to per-fold interception >= 10%',
    **results,
}
with open(OUT, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {OUT}")
