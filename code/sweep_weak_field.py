#!/usr/bin/env python3
"""
sweep_weak_field.py — Systematic sweep of weak-field comparison parameters.

Tests combinations of:
  - match_threshold: 0.3, 0.4, 0.5, 0.6, 0.7
  - low_overlap_action: 'unknown' (no penalty) vs 'conflict' (penalize)
  - weak_weight_factor: 0.5 (half weight) vs 1.0 (equal weight)

For each config: re-score 2639 pairs → run full pipeline → record metrics.
"""
import json, os, sys, copy, itertools
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

DATA = "/root/workspace/tanshu_docs/experiments/release/data"
CKPT = "/root/workspace/tanshu_docs/experiments/release/results/checkpoints_v3/glm-5.2_checkpoint.json"
OUT = "/root/workspace/tanshu_docs/experiments/release/results"

# ============================================================
# Load data once
# ============================================================
with open(f"{DATA}/dataset_v3_cleaned.json") as f:
    data = json.load(f)
pairs = data['same_type_pairs']

with open(CKPT) as f:
    checkpoint = json.load(f)

for p in pairs:
    key = p['pair_id']
    if key in checkpoint:
        verdict = checkpoint[key].get('verdict', '')
        p['label'] = 1 if verdict == 'same' else 0
    else:
        p['label'] = None

labeled = [p for p in pairs if p['label'] is not None]

# Pre-parse props
for p in labeled:
    for k in ('props_a', 'props_b'):
        if isinstance(p.get(k), str):
            p[k] = json.loads(p[k])
    p['etype'] = p['type_a'].replace('activity', 'event')

L2_ACC = 0.971
EXCLUDE_FIELDS = {'created_at', 'updated_at', 'confidence', 'labels', 'vid', 'id', 'source_pk'}
STRONG_FIELDS = {
    'equip_type', 'category', 'state', 'loc_type', 'event_type', 'org_type',
    'region', 'coordinates', 'start_date', 'occurred_at', 'end_date',
    'home_location', 'latest_reported_at', 'gender', 'nationality', 'industry',
}

def _get(props, key):
    v = props.get(key, '') or ''
    return str(v).strip().lower()

def _has_value(props, key):
    v = props.get(key, '')
    return bool(v and str(v).strip())

def _name_similarity(a, b):
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)

def _is_containment(a, b):
    a = a.lower().strip()
    b = b.lower().strip()
    if not a or not b:
        return False
    return a in b or b in a

def _get_effective_fields(props_a, props_b):
    all_fields = set()
    for k in list(props_a.keys()) + list(props_b.keys()):
        if k in EXCLUDE_FIELDS:
            continue
        all_fields.add(k)
    effective = []
    for f in sorted(all_fields):
        if _has_value(props_a, f) or _has_value(props_b, f):
            effective.append(f)
    return effective

def _has_hard_conflict(props_a, props_b, etype):
    if etype == 'person':
        ga, gb = _get(props_a, 'gender'), _get(props_b, 'gender')
        if ga and gb and ga != gb:
            return True
    if etype == 'equipment':
        ca, cb = _get(props_a, 'equip_type'), _get(props_b, 'equip_type')
        if ca and cb and ca != cb:
            aircraft_kw = ['aircraft', 'plane', 'helicopter', 'fighter', 'bomber', 'drone']
            ship_kw = ['ship', 'carrier', 'destroyer', 'frigate', 'submarine', 'vessel']
            ca_ac = any(k in ca for k in aircraft_kw)
            cb_ac = any(k in cb for k in aircraft_kw)
            ca_sh = any(k in ca for k in ship_kw)
            cb_sh = any(k in cb for k in ship_kw)
            if (ca_ac and cb_sh) or (ca_sh and cb_ac):
                return True
    return False

# ============================================================
# Parameterized comparison + scoring
# ============================================================

def compare_field(field, va, vb, match_threshold, low_overlap_action):
    """Compare a single field. Returns 'match'|'conflict'|'unknown'."""
    if not va and not vb:
        return 'unknown'
    if not va or not vb:
        return 'unknown'
    if va == vb:
        return 'match'
    if _is_containment(va, vb):
        return 'match'
    
    is_strong = field in STRONG_FIELDS
    if is_strong:
        return 'conflict'
    
    # Weak field — use Jaccard
    sim = _name_similarity(va, vb)
    if sim >= match_threshold:
        return 'match'
    if low_overlap_action == 'conflict':
        return 'conflict'
    return 'unknown'

def score_pair(props_a, props_b, etype, name_a, name_b,
               match_threshold, low_overlap_action, weak_weight_factor,
               penalty_factor=1.5):
    """Score a pair with given parameters. Returns (score, decision)."""
    if _has_hard_conflict(props_a, props_b, etype):
        return -1.0, 'reject'
    
    fields = _get_effective_fields(props_a, props_b)
    if not fields:
        return 0.0, 'escalate'
    
    n_strong = sum(1 for f in fields if f in STRONG_FIELDS)
    n_weak = len(fields) - n_strong
    effective_n = n_strong + weak_weight_factor * n_weak
    if effective_n <= 0:
        return 0.0, 'escalate'
    
    w_strong = 1.0 / effective_n
    w_weak = weak_weight_factor / effective_n
    
    score = 0.0
    for f in fields:
        va = _get(props_a, f)
        vb = _get(props_b, f)
        result = compare_field(f, va, vb, match_threshold, low_overlap_action)
        w = w_strong if f in STRONG_FIELDS else w_weak
        if result == 'match':
            score += w
        elif result == 'conflict':
            score -= w * penalty_factor
    
    score = max(-1.0, min(1.0, score))
    
    # Name bonus
    if name_a and name_b:
        na, nb = name_a.lower().strip(), name_b.lower().strip()
        if na == nb:
            score = min(1.0, score + 0.3)
        elif _is_containment(na, nb):
            score = min(1.0, score + 0.2)
        else:
            sim = _name_similarity(na, nb)
            if sim >= 0.5:
                score = min(1.0, score + 0.15)
    
    score = max(-1.0, min(1.0, score))
    
    if score >= 0.5:
        return score, 'merge'
    elif score <= -0.3:
        return score, 'reject'
    return score, 'escalate'

def extract_features_v2(props_a, props_b, etype, name_a, name_b,
                        match_threshold, low_overlap_action):
    """Extract features for LR training with given params."""
    fields = _get_effective_fields(props_a, props_b)
    features = {}
    
    if not fields:
        features['property_overlap_rate'] = 0.0
        features['property_conflict_rate'] = 0.0
        features['n_effective_fields'] = 0
        features['name_similarity'] = 0.0
        features['name_exact'] = 0
        features['name_containment'] = 0
        return features
    
    matches = conflicts = 0
    for f in fields:
        va = _get(props_a, f)
        vb = _get(props_b, f)
        result = compare_field(f, va, vb, match_threshold, low_overlap_action)
        if result == 'match':
            matches += 1
        elif result == 'conflict':
            conflicts += 1
        features[f'fld_{f}_match'] = 1 if result == 'match' else 0
        features[f'fld_{f}_conflict'] = 1 if result == 'conflict' else 0
    
    n = len(fields)
    features['property_overlap_rate'] = matches / n
    features['property_conflict_rate'] = conflicts / n
    features['n_effective_fields'] = n
    features['name_similarity'] = _name_similarity(name_a or _get(props_a, 'name'),
                                                    name_b or _get(props_b, 'name'))
    features['name_exact'] = 1 if name_a and name_b and name_a.lower().strip() == name_b.lower().strip() else 0
    features['name_containment'] = 1 if _is_containment(name_a or '', name_b or '') else 0
    return features

def select_thresholds(proba, y_true, min_acc=75):
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
# Sweep configs
# ============================================================
configs = list(itertools.product(
    [0.3, 0.4, 0.5, 0.6, 0.7],           # match_threshold
    ['unknown', 'conflict'],               # low_overlap_action
    [0.5, 1.0],                            # weak_weight_factor
))

print(f"Sweeping {len(configs)} configurations...\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for ci, (mt, loa, wwf) in enumerate(configs):
    config_name = f"mt={mt}_loa={loa}_wwf={wwf}"
    
    # Re-score all pairs
    y = np.array([p['label'] for p in labeled])
    types = np.array([p['etype'] for p in labeled])
    
    decisions = []
    for p in labeled:
        score, dec = score_pair(p['props_a'], p['props_b'], p['etype'],
                                p['name_a'], p['name_b'],
                                mt, loa, wwf)
        decisions.append(dec)
    
    # Level 0: Fixed weights
    intercepted = [i for i, d in enumerate(decisions) if d in ('merge', 'reject')]
    correct_l0 = sum(1 for i in intercepted if (
        (decisions[i] == 'merge' and y[i] == 1) or
        (decisions[i] == 'reject' and y[i] == 0)
    ))
    l0_acc = correct_l0 / len(intercepted) * 100 if intercepted else 0
    l0_intercept = len(intercepted) / len(y) * 100
    
    # Extract features
    all_features = []
    for p in labeled:
        feats = extract_features_v2(p['props_a'], p['props_b'], p['etype'],
                                     p['name_a'], p['name_b'],
                                     mt, loa)
        all_features.append(feats)
    
    feature_names = sorted(set(k for f in all_features for k in f))
    X = np.array([[f.get(fn, 0) for fn in feature_names] for f in all_features])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    # Level 1: Unified LR with tuned thresholds
    unified_preds = np.full(len(y), -1)
    for train_idx, test_idx in skf.split(X_s, y):
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        lr.fit(X_s[train_idx], y[train_idx])
        proba_tr = lr.predict_proba(X_s[train_idx])[:, 1]
        m_t, r_t = select_thresholds(proba_tr, y[train_idx])
        proba = lr.predict_proba(X_s[test_idx])[:, 1]
        preds = np.full(len(test_idx), -1)
        preds[proba >= m_t] = 1
        preds[proba <= r_t] = 0
        unified_preds[test_idx] = preds
    
    u_mask = unified_preds != -1
    u_correct = sum(1 for i in range(len(y)) if u_mask[i] and unified_preds[i] == y[i])
    l1_acc = u_correct / u_mask.sum() * 100 if u_mask.sum() > 0 else 0
    l1_intercept = u_mask.sum() / len(y) * 100
    
    # Level 2: Hybrid per-type
    TYPE_ORDER = sorted(set(types))
    hybrid_preds = np.full(len(y), -1)
    
    for train_idx, test_idx in skf.split(X_s, y):
        lr_u = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        lr_u.fit(X_s[train_idx], y[train_idx])
        proba_tr_u = lr_u.predict_proba(X_s[train_idx])[:, 1]
        u_mt, u_rt = select_thresholds(proba_tr_u, y[train_idx])
        
        for t in TYPE_ORDER:
            tr_mask = types[train_idx] == t
            te_mask = types[test_idx] == t
            n_tr = tr_mask.sum()
            X_te_t = X_s[test_idx][te_mask]
            g_te = test_idx[te_mask]
            
            if n_tr >= 50:
                lr_t = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
                lr_t.fit(X_s[train_idx][tr_mask], y[train_idx][tr_mask])
                proba_tr = lr_t.predict_proba(X_s[train_idx][tr_mask])[:, 1]
                mt_t, rt_t = select_thresholds(proba_tr, y[train_idx][tr_mask])
                proba = lr_t.predict_proba(X_te_t)[:, 1]
            else:
                proba = lr_u.predict_proba(X_te_t)[:, 1]
                mt_t, rt_t = u_mt, u_rt
            
            preds = np.full(len(X_te_t), -1)
            preds[proba >= mt_t] = 1
            preds[proba <= rt_t] = 0
            hybrid_preds[g_te] = preds
    
    h_mask = hybrid_preds != -1
    h_correct = sum(1 for i in range(len(y)) if h_mask[i] and hybrid_preds[i] == y[i])
    l2_acc = h_correct / h_mask.sum() * 100 if h_mask.sum() > 0 else 0
    l2_intercept = h_mask.sum() / len(y) * 100
    
    # Pipeline
    p0 = l0_intercept/100 * l0_acc/100 + (1 - l0_intercept/100) * L2_ACC
    p1 = l1_intercept/100 * l1_acc/100 + (1 - l1_intercept/100) * L2_ACC
    p2 = l2_intercept/100 * l2_acc/100 + (1 - l2_intercept/100) * L2_ACC
    
    results.append({
        'config': config_name,
        'match_threshold': mt,
        'low_overlap': loa,
        'weak_weight': wwf,
        'l0_acc': round(l0_acc, 1),
        'l0_intercept': round(l0_intercept, 1),
        'l1_acc': round(l1_acc, 1),
        'l1_intercept': round(l1_intercept, 1),
        'l2_acc': round(l2_acc, 1),
        'l2_intercept': round(l2_intercept, 1),
        'pipeline_fixed': round(p0 * 100, 1),
        'pipeline_unified': round(p1 * 100, 1),
        'pipeline_hybrid': round(p2 * 100, 1),
        'n_features': len(feature_names),
    })
    
    print(f"[{ci+1:2d}/{len(configs)}] mt={mt} loa={loa:8s} wwf={wwf} | "
          f"L0={l0_acc:5.1f}%/{l0_intercept:4.0f}% "
          f"L1={l1_acc:5.1f}%/{l1_intercept:4.0f}% "
          f"L2={l2_acc:5.1f}%/{l2_intercept:4.0f}% "
          f"pipe={p2*100:5.1f}%")

# Save full results
with open(f"{OUT}/sweep_results.json", 'w') as f:
    json.dump(results, f, indent=2)

# Print summary table sorted by pipeline_hybrid
print("\n" + "=" * 100)
print("SUMMARY (sorted by pipeline_hybrid)")
print("=" * 100)
print(f"{'mt':>4} {'low_overlap':>12} {'wwf':>5} | {'L0_acc':>6} {'L0_int':>6} {'L1_acc':>6} {'L1_int':>6} {'L2_acc':>6} {'L2_int':>6} {'P_fixed':>7} {'P_unif':>7} {'P_hyb':>7}")
print("-" * 100)
for r in sorted(results, key=lambda x: x['pipeline_hybrid'], reverse=True):
    print(f"{r['match_threshold']:>4} {r['low_overlap']:>12} {r['weak_weight']:>5} | "
          f"{r['l0_acc']:>5.1f}% {r['l0_intercept']:>5.1f}% "
          f"{r['l1_acc']:>5.1f}% {r['l1_intercept']:>5.1f}% "
          f"{r['l2_acc']:>5.1f}% {r['l2_intercept']:>5.1f}% "
          f"{r['pipeline_fixed']:>6.1f}% {r['pipeline_unified']:>6.1f}% {r['pipeline_hybrid']:>6.1f}%")
