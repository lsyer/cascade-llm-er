#!/usr/bin/env python3
"""compute_prf1.py — Precision/Recall/F1 for every reported cascade configuration.

Replays per-pair final verdicts (L1 interception -> L1 label; escalation -> L2
checkpoint verdict; unresolved -> strict error), verifies aggregate counts against
the saved experiment outputs (pipeline_accuracy_empirical.json,
threshold_selection_experiments.json, final_operating_points.json), then computes
the confusion matrix for the same/different decision.

Conventions:
  - strict: unresolved pairs counted as errors (consistent with strict accuracy).
    For P/R/F1 they are forced to the wrong class (FP or FN).
  - resolved-only: unresolved pairs excluded from the denominator.
Output: results/prf1_metrics.json
"""
import json, os, re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

BASE = os.path.join(os.path.dirname(__file__), '..')
RESULTS = os.path.join(BASE, 'results')

# ---------- load ----------
scored = json.load(open(os.path.join(RESULTS, 'scored_pairs.json')))
pair_ids = [p['pair_id'] for p in scored]
y = np.array([p['label'] for p in scored])
types = np.array([p['type_a'].replace('activity', 'event') for p in scored])
N = len(y)

all_feature_names = set()
for p in scored:
    all_feature_names.update(p['features_v2'].keys())
feature_names = sorted(all_feature_names)
X = np.array([[p['features_v2'].get(f, 0) for f in feature_names] for p in scored])

ckpt = json.load(open(os.path.join(RESULTS, 'checkpoints_v3', 'glm-5-turbo_checkpoint.json')))

def parse_verdict(v):
    """Strict mapping, identical to run_experiments.verdict_to_label: only exact
    'same'/'different' count; content-filter errors (raw ERROR_400) stay unresolved."""
    v = str(v).strip().lower()
    if v == 'same':
        return 1
    if v == 'different':
        return 0
    return -1

l2_label = {}
for pid in pair_ids:
    entry = ckpt.get(pid)
    l2_label[pid] = parse_verdict(entry.get('verdict', entry.get('raw', ''))) if entry else -1

# ---------- recover folds + OOF probabilities ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_of = np.full(N, -1)
lr_oof = np.full(N, np.nan)
for fi, (tr, te) in enumerate(skf.split(X_scaled, y)):
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_scaled[tr], y[tr])
    lr_oof[te] = lr.predict_proba(X_scaled[te])[:, 1]
    fold_of[te] = fi

d = np.load(os.path.join(RESULTS, 'oof_probabilities.npz'))
assert np.allclose(lr_oof, d['lr_proba'], atol=1e-9), "fold recovery failed: lr OOF mismatch"
rf_oof = d['rf_proba']
print(f"fold recovery verified (lr OOF max diff {np.abs(lr_oof - d['lr_proba']).max():.2e})")

# hybrid OOF: per-type LR (n_train>=50) else unified fallback (replicate run_experiments.py)
TYPE_ORDER = sorted(set(types))
hybrid_oof = np.full(N, np.nan)
for tr, te in skf.split(X_scaled, y):
    lr_u = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_u.fit(X_scaled[tr], y[tr])
    for t in TYPE_ORDER:
        tr_mask = types[tr] == t
        te_mask = types[te] == t
        idx = te[te_mask]
        if len(idx) == 0:
            continue
        if tr_mask.sum() >= 50:
            lr_t = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
            lr_t.fit(X_scaled[tr][tr_mask], y[tr][tr_mask])
            hybrid_oof[idx] = lr_t.predict_proba(X_scaled[idx])[:, 1]
        else:
            hybrid_oof[idx] = lr_u.predict_proba(X_scaled[idx])[:, 1]
assert not np.isnan(hybrid_oof).any()

# ---------- replay engine ----------
def replay(preds):
    """preds: -1 = escalate. Returns final labels + stats dict."""
    final = np.full(N, -2)  # -2 unresolved
    l1_decided = l1_correct = l2_used = l2_correct = unresolved = 0
    for i in range(N):
        if preds[i] != -1:
            l1_decided += 1
            final[i] = preds[i]
            if preds[i] == y[i]:
                l1_correct += 1
        else:
            l2_used += 1
            lbl = l2_label[pair_ids[i]]
            if lbl == -1:
                unresolved += 1
            else:
                final[i] = lbl
                if lbl == y[i]:
                    l2_correct += 1
    stats = {
        'total': N, 'resolved': N - unresolved, 'unresolved': unresolved,
        'l1_decided': l1_decided, 'l1_correct': l1_correct,
        'l2_used': l2_used, 'l2_correct': l2_correct,
        'l1_acc_pct': round(l1_correct / l1_decided * 100, 2) if l1_decided else 0.0,
        'strict_pipeline_acc_pct': round((l1_correct + l2_correct) / N * 100, 2),
        'intercept_pct': round(l1_decided / N * 100, 2),
        'llm_calls_pct': round(l2_used / N * 100, 2),
    }
    return final, stats

def prf1(final, strict=True):
    """Confusion matrix + P/R/F1 for class 'same' (positive)."""
    f = final.copy()
    if strict:
        # unresolved -> forced wrong prediction
        for i in np.where(f == -2)[0]:
            f[i] = 1 - y[i]
    else:
        keep = f != -2
        f, yy = f[keep], y[keep]
    tp = int(((f == 1) & (y[:len(f)] == 1)).sum())
    fp = int(((f == 1) & (y[:len(f)] == 0)).sum())
    tn = int(((f == 0) & (y[:len(f)] == 0)).sum())
    fn = int(((f == 0) & (y[:len(f)] == 1)).sum())
    prec = tp / (tp + fp) * 100 if tp + fp else 0.0
    rec = tp / (tp + fn) * 100 if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / (tp + fp + tn + fn) * 100
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision_pct': round(prec, 2), 'recall_pct': round(rec, 2),
            'f1_pct': round(f1, 2), 'accuracy_pct': round(acc, 2),
            'false_merge': fp, 'false_reject': fn}

def threshold_preds(scores, mt, rt):
    p = np.full(N, -1)
    p[scores >= mt] = 1
    p[scores <= rt] = 0
    return p

def fold_threshold_preds(oof, th_by_fold):
    """Per-fold saved thresholds applied to OOF probabilities."""
    p = np.full(N, -1)
    for i in range(N):
        th = th_by_fold[fold_of[i]]
        if oof[i] >= th['merge_t']:
            p[i] = 1
        elif oof[i] <= th['reject_t']:
            p[i] = 0
    return p

def hybrid_fold_threshold_preds(th_entries):
    """Hybrid per-fold per-type thresholds (list in fold-major, TYPE_ORDER order)."""
    lookup = {}
    ti = 0
    folds = list(skf.split(X_scaled, y))
    for fi, (tr, te) in enumerate(folds):
        for t in TYPE_ORDER:
            if not (types[te] == t).any():
                continue
            entry = th_entries[ti]; ti += 1
            assert entry['type'] == t, f"entry order mismatch: {entry['type']} != {t}"
            lookup[(fi, t)] = entry
    assert ti == len(th_entries), f"consumed {ti} of {len(th_entries)} entries"
    p = np.full(N, -1)
    for i in range(N):
        entry = lookup[(fold_of[i], types[i])]
        if hybrid_oof[i] >= entry['merge_t']:
            p[i] = 1
        elif hybrid_oof[i] <= entry['reject_t']:
            p[i] = 0
    return p

# ---------- configurations ----------
tse = json.load(open(os.path.join(RESULTS, 'threshold_selection_experiments.json')))
fop = json.load(open(os.path.join(RESULTS, 'final_operating_points.json')))

configs = {}
# fixed cascade (rule decisions as recorded)
configs['fixed_cascade'] = np.array([1 if p['l1_decision_v2'] == 'merge' else 0 if p['l1_decision_v2'] == 'reject' else -1 for p in scored])
# pure L2
configs['pure_l2'] = np.full(N, -1)
# Table 2 mode 1: aligned thresholds 0.8/0.3
configs['t2_mode1_unified'] = threshold_preds(lr_oof, 0.8, 0.3)
configs['t2_mode1_hybrid'] = threshold_preds(hybrid_oof, 0.8, 0.3)
# Table 2 modes 2/3: per-fold thresholds
configs['t2_mode2_unified'] = fold_threshold_preds(lr_oof, tse['mode_2_match_fixed_pipeline_max_intercept']['unified_thresholds_by_fold'])
configs['t2_mode2_hybrid'] = hybrid_fold_threshold_preds(tse['mode_2_match_fixed_pipeline_max_intercept']['hybrid_thresholds_by_fold'])
configs['t2_mode3_unified'] = fold_threshold_preds(lr_oof, tse['mode_3_maximize_empirical_pipeline']['unified_thresholds_by_fold'])
configs['t2_mode3_hybrid'] = hybrid_fold_threshold_preds(tse['mode_3_maximize_empirical_pipeline']['hybrid_thresholds_by_fold'])
# Table 4: l1-acc-max operating points
configs['t4_fixed'] = threshold_preds(np.array([p['l1_score_v2'] for p in scored]), fop['Fixed L1']['merge_t'], fop['Fixed L1']['reject_t'])
configs['t4_lr'] = threshold_preds(lr_oof, fop['Unified LR']['merge_t'], fop['Unified LR']['reject_t'])
configs['t4_rf'] = threshold_preds(rf_oof, fop['Random Forest']['merge_t'], fop['Random Forest']['reject_t'])

# ---------- verify replay against saved aggregates ----------
verifs = {
    'fixed_cascade': ('pipeline_accuracy_empirical.json', 'fixed',
                      ['l1_decided', 'l1_correct', 'l2_used', 'l2_correct'], 'strict_pipeline_acc_pct'),
}
def close(a, b, tol=0.06):
    return abs(a - b) <= tol

results = {}
all_ok = True
for name, preds in configs.items():
    final, stats = replay(preds)
    results[name] = {'stats': stats, 'strict': prf1(final, strict=True), 'resolved_only': prf1(final, strict=False)}
    print(f"{name:22s} strict_acc={stats['strict_pipeline_acc_pct']:6.2f}  "
          f"P={results[name]['strict']['precision_pct']:5.2f} R={results[name]['strict']['recall_pct']:5.2f} "
          f"F1={results[name]['strict']['f1_pct']:5.2f}  (intercept {stats['intercept_pct']:5.1f}%, unresolved {stats['unresolved']})")

# explicit verifications
pa = json.load(open(os.path.join(RESULTS, 'pipeline_accuracy_empirical.json')))['fixed']
s = results['fixed_cascade']['stats']
for k in ['l1_decided', 'l1_correct', 'l2_used', 'l2_correct']:
    ok = s[k] == pa[k]; all_ok &= ok
    print(f"verify fixed.{k}: replay={s[k]} saved={pa[k]} {'OK' if ok else 'MISMATCH'}")
ok = close(s['strict_pipeline_acc_pct'], pa['strict_accuracy_pct']); all_ok &= ok
print(f"verify fixed.strict_acc: replay={s['strict_pipeline_acc_pct']} saved={pa['strict_accuracy_pct']} {'OK' if ok else 'MISMATCH'}")

for mode_key, cfgs in [('mode_1_probability_aligned_to_fixed', ['t2_mode1_unified', 't2_mode1_hybrid']),
                       ('mode_2_match_fixed_pipeline_max_intercept', ['t2_mode2_unified', 't2_mode2_hybrid']),
                       ('mode_3_maximize_empirical_pipeline', ['t2_mode3_unified', 't2_mode3_hybrid'])]:
    for scorer, cfg in [('unified', cfgs[0]), ('hybrid', cfgs[1])]:
        saved = tse[mode_key][scorer]
        s = results[cfg]['stats']
        for k in ['l1_decided', 'l1_correct', 'l2_used', 'l2_correct']:
            ok = s[k] == saved[k]; all_ok &= ok
            if not ok:
                print(f"verify {cfg}.{k}: replay={s[k]} saved={saved[k]} MISMATCH")
        ok = close(s['strict_pipeline_acc_pct'], saved.get('strict_pipeline_acc_pct', saved.get('pipeline_acc_pct', 0)))
        all_ok &= ok
        print(f"verify {cfg}: counts {'OK' if all(s[k]==saved[k] for k in ['l1_decided','l1_correct','l2_used','l2_correct']) else 'MISMATCH'}, "
              f"strict replay={s['strict_pipeline_acc_pct']} saved={saved.get('strict_pipeline_acc_pct','n/a')}")

for cfg, key in [('t4_fixed', 'Fixed L1'), ('t4_lr', 'Unified LR'), ('t4_rf', 'Random Forest')]:
    s = results[cfg]['stats']
    ok = close(s['strict_pipeline_acc_pct'], fop[key]['pipeline_empirical'])
    all_ok &= ok
    print(f"verify {cfg}: pipeline replay={s['strict_pipeline_acc_pct']} saved={fop[key]['pipeline_empirical']} {'OK' if ok else 'MISMATCH'}")

pl2 = json.load(open(os.path.join(RESULTS, 'pipeline_vs_pure_l2.json')))
print(f"pure_l2 saved check: {json.dumps({k: pl2[k] for k in list(pl2)[:4]}, ensure_ascii=False)[:200]}")

results['_verification'] = {'all_ok': bool(all_ok), 'lr_oof_max_diff': float(np.abs(lr_oof - d['lr_proba']).max())}
json.dump(results, open(os.path.join(RESULTS, 'prf1_metrics.json'), 'w'), indent=1, ensure_ascii=False)
print(f"\nALL VERIFICATIONS {'PASSED' if all_ok else 'FAILED'} -> results/prf1_metrics.json")
