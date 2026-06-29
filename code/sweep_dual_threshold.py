#!/usr/bin/env python3
"""
sweep_dual_threshold.py — Dual threshold sweep for L1 routing.

Sweeps both merge_threshold (0.40-0.90) and reject_threshold (-0.10 to -0.50)
to find the optimal pair for pipeline accuracy.

Uses v2 scorer config: mt=0.6 (field comparison), conflict, wwf=1.0
"""
import json, os, sys, itertools
import numpy as np
from collections import Counter

DATA = "/root/workspace/tanshu_docs/experiments/release/data"
CKPT = "/root/workspace/tanshu_docs/experiments/release/results/checkpoints_v3/glm-5.2_checkpoint.json"
OUT = "/root/workspace/tanshu_docs/experiments/release/results"

# ============================================================
# Load data
# ============================================================
with open(f"{DATA}/dataset_v3_cleaned.json") as f:
    pairs = json.load(f)['same_type_pairs']
with open(CKPT) as f:
    checkpoint = json.load(f)
for p in pairs:
    v = checkpoint.get(p['pair_id'], {}).get('verdict', '')
    p['label'] = 1 if v == 'same' else 0
labeled = [p for p in pairs if p.get('label') is not None]
for p in labeled:
    for k in ('props_a', 'props_b'):
        if isinstance(p.get(k), str):
            p[k] = json.loads(p[k])
    p['etype'] = p['type_a'].replace('activity', 'event')

L2_ACC = 0.971
EXCLUDE = {'created_at', 'updated_at', 'confidence', 'labels', 'vid', 'id', 'source_pk'}
STRONG = {
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

def _jaccard(a, b):
    wa = set(a.lower().split()); wb = set(b.lower().split())
    if not wa or not wb: return 0.0
    return len(wa & wb) / len(wa | wb)

def _containment(a, b):
    a = a.lower().strip(); b = b.lower().strip()
    return bool(a and b and (a in b or b in a))

def _effective_fields(pa, pb):
    all_f = set()
    for k in list(pa.keys()) + list(pb.keys()):
        if k not in EXCLUDE: all_f.add(k)
    return sorted([f for f in all_f if _has_value(pa, f) or _has_value(pb, f)])

def _hard_conflict(pa, pb, etype):
    if etype == 'person':
        ga, gb = _get(pa, 'gender'), _get(pb, 'gender')
        if ga and gb and ga != gb: return True
    if etype == 'equipment':
        ca, cb = _get(pa, 'equip_type'), _get(pb, 'equip_type')
        if ca and cb and ca != cb:
            ac_kw = ['aircraft', 'plane', 'helicopter', 'fighter', 'bomber', 'drone']
            sh_kw = ['ship', 'carrier', 'destroyer', 'frigate', 'submarine', 'vessel']
            if (any(k in ca for k in ac_kw) and any(k in cb for k in sh_kw)) or \
               (any(k in ca for k in sh_kw) and any(k in cb for k in ac_kw)):
                return True
    return False

# v2 scorer params: field match_threshold=0.6, conflict on low overlap, equal weight
FIELD_MT = 0.65  # Jaccard threshold for weak-field match/conflict (locked from fine sweep)

def compare_field(field, va, vb):
    if not va or not vb: return 'unknown'
    if va == vb: return 'match'
    if _containment(va, vb): return 'match'
    if field in STRONG:
        return 'conflict'
    sim = _jaccard(va, vb)
    if sim >= FIELD_MT: return 'match'
    return 'conflict'  # low_overlap=conflict

def score_pair_raw(pa, pb, etype, name_a, name_b):
    """Compute raw score (before routing thresholds)."""
    if _hard_conflict(pa, pb, etype):
        return -1.0
    
    fields = _effective_fields(pa, pb)
    if not fields:
        return 0.0
    
    n = len(fields)
    weight = 1.0 / n
    score = 0.0
    
    for f in fields:
        va = _get(pa, f); vb = _get(pb, f)
        result = compare_field(f, va, vb)
        if result == 'match':
            score += weight
        elif result == 'conflict':
            score -= weight * 1.5  # penalty_factor
    
    score = max(-1.0, min(1.0, score))
    
    # Name bonus
    if name_a and name_b:
        na, nb = name_a.lower().strip(), name_b.lower().strip()
        if na == nb:
            score = min(1.0, score + 0.3)
        elif _containment(na, nb):
            score = min(1.0, score + 0.2)
        elif _jaccard(na, nb) >= 0.5:
            score = min(1.0, score + 0.15)
    
    return max(-1.0, min(1.0, score))

# ============================================================
# Pre-compute all raw scores once
# ============================================================
print("Pre-computing raw scores for 2639 pairs...")
raw_scores = []
for p in labeled:
    s = score_pair_raw(p['props_a'], p['props_b'], p['etype'], p['name_a'], p['name_b'])
    raw_scores.append((s, p['label']))
print(f"Done. Score range: [{min(s for s,_ in raw_scores):.2f}, {max(s for s,_ in raw_scores):.2f}]")

# ============================================================
# Sweep dual thresholds
# ============================================================
merge_thresholds = np.arange(0.40, 0.91, 0.05)  # 0.40, 0.45, ..., 0.90
reject_thresholds = np.arange(-0.50, -0.09, 0.05)  # -0.50, -0.45, ..., -0.10

print(f"\nSweeping {len(merge_thresholds)} merge × {len(reject_thresholds)} reject = {len(merge_thresholds)*len(reject_thresholds)} configs\n")

results = []
n_total = len(raw_scores)

for mt in merge_thresholds:
    for rt in reject_thresholds:
        merge_n = merge_correct = 0
        reject_n = reject_correct = 0
        escalate_n = 0
        
        for score, label in raw_scores:
            if score >= mt:
                merge_n += 1
                if label == 1: merge_correct += 1
            elif score <= rt:
                reject_n += 1
                if label == 0: reject_correct += 1
            else:
                escalate_n += 1
        
        intercepted = merge_n + reject_n
        if intercepted == 0:
            continue
        
        correct = merge_correct + reject_correct
        l1_acc = correct / intercepted * 100
        interception_rate = intercepted / n_total * 100
        
        # Precision breakdown
        merge_prec = merge_correct / merge_n * 100 if merge_n else 0
        reject_prec = reject_correct / reject_n * 100 if reject_n else 0
        
        # Pipeline (fixed L1 + L2)
        escalation_rate = escalate_n / n_total
        pipeline = (interception_rate/100) * (l1_acc/100) + escalation_rate * L2_ACC
        pipeline *= 100
        
        # L2 cost = how many pairs need L2
        l2_cost = escalation_rate * 100
        
        results.append({
            'merge_threshold': round(float(mt), 2),
            'reject_threshold': round(float(rt), 2),
            'l1_accuracy': round(l1_acc, 1),
            'interception': round(interception_rate, 1),
            'merge_n': merge_n,
            'merge_correct': merge_correct,
            'merge_precision': round(merge_prec, 1),
            'reject_n': reject_n,
            'reject_correct': reject_correct,
            'reject_precision': round(reject_prec, 1),
            'escalate_n': escalate_n,
            'l2_cost': round(l2_cost, 1),
            'pipeline': round(pipeline, 1),
        })

# Save
with open(f"{OUT}/dual_threshold_sweep.json", 'w') as f:
    json.dump(results, f, indent=2)

# ============================================================
# Print results
# ============================================================

# Top 10 by pipeline
print("=" * 120)
print("TOP 15 by Pipeline Accuracy")
print("=" * 120)
print(f"{'merge_t':>8} {'reject_t':>9} | {'L1_acc':>6} {'Interc':>6} {'Merge_P':>7} {'Rej_P':>6} {'L2_cost':>7} {'Pipeline':>8}")
print("-" * 80)
for r in sorted(results, key=lambda x: x['pipeline'], reverse=True)[:15]:
    print(f"{r['merge_threshold']:>7.2f} {r['reject_threshold']:>8.2f} | "
          f"{r['l1_accuracy']:>5.1f}% {r['interception']:>5.1f}% "
          f"{r['merge_precision']:>6.1f}% {r['reject_precision']:>5.1f}% "
          f"{r['l2_cost']:>6.1f}% {r['pipeline']:>7.1f}%")

# Heatmap-style: pipeline by merge×reject
print("\n" + "=" * 120)
print("Pipeline Accuracy Heatmap (rows=merge_threshold, cols=reject_threshold)")
print("=" * 120)
header = "merge\\reject"
for rt in reject_thresholds:
    header += f" {rt:>6.2f}"
print(header)
for mt in merge_thresholds:
    row = f"  {mt:>5.2f}    "
    for rt in reject_thresholds:
        matching = [r for r in results if r['merge_threshold'] == round(float(mt),2) and r['reject_threshold'] == round(float(rt),2)]
        if matching:
            p = matching[0]['pipeline']
            row += f" {p:>6.1f}"
        else:
            row += f"    ---"
    print(row)

# Also show L2 cost heatmap
print("\n" + "=" * 120)
print("L2 Cost (%) Heatmap")
print("=" * 120)
header = "merge\\reject"
for rt in reject_thresholds:
    header += f" {rt:>6.2f}"
print(header)
for mt in merge_thresholds:
    row = f"  {mt:>5.2f}    "
    for rt in reject_thresholds:
        matching = [r for r in results if r['merge_threshold'] == round(float(mt),2) and r['reject_threshold'] == round(float(rt),2)]
        if matching:
            c = matching[0]['l2_cost']
            row += f" {c:>6.1f}"
        else:
            row += f"    ---"
    print(row)

# Find the knee point: highest pipeline with L2 cost < 30%
print("\n" + "=" * 80)
print("BEST with L2 cost < 30%")
print("=" * 80)
feasible = [r for r in results if r['l2_cost'] < 30]
if feasible:
    best = max(feasible, key=lambda x: x['pipeline'])
    print(f"merge_t={best['merge_threshold']}, reject_t={best['reject_threshold']}")
    print(f"  L1={best['l1_accuracy']:.1f}%, interception={best['interception']:.1f}%, "
          f"L2_cost={best['l2_cost']:.1f}%, pipeline={best['pipeline']:.1f}%")

# Find best balance: acc > 75% with max interception
print("\nBEST with L1_acc > 75% (reliable L1, maximize interception)")
reliable = [r for r in results if r['l1_accuracy'] > 75]
if reliable:
    best = max(reliable, key=lambda x: x['interception'])
    print(f"merge_t={best['merge_threshold']}, reject_t={best['reject_threshold']}")
    print(f"  L1={best['l1_accuracy']:.1f}%, interception={best['interception']:.1f}%, "
          f"L2_cost={best['l2_cost']:.1f}%, pipeline={best['pipeline']:.1f}%")
