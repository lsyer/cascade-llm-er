#!/usr/bin/env python3
"""
sweep_jaccard_fine.py — Fine-grained Jaccard threshold sweep at 0.05 intervals.
"""
import json, os, sys
import numpy as np
from collections import Counter

DATA = "/root/workspace/tanshu_docs/experiments/release/data"
CKPT = "/root/workspace/tanshu_docs/experiments/release/results/checkpoints_v3/glm-5.2_checkpoint.json"
OUT = "/root/workspace/tanshu_docs/experiments/release/results"

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

def compare_field(field, va, vb, mt):
    if not va or not vb: return 'unknown'
    if va == vb: return 'match'
    if _containment(va, vb): return 'match'
    if field in STRONG: return 'conflict'
    sim = _jaccard(va, vb)
    if sim >= mt: return 'match'
    return 'conflict'

MERGE_T = 0.6
REJECT_T = -0.4

# Pre-compute all raw scores for each threshold
thresholds = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 to 0.95

print(f"Sweeping {len(thresholds)} Jaccard thresholds at 0.05 intervals\n")
print(f"{'τ':>5} | {'L0_acc':>6} {'Interc':>6} {'M_prec':>6} {'R_prec':>6} {'FP':>4} {'FN':>4} {'FP+FN':>5} {'Pipeline':>8}")
print("-" * 75)

results = []
for mt in thresholds:
    merge_n = merge_c = reject_n = reject_c = esc_n = 0
    fp = fn = 0
    for p in labeled:
        pa, pb = p['props_a'], p['props_b']
        et = p['etype']
        
        if _hard_conflict(pa, pb, et):
            score = -1.0
        else:
            fields = _effective_fields(pa, pb)
            if not fields:
                score = 0.0
            else:
                weight = 1.0 / len(fields)
                score = 0.0
                for f in fields:
                    va = _get(pa, f); vb = _get(pb, f)
                    result = compare_field(f, va, vb, mt)
                    if result == 'match': score += weight
                    elif result == 'conflict': score -= weight * 1.5
                score = max(-1.0, min(1.0, score))
                na, nb = (p.get('name_a') or '').lower().strip(), (p.get('name_b') or '').lower().strip()
                if na and nb:
                    if na == nb: score = min(1.0, score + 0.3)
                    elif _containment(na, nb): score = min(1.0, score + 0.2)
                    elif _jaccard(na, nb) >= 0.5: score = min(1.0, score + 0.15)
                score = max(-1.0, min(1.0, score))
        
        if score >= MERGE_T:
            merge_n += 1
            if p['label'] == 1: merge_c += 1
            else: fp += 1
        elif score <= REJECT_T:
            reject_n += 1
            if p['label'] == 0: reject_c += 1
            else: fn += 1
        else:
            esc_n += 1
    
    intercepted = merge_n + reject_n
    correct = merge_c + reject_c
    l0_acc = correct / intercepted * 100 if intercepted else 0
    intercept_rate = intercepted / len(labeled) * 100
    m_prec = merge_c / merge_n * 100 if merge_n else 0
    r_prec = reject_c / reject_n * 100 if reject_n else 0
    pipeline = (intercept_rate/100) * (l0_acc/100) + (esc_n/len(labeled)) * L2_ACC
    
    marker = ""
    if mt == 0.7: marker = " ← chosen"
    
    print(f"{mt:>5.2f} | {l0_acc:>5.1f}% {intercept_rate:>5.1f}% {m_prec:>5.1f}% {r_prec:>5.1f}% {fp:>4} {fn:>4} {fp+fn:>5} {pipeline*100:>7.1f}%{marker}")
    
    results.append({
        'tau': mt, 'l0_acc': round(l0_acc, 1), 'intercept': round(intercept_rate, 1),
        'merge_prec': round(m_prec, 1), 'reject_prec': round(r_prec, 1),
        'fp': fp, 'fn': fn, 'total_errors': fp + fn,
        'pipeline': round(pipeline * 100, 1)
    })

with open(f"{OUT}/jaccard_fine_sweep.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} configs to {OUT}/jaccard_fine_sweep.json")
