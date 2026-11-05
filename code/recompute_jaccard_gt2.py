#!/usr/bin/env python3
"""recompute_jaccard_gt2.py — Jaccard-threshold (tau) sweep for the fixed L1
scorer under GT v2 labels with strict empirical pipeline replay.

For each tau in 0.05..0.95 (19 configs), l1_scorer.FIELD_MATCH_THRESHOLD is
patched to tau, all 2,591 labeled pairs are re-scored (l1_score_with_name,
fixed thresholds 0.6 / -0.4), and both local L1 metrics and the strict
empirical pipeline (L1-correct on intercepted + GLM-correct on escalated)
are recorded. Output: results/jaccard_sensitivity_gt2.json (same field names
as the legacy results/jaccard_sensitivity.json so the figure script works).
"""
import json, os, sys
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(BASE, 'code'))
import l1_scorer

DATA = os.path.join(BASE, 'data')
RES = os.path.join(BASE, 'results')

data = json.load(open(os.path.join(DATA, 'dataset_v3_cleaned.json')))
pairs = data['same_type_pairs']
gt = json.load(open(os.path.join(DATA, 'minec_ground_truth_v2.json')))
gt_map = {r['pair_id']: r for r in gt['records']}
labeled = []
for p in pairs:
    rec = gt_map.get(p['pair_id'])
    if rec and rec.get('final_label') in ('same', 'different'):
        q = dict(p)
        q['label'] = 1 if rec['final_label'] == 'same' else 0
        labeled.append(q)
print(f"labeled pairs: {len(labeled)}")

ck = json.load(open(os.path.join(RES, 'checkpoints_v3', 'glm-5-turbo_checkpoint.json')))
def l2v(pid):
    v = str(ck.get(pid, {}).get('verdict', '')).strip().lower()
    return 1 if v == 'same' else (0 if v == 'different' else -1)

ORIG = l1_scorer.FIELD_MATCH_THRESHOLD
rows = []
for tau in [round(0.05 * i, 2) for i in range(1, 20)]:
    l1_scorer.FIELD_MATCH_THRESHOLD = tau
    dec = []
    for p in labeled:
        pa, pb = p.get('props_a', {}), p.get('props_b', {})
        if isinstance(pa, str):
            pa = json.loads(pa)
        if isinstance(pb, str):
            pb = json.loads(pb)
        et = p['type_a']
        if et == 'activity':
            et = 'event'
        score, decision, _ = l1_scorer.l1_score_with_name(
            p.get('name_a', ''), p.get('name_b', ''), pa, pb, et)
        dec.append((score, decision, p['label'], p['pair_id']))
    merge = [d for d in dec if d[1] == 'merge']
    reject = [d for d in dec if d[1] == 'reject']
    inter = merge + reject
    fp = sum(1 for d in merge if d[2] == 0)
    fn = sum(1 for d in reject if d[2] == 1)
    l1_ok = sum(1 for d in inter if (1 if d[1] == 'merge' else 0) == d[2])
    esc_pids = [d[3] for d in dec if d[1] == 'escalate']
    l2_ok = sum(1 for q in esc_pids if l2v(q) == dict((d[3], d[2]) for d in dec)[q])
    n = len(dec)
    rows.append({
        'tau': tau,
        'l0_acc': round(l1_ok / len(inter) * 100, 1) if inter else 0,
        'intercept': round(len(inter) / n * 100, 1),
        'merge_prec': round((len(merge) - fp) / len(merge) * 100, 1) if merge else 0,
        'reject_prec': round((len(reject) - fn) / len(reject) * 100, 1) if reject else 0,
        'fp': fp, 'fn': fn, 'total_errors': fp + fn,
        'pipeline': round((l1_ok + l2_ok) / n * 100, 1),
    })
    print(rows[-1])
l1_scorer.FIELD_MATCH_THRESHOLD = ORIG
json.dump(rows, open(os.path.join(RES, 'jaccard_sensitivity_gt2.json'), 'w'), indent=2)
print('saved results/jaccard_sensitivity_gt2.json')
