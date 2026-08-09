#!/usr/bin/env python3
"""Compute pure L2 (GLM-5-Turbo) accuracy on ALL 2591 pairs.
The checkpoint has L2 verdicts for every pair (not just escalated ones).
Compare with L2 accuracy on the 1591 escalated subset."""
import json, os, re

BASE = os.path.join(os.path.dirname(__file__), '..')
CKPT = os.path.join(BASE, 'results', 'checkpoints_v3', 'glm-5-turbo_checkpoint.json')
SCORED = os.path.join(BASE, 'results', 'scored_pairs.json')

with open(CKPT) as f:
    ckpt = json.load(f)
with open(SCORED) as f:
    scored = json.load(f)

# Build label map
label_map = {p['pair_id']: p['label'] for p in scored}
decision_map = {p['pair_id']: p['l1_decision_v2'] for p in scored}

print(f"Checkpoint entries: {len(ckpt)}")
print(f"Scored pairs: {len(scored)}")

# Parse verdict -> 0/1
def parse_verdict(v):
    if isinstance(v, dict):
        v = v.get('verdict', v.get('raw', ''))
    v = str(v).strip().lower()
    # Match "same" / "different" / "match" / "not match"
    if re.search(r'\bsame\b|\bmatch\b|\bidentical\b|yes|是', v):
        return 1
    if re.search(r'\bdifferent\b|\bnot.*match\b|\bdistinct\b|no|否|不同', v):
        return 0
    return None  # unknown

# Compute L2 accuracy
stats = {'all': [0,0,0], 'intercepted': [0,0,0], 'escalated': [0,0,0]}
# [correct, wrong, unknown]

for pid, label in label_map.items():
    entry = ckpt.get(pid)
    if not entry:
        continue
    verdict = entry.get('verdict', entry.get('raw', ''))
    pred = parse_verdict(verdict)
    decision = decision_map.get(pid, 'escalate')
    
    for subset in ['all']:
        if pred is None:
            stats[subset][2] += 1
        elif pred == label:
            stats[subset][0] += 1
        else:
            stats[subset][1] += 1
    
    # Also by subset
    if decision in ('merge', 'reject'):
        sub = 'intercepted'
    else:
        sub = 'escalated'
    if pred is None:
        stats[sub][2] += 1
    elif pred == label:
        stats[sub][0] += 1
    else:
        stats[sub][1] += 1

print(f"\n{'Subset':>15} {'N':>5} {'Correct':>8} {'Wrong':>6} {'Unknown':>8} {'Accuracy':>9}")
for sub, (c, w, u) in stats.items():
    n = c + w + u
    resolved = c + w
    acc = c / resolved * 100 if resolved > 0 else 0
    print(f"{sub:>15} {n:>5} {c:>8} {w:>6} {u:>8} {acc:>8.1f}%")

# Key numbers
all_resolved = stats['all'][0] + stats['all'][1]
esc_resolved = stats['escalated'][0] + stats['escalated'][1]
int_resolved = stats['intercepted'][0] + stats['intercepted'][1]
print(f"\n=== Key numbers ===")
print(f"Pure L2 on ALL 2591 pairs: {stats['all'][0]}/{all_resolved} = {stats['all'][0]/all_resolved*100:.1f}% ({stats['all'][2]} unknown)")
print(f"L2 on escalated (1591):   {stats['escalated'][0]}/{esc_resolved} = {stats['escalated'][0]/esc_resolved*100:.1f}% ({stats['escalated'][2]} unknown)")
print(f"L2 on intercepted (1000): {stats['intercepted'][0]}/{int_resolved} = {stats['intercepted'][0]/int_resolved*100:.1f}% ({stats['intercepted'][2]} unknown)")
