#!/usr/bin/env python3
"""Cluster-level evaluation of cascade policies (offline, no LLM calls).

Rationale: reviewers asked for cluster-level (not just pair-level) evidence.
We compute, for each policy, the transitive closure of its same-entity
decisions over the candidate-pair graph, and compare the induced entity
clustering against the ground-truth clustering (transitive closure of GT
`same` labels) with pairwise precision/recall/F1 over entity pairs that
share a connected component of the candidate graph (the adjudicable
universe; name-blocking cannot compare pairs it never generated).

Policies:
  fixed      : fixed L1 thresholds 0.6 / -0.4, escalated pairs -> L2 verdict
  lr         : unified-LR OOF probabilities, thresholds 0.85 / 0.05
               (pipeline-aware frontier point, strict 91.66 @ 69.9 cost)
  pure_l2    : GLM-5-Turbo verdict on every pair

Strict convention: unresolved (unparseable) verdicts count as `different`.

Output: results/cluster_level_metrics.json
"""
import json
import os
import re
from collections import defaultdict

import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
RES = os.path.join(BASE, 'results')
CKPT = os.path.join(RES, 'checkpoints_v3', 'glm-5-turbo_checkpoint.json')

scored = json.load(open(os.path.join(RES, 'scored_pairs.json')))
pair_ids = [p['pair_id'] for p in scored]
y = np.array([p['label'] for p in scored])          # 1 = same
fixed_scores = np.array([p['l1_score_v2'] for p in scored])
vids_a = [p['vid_a'] for p in scored]
vids_b = [p['vid_b'] for p in scored]
N = len(y)

ckpt = json.load(open(CKPT))


def parse_verdict(v):
    if isinstance(v, dict):
        v = v.get('verdict', v.get('raw', ''))
    v = str(v).strip().lower()
    if re.search(r'\bdifferent\b|\bnot.*match\b|\bnot.*same\b|\bdistinct\b|\bno\b|\bunknown\b|否|不同', v):
        return 0
    if re.search(r'\bsame\b|\bmatch\b|\bidentical\b|\byes\b|是', v):
        return 1
    return None


l2_preds = np.full(N, -1)
for i, pid in enumerate(pair_ids):
    entry = ckpt.get(pid)
    if entry:
        l2_preds[i] = parse_verdict(entry.get('verdict', entry.get('raw', '')))

lr_proba = np.load(os.path.join(RES, 'oof_probabilities.npz'))['lr_proba']


def policy_decisions():
    dec = {}
    # fixed L1 0.6 / -0.4
    fixed = []
    for i in range(N):
        if fixed_scores[i] >= 0.6:
            fixed.append(1)
        elif fixed_scores[i] <= -0.4:
            fixed.append(0)
        else:
            v = l2_preds[i]
            fixed.append(1 if v == 1 else 0)   # strict: unknown -> not merged
    dec['fixed_l1_cascade'] = fixed
    # unified LR OOF 0.85 / 0.05
    lr = []
    for i in range(N):
        if lr_proba[i] >= 0.85:
            lr.append(1)
        elif lr_proba[i] <= 0.05:
            lr.append(0)
        else:
            v = l2_preds[i]
            lr.append(1 if v == 1 else 0)
    dec['pipeline_aware_lr'] = lr
    # pure L2
    dec['pure_l2'] = [1 if l2_preds[i] == 1 else 0 for i in range(N)]
    return dec


# ---- cluster machinery ----------------------------------------------------
class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def closure(decisions):
    d = DSU()
    for i in range(N):
        if decisions[i] == 1:
            d.union(vids_a[i], vids_b[i])
    return d


# adjudicable universe: entity pairs sharing a candidate-graph component
cand = DSU()
for i in range(N):
    cand.union(vids_a[i], vids_b[i])
comp_members = defaultdict(list)
all_vids = set(vids_a) | set(vids_b)
for v in sorted(all_vids):
    comp_members[cand.find(v)].append(v)
universe_pairs = 0
gt_same_pairs = set()
gt_d = closure(y.tolist())
for members in comp_members.values():
    members = sorted(members)
    for ai in range(len(members)):
        for bi in range(ai + 1, len(members)):
            universe_pairs += 1
            if gt_d.find(members[ai]) == gt_d.find(members[bi]):
                gt_same_pairs.add((members[ai], members[bi]))

out = {
    'protocol': ('pairwise cluster metrics over entity pairs sharing a candidate-graph '
                 'component (%d entities, %d adjudicable pairs)' % (len(all_vids), universe_pairs)),
    'gt_same_entity_pairs': len(gt_same_pairs),
    'gt_clusters_ge2': sum(1 for members in comp_members.values()
                           if len(members) >= 2
                           and any(gt_d.find(m) == gt_d.find(members[0]) and m != members[0]
                                   for m in members[1:])),
    'policies': {},
}

for name, decisions in policy_decisions().items():
    pd_ = closure(decisions)
    pred_same = set()
    for members in comp_members.values():
        members = sorted(members)
        for ai in range(len(members)):
            for bi in range(ai + 1, len(members)):
                if pd_.find(members[ai]) == pd_.find(members[bi]):
                    pred_same.add((members[ai], members[bi]))
    tp = len(pred_same & gt_same_pairs)
    prec = tp / len(pred_same) * 100 if pred_same else 0.0
    rec = tp / len(gt_same_pairs) * 100
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # count predicted clusters with >= 2 members
    pc = defaultdict(list)
    for members in comp_members.values():
        for m in members:
            pc[pd_.find(m)].append(m)
    out['policies'][name] = {
        'pairwise_precision_pct': round(prec, 1),
        'pairwise_recall_pct': round(rec, 1),
        'pairwise_f1_pct': round(f1, 1),
        'predicted_same_pairs': len(pred_same),
        'true_positives': tp,
        'predicted_clusters_ge2': sum(1 for v in pc.values() if len(v) >= 2),
    }
    print(name, out['policies'][name])

json.dump(out, open(os.path.join(RES, 'cluster_level_metrics.json'), 'w'),
          indent=1, ensure_ascii=False)
print('universe:', universe_pairs, 'gt same pairs:', len(gt_same_pairs))
print('written: results/cluster_level_metrics.json')
