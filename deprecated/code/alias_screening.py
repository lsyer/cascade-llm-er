#!/usr/bin/env python3
"""alias_screening.py — Triage the 403 alias-overlap non-candidate pairs into
rule categories, and draw a stratified sample for expert verification.

Categories:
  cat1_format_variant: normalized names identical (punctuation/spacing variants) -> near-certain duplicate
  cat2_containment:    one normalized name contains the other (title/qualifier variants) -> likely duplicate, needs judgment
  cat3_alias_only:     names differ beyond containment, shared alias/attribute -> uncertain
Output: results/alias_screening.json + results/alias_sample_review.md
"""
import json, os, re, random
from collections import defaultdict

BASE = os.path.join(os.path.dirname(__file__), '..')
ents = json.load(open('/root/workspace/tanshu-specs/experiments/minec_entity_snapshots/entities.json'))
ds = json.load(open(os.path.join(BASE, 'data', 'dataset_v3_cleaned.json')))

def norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower()) if s else ''

cand_set = {frozenset((p['vid_a'], p['vid_b'])) for p in ds['same_type_pairs'] + ds['cross_type_pairs']}

def ent_aliases(vid):
    e = ents.get(vid, {})
    al = set()
    for v in [e.get('aliases'), e.get('props', {}).get('aliases')]:
        if v:
            for a in re.split(r'[;|,]', str(v)):
                if norm(a):
                    al.add(norm(a))
    n = norm(e.get('name') or e.get('props', {}).get('name'))
    if n:
        al.add(n)
    return al

alias_index = defaultdict(list)
for vid in ents:
    for a in ent_aliases(vid):
        alias_index[a].append(vid)

pairs = set()
for a, vids in alias_index.items():
    if len(vids) < 2 or len(vids) > 20:
        continue
    tv = sorted(vids)
    for i in range(len(tv)):
        for j in range(i + 1, len(tv)):
            v1, v2 = tv[i], tv[j]
            if ents[v1].get('tag') != ents[v2].get('tag'):
                continue
            key = frozenset((v1, v2))
            if key in cand_set or key in pairs:
                continue
            pairs.add(key)

def categorize(v1, v2):
    n1 = norm(ents[v1].get('name'))
    n2 = norm(ents[v2].get('name'))
    if n1 == n2:
        return 'cat1_format_variant'
    if n1 in n2 or n2 in n1:
        return 'cat2_containment'
    return 'cat3_alias_only'

cats = defaultdict(list)
for key in pairs:
    v1, v2 = sorted(key)
    cats[categorize(v1, v2)].append((v1, v2))

random.seed(42)
sample = []
# Initial draw: 40/40/20 (100 pairs) — exactly reproducible with this seed.
# cat3 was later extended +30 in a follow-up review round (final 40/40/50 = 130).
# Authoritative record of all 130 sampled pairs:
#   results/blocking_pair_completeness.json -> verdicts (vid_a/vid_b/cat).
quota = {'cat1_format_variant': 40, 'cat2_containment': 40, 'cat3_alias_only': 20}
for c, q in quota.items():
    pool = sorted(cats[c])
    random.shuffle(pool)
    sample += [(c, v1, v2) for v1, v2 in pool[:q]]

def desc(vid):
    e = ents[vid]
    al = e.get('props', {}).get('aliases') or e.get('aliases') or ''
    return f"{vid} | {e.get('name')} | aliases: {al}" if al else f"{vid} | {e.get('name')}"

out = {'totals': {c: len(v) for c, v in cats.items()}, 'total_pairs': len(pairs),
       'sample_quota': quota}
json.dump(out, open(os.path.join(BASE, 'results', 'alias_screening.json'), 'w'), indent=1, ensure_ascii=False)

lines = ['# 别名探针403对：分层抽样人工核验清单（样本100）', '',
         f"类别规模：cat1格式变体 {len(cats['cat1_format_variant'])} | cat2包含关系 {len(cats['cat2_containment'])} | cat3仅别名共享 {len(cats['cat3_alias_only'])}", '',
         '判定标准：same = 同一真实世界实体；different = 不同实体；unsure = 证据不足', '']
for i, (c, v1, v2) in enumerate(sample, 1):
    lines.append(f"## {i}. [{c}] {ents[v1].get('tag')}")
    lines.append(f"- A: {desc(v1)}")
    lines.append(f"- B: {desc(v2)}")
    lines.append(f"- 判定: ")
open(os.path.join(BASE, 'results', 'alias_sample_review.md'), 'w', encoding='utf-8').write('\n'.join(lines))
print(json.dumps(out, indent=1))
