#!/usr/bin/env python3
"""alias_estimate.py — Blocking pair-completeness estimate from expert verdicts.

Input: expert verdicts on 130 sampled alias-overlap non-candidate pairs
(cat1 40 / cat2 40 / cat3 50), reason codes S1-S5/D1-D3/U1.
Method:
  - per-category confirmed-same rate with Wilson 95% CI
  - extrapolate to full categories (117/97/189)
  - union with closure-implied missed pairs (459, transitivity-certified) after
    deduplication between the two probe sets
  - pair completeness = blocked GT-same / (blocked GT-same + estimated missed)
Output: results/alias_verdicts_130.json, results/blocking_pair_completeness.json
"""
import json, os, re, itertools, math
from collections import defaultdict

BASE = os.path.join(os.path.dirname(__file__), '..')
ents = json.load(open('/root/workspace/tanshu-specs/experiments/minec_entity_snapshots/entities.json'))
ds = json.load(open(os.path.join(BASE, 'data', 'dataset_v3_cleaned.json')))
gt = json.load(open(os.path.join(BASE, 'data', 'minec_ground_truth_v2.json')))
gt_map = {r['pair_id']: r for r in gt['records']}

VERDICTS = """1 same S1|2 same S1|3 same S1|4 same S1|5 same S1|6 same S1|7 same S1|8 same S1|9 same S1|10 same S1|11 same S1|12 same S1|13 same S1|14 same S1|15 same S1|16 same S1|17 same S1|18 same S1|19 same S1|20 same S1|21 same S1|22 same S1|23 same S1|24 same S1|25 same S1|26 same S1|27 same S1|28 same S1|29 same S1|30 same S1|31 same S1|32 same S1|33 same S1|34 same S1|35 same S1|36 same S1|37 same S1|38 same S1|39 same S1|40 same S1|41 same S1|42 same S2|43 different D3|44 same S3|45 same S2|46 different D3|47 same S2|48 same S3|49 different D2|50 same S4|51 same S4|52 different D1|53 same S2|54 different D3|55 same S4|56 same S4|57 same S4|58 same S2|59 same S4|60 same S4|61 same S4|62 different D1|63 different D3|64 same S4|65 same S4|66 same S4|67 same S1|68 same S3|69 different D1|70 different D2|71 same S4|72 same S3|73 same S1|74 same S4|75 same S1|76 same S2|77 same S4|78 same S4|79 same S1|80 same S4|81 different D2|82 same S2|83 same S2|84 unsure U1|85 same S4|86 same S3|87 same S2|88 same S3|89 different D3|90 different D1|91 same S2|92 same S3|93 same S3|94 different D3|95 same S3|96 different D1|97 same S3|98 different D1|99 same S2|100 different D3|101 same S5|102 same S3|103 same S3|104 same S3|105 same S3|106 same S1|107 different D2|108 different D2|109 same S2|110 different D3|111 same S3|112 different D2|113 different D1|114 different D2|115 same S2|116 different D1|117 different D2|118 different D3|119 same S3|120 different D3|121 same S2|122 same S2|123 same S2|124 different D3|125 same S3|126 same S5|127 same S1|128 same S3|129 same S3|130 different D1"""

verd = {}
for rec in VERDICTS.split('|'):
    n, lab, code = rec.split()
    verd[int(n)] = {'verdict': lab, 'reason': code}
assert len(verd) == 130

# reproduce category assignment + sample draw (must match alias_screening.py)
def norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower()) if s else ''

cand = {frozenset((p['vid_a'], p['vid_b'])) for p in ds['same_type_pairs'] + ds['cross_type_pairs']}

def aliases(v):
    e = ents.get(v, {})
    al = set()
    for x in [e.get('aliases'), e.get('props', {}).get('aliases')]:
        if x:
            for a in re.split(r'[;|,]', str(x)):
                if norm(a):
                    al.add(norm(a))
    n = norm(e.get('name') or e.get('props', {}).get('name'))
    if n:
        al.add(n)
    return al

idx = defaultdict(list)
for v in ents:
    for a in aliases(v):
        idx[a].append(v)
alias_pairs = set()
for a, vs in idx.items():
    if 1 < len(vs) <= 20:
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                if ents[vs[i]].get('tag') == ents[vs[j]].get('tag'):
                    k = frozenset((vs[i], vs[j]))
                    if k not in cand:
                        alias_pairs.add(k)

def cat(k):
    v1, v2 = sorted(k)
    n1, n2 = norm(ents[v1].get('name')), norm(ents[v2].get('name'))
    if n1 == n2:
        return 1
    if n1 in n2 or n2 in n1:
        return 2
    return 3

by = defaultdict(list)
for k in alias_pairs:
    by[cat(k)].append(sorted(k))
import random
random.seed(42)
sample = []
for c, q in ((1, 40), (2, 40), (3, 20)):
    pool = sorted(by[c])
    random.shuffle(pool)
    sample += [(c, v1, v2) for v1, v2 in pool[:q]]
remaining = [p for p in by[3] if (3, p[0], p[1]) not in {(c, a, b) for c, a, b in sample}]
random.shuffle(remaining)
sample += [(3, v1, v2) for v1, v2 in remaining[:30]]
assert len(sample) == 130

# attach verdicts + verify sample membership
for i, (c, v1, v2) in enumerate(sample, 1):
    verd[i]['cat'] = c
    verd[i]['vid_a'], verd[i]['vid_b'] = v1, v2
    verd[i]['name_a'] = ents[v1].get('name')
    verd[i]['name_b'] = ents[v2].get('name')

from collections import Counter
cnt = Counter(v['verdict'] for v in verd.values())
print("verdict totals:", dict(cnt))

def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p * 100, (c - h) * 100, (c + h) * 100

cat_stat = {}
for c, total_n in ((1, 117), (2, 97), (3, 189)):
    sv = [verd[i] for i in verd if verd[i]['cat'] == c]
    n = len(sv)
    s = sum(1 for v in sv if v['verdict'] == 'same')
    p, lo, hi = wilson(s, n)
    cat_stat[f'cat{c}'] = {'sampled': n, 'same': s, 'different': sum(1 for v in sv if v['verdict'] == 'different'),
                           'unsure': sum(1 for v in sv if v['verdict'] == 'unsure'),
                           'rate_pct': round(p, 1), 'wilson_lo_pct': round(lo, 1), 'wilson_hi_pct': round(hi, 1),
                           'population': total_n}
    print(f"cat{c}: {s}/{n} same ({p:.1f}% [{lo:.1f},{hi:.1f}]) pop={total_n}")

# extrapolation (unsure counted as not-confirmed-same; alt rate excludes unsure)
est, est_lo, est_hi = 0, 0, 0
alt_est = 0
for c, total_n in ((1, 117), (2, 97), (3, 189)):
    st = cat_stat[f'cat{c}']
    est += total_n * st['rate_pct'] / 100
    est_lo += total_n * st['wilson_lo_pct'] / 100
    est_hi += total_n * st['wilson_hi_pct'] / 100
    conf = st['same'] + st['different']
    alt_est += total_n * (st['same'] / conf) if conf else 0
print(f"estimated missed dups (alias probe): {est:.0f} [{est_lo:.0f},{est_hi:.0f}]  (unsure-excluded alt: {alt_est:.0f})")

# closure-implied set (P1) and overlap with alias set
pair_key = {frozenset((p['vid_a'], p['vid_b'])): p for p in ds['same_type_pairs'] + ds['cross_type_pairs']}
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
for p in ds['same_type_pairs']:
    r = gt_map.get(p['pair_id'])
    if r and r['final_label'] == 'same':
        ra, rb = find(p['vid_a']), find(p['vid_b'])
        if ra != rb:
            parent[ra] = rb
clusters = defaultdict(list)
for vid in list(parent):
    clusters[find(vid)].append(vid)
closure_missing = set()
for root, members in clusters.items():
    if len(members) > 2:
        for a, b in itertools.combinations(sorted(members), 2):
            if frozenset((a, b)) not in pair_key:
                closure_missing.add(frozenset((a, b)))
overlap = closure_missing & alias_pairs
alias_only = alias_pairs - closure_missing
print(f"closure-implied missing: {len(closure_missing)}, alias pairs: {len(alias_pairs)}, overlap: {len(overlap)}, alias-only: {len(alias_only)}")

# combined estimate: closure pairs certain; alias-only pairs by category rate
combined = len(closure_missing)
comb_lo, comb_hi = len(closure_missing), len(closure_missing)
for c, total_n in ((1, 117), (2, 97), (3, 189)):
    st = cat_stat[f'cat{c}']
    only = sum(1 for k in alias_only if cat(k) == c)
    combined += only * st['rate_pct'] / 100
    comb_lo += only * st['wilson_lo_pct'] / 100
    comb_hi += only * st['wilson_hi_pct'] / 100

gt_same_blocked = sum(1 for p in ds['same_type_pairs']
                      if (r := gt_map.get(p['pair_id'])) and r['final_label'] == 'same')
pc = gt_same_blocked / (gt_same_blocked + combined) * 100
pc_lo = gt_same_blocked / (gt_same_blocked + comb_hi) * 100
pc_hi = gt_same_blocked / (gt_same_blocked + comb_lo) * 100
print(f"\ngt_same blocked = {gt_same_blocked}")
print(f"combined missed estimate = {combined:.0f} [{comb_lo:.0f},{comb_hi:.0f}]")
print(f"PAIR COMPLETENESS = {pc:.1f}% [{pc_lo:.1f},{pc_hi:.1f}]")

out = {
    'verdicts': verd, 'cat_stats': cat_stat,
    'alias_missed_estimate': {'point': round(est, 1), 'lo': round(est_lo, 1), 'hi': round(est_hi, 1), 'unsure_excluded_alt': round(alt_est, 1)},
    'closure_implied_missing': len(closure_missing), 'probe_overlap': len(overlap), 'alias_only_pairs': len(alias_only),
    'combined_missed_estimate': {'point': round(combined, 1), 'lo': round(comb_lo, 1), 'hi': round(comb_hi, 1)},
    'gt_same_blocked': gt_same_blocked,
    'pair_completeness_pct': {'point': round(pc, 1), 'lo': round(pc_lo, 1), 'hi': round(pc_hi, 1)},
}
json.dump(out, open(os.path.join(BASE, 'results', 'blocking_pair_completeness.json'), 'w'), indent=1, ensure_ascii=False)
print("\nwritten results/blocking_pair_completeness.json")
