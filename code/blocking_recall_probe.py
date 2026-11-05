#!/usr/bin/env python3
"""blocking_recall_probe.py — Estimate pair completeness (blocking recall) for MINEC.

Three probes, no new LLM calls:
  P1. Transitive-closure check inside the labeled candidate set:
      if GT says A~B and B~C, then A~C is implied. Count implied pairs missing
      from the candidate set (missed by blocking), and consistency violations
      inside the set (A~C blocked but labeled different).
  P2. Cross-type same-name audit: the 19 exact-name cross-type pairs excluded
      from evaluation. Report their type combinations; flag any that could be
      true duplicates (blocking design miss).
  P3. Alias/attribute-overlap probe on non-candidates: same-type entity pairs
      sharing at least one normalized alias (or exact key attribute value:
      hull_number / imo / coordinates) that were NOT generated as candidates.
      These are potential name-disjoint duplicates missed by blocking; report
      count + samples for manual inspection.
Output: results/blocking_recall_probe.json
"""
import json, os, re, itertools
from collections import defaultdict

BASE = os.path.join(os.path.dirname(__file__), '..')
_SNAP_DIR = os.environ.get('MINEC_ENTITY_SNAPSHOTS',
                           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', '..', 'minec_entity_snapshots'))
ents = json.load(open(os.path.join(_SNAP_DIR, 'entities.json')))
ds = json.load(open(os.path.join(BASE, 'data', 'dataset_v3_cleaned.json')))
gt = json.load(open(os.path.join(BASE, 'data', 'minec_ground_truth_v2.json')))
gt_map = {r['pair_id']: r for r in gt['records']}

def norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower()) if s else ''

# ---------- P1: transitive closure ----------
pair_key = {}
for p in ds['same_type_pairs'] + ds['cross_type_pairs']:
    pair_key[frozenset((p['vid_a'], p['vid_b']))] = p

# union-find over GT-same labels
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb

n_same = n_diff = 0
for p in ds['same_type_pairs']:
    r = gt_map.get(p['pair_id'])
    if not r or r['final_label'] not in ('same', 'different'):
        continue
    if r['final_label'] == 'same':
        union(p['vid_a'], p['vid_b'])
        n_same += 1
    else:
        n_diff += 1

clusters = defaultdict(list)
for vid in list(parent):
    clusters[find(vid)].append(vid)
multi = {k: v for k, v in clusters.items() if len(v) > 2}
implied_missing = []
inconsistent = []
for root, members in multi.items():
    for a, b in itertools.combinations(sorted(members), 2):
        key = frozenset((a, b))
        if key not in pair_key:
            implied_missing.append((a, b))
        else:
            r = gt_map.get(pair_key[key]['pair_id'])
            if r and r['final_label'] == 'different':
                inconsistent.append((a, b))

# ---------- P2: cross-type audit ----------
cross = ds['cross_type_pairs']
cross_types = defaultdict(int)
for p in cross:
    cross_types[frozenset((p['type_a'], p['type_b']))] += 1

# ---------- P3: alias/attribute overlap among non-candidates ----------
# entity props: aliases, hull/imo-like keys
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

KEY_ATTRS = ['hull_number', 'imo', 'callsign', 'registry_number', 'icao', 'iata',
             'coordinates', 'lat', 'lon', 'latitude', 'longitude']
def key_attrs(vid):
    e = ents.get(vid, {})
    out = {}
    pr = e.get('props', {})
    for k in KEY_ATTRS:
        v = pr.get(k)
        if v and norm(v):
            out[k] = norm(v)
    return out

# index entities by type
by_type = defaultdict(list)
for vid, e in ents.items():
    by_type[e.get('tag', '')].append(vid)

cand_set = {frozenset((p['vid_a'], p['vid_b'])) for p in ds['same_type_pairs'] + ds['cross_type_pairs']}

alias_index = defaultdict(list)   # alias -> vids (typed)
for vid in ents:
    for a in ent_aliases(vid):
        alias_index[a].append(vid)

alias_hits = []
for a, vids in alias_index.items():
    if len(vids) < 2 or len(vids) > 20:
        continue
    for v1, v2 in itertools.combinations(sorted(vids), 2):
        e1, e2 = ents[v1], ents[v2]
        if e1.get('tag') != e2.get('tag'):
            continue  # cross-type handled in P2
        if frozenset((v1, v2)) in cand_set:
            continue
        # skip if the shared alias is just the entity's own name AND pair was fuzzy-blocked already
        alias_hits.append((v1, v2, a))

attr_index = defaultdict(list)
for vid in ents:
    for k, v in key_attrs(vid).items():
        attr_index[(k, v)].append(vid)
attr_hits = []
for (k, v), vids in attr_index.items():
    if len(vids) < 2:
        continue
    for v1, v2 in itertools.combinations(sorted(vids), 2):
        e1, e2 = ents[v1], ents[v2]
        if e1.get('tag') != e2.get('tag'):
            continue
        if frozenset((v1, v2)) in cand_set:
            continue
        attr_hits.append((v1, v2, k, v))

def dedup(hits):
    seen, out = set(), []
    for h in hits:
        key = frozenset((h[0], h[1]))
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out

alias_hits_d = dedup(alias_hits)
attr_hits_d = dedup(attr_hits)

def desc(vid):
    e = ents[vid]
    return f"{vid}({e.get('tag')}): {e.get('name')}"

report = {
    'P1_closure': {
        'gt_same_pairs': n_same, 'gt_diff_pairs': n_diff,
        'clusters_gt2': len(multi),
        'implied_pairs_missing_from_candidates': len(implied_missing),
        'inconsistent_inside_candidates': len(inconsistent),
        'missing_examples': [f"{desc(a)} <-> {desc(b)}" for a, b in implied_missing[:10]],
        'inconsistent_examples': [f"{desc(a)} <-> {desc(b)}" for a, b in inconsistent[:10]],
    },
    'P2_cross_type': {
        'n_cross_type_same_name_excluded': len(cross),
        'type_combinations': {f"{'/'.join(sorted(k))}": v for k, v in cross_types.items()},
        'examples': [f"{p['name_a']} [{p['type_a']} vs {p['type_b']}]" for p in cross[:10]],
    },
    'P3_noncandidate_overlap': {
        'alias_overlap_pairs': len(alias_hits_d),
        'alias_examples_v2': [f"{desc(v1)} <-> {desc(v2)} shared[{a[:40]}]" for v1, v2, a in alias_hits_d[:15]],
        'attr_overlap_pairs': len(attr_hits_d),
        'attr_examples': [f"{desc(v1)} <-> {desc(v2)} shared[{k}={v[:30]}]" for v1, v2, k, v in attr_hits_d[:15]],
    },
}
json.dump(report, open(os.path.join(BASE, 'results', 'blocking_recall_probe.json'), 'w'), indent=1, ensure_ascii=False)
print(json.dumps(report, indent=1, ensure_ascii=False)[:3000])
