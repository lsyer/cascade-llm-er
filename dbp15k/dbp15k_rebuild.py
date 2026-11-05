#!/usr/bin/env python3
"""dbp15k_rebuild.py — Deterministic rebuild of the DBP15K transfer set + empirical cascade.

Fixes vs original run_dbp15k_hierarchy.py:
  - pos sample drawn from sorted(gold_set) (original used list(gold_set): order
    depends on PYTHONHASHSEED -> irreproducible)
  - identical feature extraction, thresholds (0.5/0.3), LR config (KFold 5 rs=42,
    l1_score excluded from training)
  - empirical pipeline replay: escalated pairs get GLM-5-Turbo verdicts from the
    existing checkpoint; missing pairs are collected for new API calls
Stages:
  --dry-run : build set, replay L1, report checkpoint coverage + missing list
  --full    : after missing calls are run (dbp15k_missing_l2.json), emit final table
"""
import json, os, re, random, argparse
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'JAPE', 'data', 'dbp15k', 'zh_en')
RES = os.path.join(BASE, 'results')

def normalize_name(name):
    name = name.lower().strip()
    name = re.sub(r'[_\-()]', ' ', name)
    return re.sub(r'\s+', ' ', name)

def load_labels(fp):
    d = {}
    for line in open(fp):
        p = line.strip().split('\t')
        if len(p) >= 3: d[p[0]] = {'label': p[1], 'trans': p[2]}
        elif len(p) == 2: d[p[0]] = {'label': p[1], 'trans': p[1]}
    return d

def load_ILLs(fp):
    return set(tuple(l.strip().split('\t')) for l in open(fp) if len(l.strip().split('\t')) == 2)

def load_attrs(fp):
    d = {}
    for line in open(fp):
        p = line.strip().split('\t')
        if len(p) >= 2: d[p[0]] = set(x.split('/')[-1].lower() for x in p[1:])
    return d

def load_rel_triples(fp):
    d = defaultdict(list)
    for line in open(fp):
        p = line.strip().split('\t')
        if len(p) == 3: d[p[0]].append((p[1].split('/')[-1], p[2].split('/')[-1]))
    return d

zh_labels = load_labels(os.path.join(DATA, 's_labels'))
en_labels = load_labels(os.path.join(DATA, 't_labels'))
gold_set = load_ILLs(os.path.join(DATA, 'ent_ILLs'))
zh_attrs = load_attrs(os.path.join(DATA, 'training_attrs_1'))
en_attrs = load_attrs(os.path.join(DATA, 'training_attrs_2'))
zh_triples = load_rel_triples(os.path.join(DATA, 's_triples'))
en_triples = load_rel_triples(os.path.join(DATA, 't_triples'))

random.seed(42)
pos_sample = random.sample(sorted(gold_set), 2000)
en_name_index = defaultdict(list)
for uri in list(en_labels.keys()):
    label = en_labels.get(uri, {}).get('label', '')
    n = normalize_name(label)
    if n:
        first = n.split()[0][:3] if n.split() else n[:3]
        en_name_index[first].append(uri)
neg_pairs = []
attempts = 0
zh_uris = list(zh_labels.keys())
while len(neg_pairs) < 2000 and attempts < 50000:
    attempts += 1
    zh_uri = random.choice(zh_uris)
    if zh_uri not in zh_labels:
        continue
    trans = zh_labels[zh_uri].get('trans', '')
    n = normalize_name(trans)
    if not n:
        continue
    prefix = n.split()[0][:3] if n.split() else n[:3]
    candidates = en_name_index.get(prefix, [])
    if not candidates:
        continue
    en_uri = random.choice(candidates)
    if (zh_uri, en_uri) not in gold_set:
        neg_pairs.append((zh_uri, en_uri))
all_pairs = [(z, e, 1) for z, e in pos_sample] + [(z, e, 0) for z, e in neg_pairs]
print(f"set: {len(all_pairs)} pairs ({len(pos_sample)} pos + {len(neg_pairs)} neg), attempts={attempts}")

def extract_features(zh_uri, en_uri):
    f = {}
    zh_trans = zh_labels.get(zh_uri, {}).get('trans', '')
    en_label = en_labels.get(en_uri, {}).get('label', '')
    n1, n2 = normalize_name(zh_trans), normalize_name(en_label)
    t1, t2 = set(n1.split()), set(n2.split())
    f['name_overlap'] = len(t1 & t2) / max(len(t1 | t2), 1) if t1 and t2 else 0.0
    f['name_exact'] = 1.0 if (n1 == n2 and n1) else 0.0
    f['name_contains'] = 1.0 if (n1 and n2 and (n1 in n2 or n2 in n1)) else 0.0
    za, ea = zh_attrs.get(zh_uri, set()), en_attrs.get(en_uri, set())
    f['attr_overlap'] = len(za & ea) / max(len(za | ea), 1) if za and ea else 0.0
    f['attr_count_diff'] = abs(len(za) - len(ea))
    zd, ed = len(zh_triples.get(zh_uri, [])), len(en_triples.get(en_uri, []))
    f['degree_ratio'] = min(zd, ed) / max(zd, ed) if zd > 0 and ed > 0 else 0.0
    f['degree_diff'] = abs(zd - ed)
    f['zh_degree'] = zd
    f['en_degree'] = ed
    zl = zh_uri.split('/')[-1].lower()
    el = en_uri.split('/')[-1].lower()
    zl = re.sub(r'%[0-9a-fA-F]{2}', '_', zl)
    el = re.sub(r'%[0-9a-fA-F]{2}', '_', el)
    ut1, ut2 = set(re.split(r'[_\-]', zl)), set(re.split(r'[_\-]', el))
    f['uri_overlap'] = len(ut1 & ut2) / max(len(ut1 | ut2), 1) if ut1 and ut2 else 0.0
    f['zh_attr_count'] = len(za)
    f['en_attr_count'] = len(ea)
    f['shared_attr_ratio'] = (len(za & ea) / min(len(za), len(ea))) if (za and ea) else 0.0
    f['l1_score'] = (f['name_overlap'] * 0.40 + f['attr_overlap'] * 0.35 +
                     f['degree_ratio'] * 0.15 + f['uri_overlap'] * 0.10)
    if f['attr_overlap'] == 0 and (zd > 5 or ed > 5):
        f['l1_score'] *= 0.7
    return f

FN = ['name_overlap', 'name_exact', 'name_contains', 'attr_overlap', 'attr_count_diff',
      'degree_ratio', 'degree_diff', 'zh_degree', 'en_degree', 'uri_overlap',
      'zh_attr_count', 'en_attr_count', 'shared_attr_ratio', 'l1_score']
feats = [extract_features(z, e) for z, e, _ in all_pairs]
y = np.array([l for _, _, l in all_pairs])
X = np.array([[f[k] for k in FN] for f in feats])

MERGE_TH, REJECT_TH = 0.5, 0.3
l1s = X[:, FN.index('l1_score')]
TRAIN = [f for f in FN if f != 'l1_score']
Xt = X[:, [FN.index(f) for f in TRAIN]]
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.full(len(y), np.nan)
for tr, te in kf.split(X):
    lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    lr.fit(Xt[tr], y[tr])
    oof[te] = lr.predict_proba(Xt[te])[:, 1]

def l1_stats(scores, name):
    pred = np.where(scores >= MERGE_TH, 1, np.where(scores < REJECT_TH, 0, -1))
    dec = pred != -1
    acc = (pred[dec] == y[dec]).mean()
    print(f"{name}: L1 acc={acc:.4f} intercept={dec.mean():.4f} escalated={(~dec).sum()}")
    return pred, acc, dec.mean()

fpred, facc, fint = l1_stats(l1s, 'fixed   ')
apred, aacc, aint = l1_stats(oof, 'adaptive')

fix_esc_idx = [i for i in range(len(y)) if fpred[i] == -1]
adp_esc_idx = [i for i in range(len(y)) if apred[i] == -1]
union_esc = sorted(set(fix_esc_idx) | set(adp_esc_idx))
print(f"escalated: fixed {len(fix_esc_idx)}, adaptive {len(adp_esc_idx)}, union {len(union_esc)}")

# checkpoint coverage
res = json.load(open(os.path.join(RES, 'dbp15k_glm5turbo_results.json')))
ck = {}
for p in res['predictions']:
    ck[(p['z'], p['e'])] = p['l2_prediction']

def parse_l2(v):
    v = str(v).strip().lower()
    if v in ('same', '1', 'yes', 'match'): return 1
    if v in ('different', '0', 'no'): return 0
    return -1

missing = []
cov_f = cov_a = 0
for i in fix_esc_idx:
    z, e, _ = all_pairs[i]
    if parse_l2(ck.get((z, e), '')) != -1: cov_f += 1
    else: missing.append((z, e))
for i in adp_esc_idx:
    z, e, _ = all_pairs[i]
    if parse_l2(ck.get((z, e), '')) != -1: cov_a += 1
    elif (z, e) not in missing: missing.append((z, e))
print(f"checkpoint coverage: fixed {cov_f}/{len(fix_esc_idx)}, adaptive {cov_a}/{len(adp_esc_idx)}")
print(f"NEW L2 CALLS NEEDED: {len(missing)}")

# save state for --full stage
np.savez(os.path.join(RES, 'dbp15k_rebuild_state.npz'), l1s=l1s, oof=oof, y=y, fpred=fpred, apred=apred)
json.dump({'all_pairs': all_pairs, 'fix_esc_idx': fix_esc_idx, 'adp_esc_idx': adp_esc_idx,
           'union_esc_idx': union_esc, 'missing': missing,
           'l1_stats': {'fixed': {'acc': float(facc), 'intercept': float(fint)},
                        'adaptive': {'acc': float(aacc), 'intercept': float(aint)}}},
          open(os.path.join(RES, 'dbp15k_rebuild_state.json'), 'w'), ensure_ascii=False, indent=1)
print("state saved")
