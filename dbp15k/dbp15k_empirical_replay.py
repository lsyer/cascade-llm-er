#!/usr/bin/env python3
"""dbp15k_empirical_replay.py — Replace formula-based DBP15K pipeline numbers with
true end-to-end empirical cascade replay (R2 detailed comment #2).

Steps:
 1. Reproduce the hierarchy pair set (4,000 pairs) and OOF LR probabilities;
    verify against stored fold_results (tp/fp/tn/fn/escalated per fold).
 2. Determine escalated sets: fixed (l1_score in [0.3,0.5)) and adaptive
    (OOF proba in [0.3,0.5)).
 3. Check L2 checkpoint coverage ((z,e) -> verdict) for both escalated sets.
 4. Replay final verdicts: intercepted -> L1 label, escalated -> L2 verdict,
    missing verdict -> unresolved (strict: wrong).
Output: results/dbp15k_empirical_pipeline.json + coverage report.
"""
import json, os, re, random
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'JAPE', 'data', 'dbp15k', 'zh_en')

def norm(s):
    s = s.lower().strip()
    s = re.sub(r'[_\-()]', ' ', s)
    return re.sub(r'\s+', ' ', s)

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

def load_triples(fp):
    d = defaultdict(list)
    for line in open(fp):
        p = line.strip().split('\t')
        if len(p) == 3: d[p[0]].append((p[1].split('/')[-1], p[2].split('/')[-1]))
    return d

zh_labels = load_labels(os.path.join(DATA, 's_labels'))
en_labels = load_labels(os.path.join(DATA, 't_labels'))
gold = load_ILLs(os.path.join(DATA, 'ent_ILLs'))
zh_attrs = load_attrs(os.path.join(DATA, 'training_attrs_1'))
en_attrs = load_attrs(os.path.join(DATA, 'training_attrs_2'))
zh_tris = load_triples(os.path.join(DATA, 's_triples'))
en_tris = load_triples(os.path.join(DATA, 't_triples'))

random.seed(42)
pos = random.sample(sorted(gold), 2000)
idx = defaultdict(list)
for eu in list(en_labels.keys()):
    prefix = norm(en_labels[eu].get('label', eu))[:3]
    idx[prefix].append(eu)
neg = []
for z, e in pos:
    zt = norm(zh_labels[z].get('trans', zh_labels[z].get('label', z)))
    candidates = []
    for pl in (3, 2, 1):
        if zt[:pl] in idx:
            candidates = idx[zt[:pl]]
            break
    if not candidates:
        candidates = list(en_labels.keys())
    for _ in range(2):
        e_neg = random.choice(candidates)
        if (z, e_neg) not in gold and e_neg != e:
            neg.append((z, e_neg))
all_pairs = [(z, e, 1) for z, e in pos] + [(z, e, 0) for z, e in neg]
print(f"pairs reproduced: {len(all_pairs)} (pos {len(pos)}, neg {len(neg)})")

# ---- features (mirror run_dbp15k_hierarchy.extract_features) ----
def name_tokens(uri, labels, zh=False):
    if zh:
        s = labels.get(uri, {}).get('trans', labels.get(uri, {}).get('label', uri))
    else:
        s = labels.get(uri, {}).get('label', uri)
    return set(norm(s).split())

def extract_features(z, e):
    f = {}
    zt = set(norm(zh_labels[z].get('trans', zh_labels[z].get('label', z))).split())
    et = set(norm(en_labels[e].get('label', e)).split())
    f['name_overlap'] = len(zt & et) / max(len(zt | et), 1) if (zt or et) else 0.0
    f['name_exact'] = 1.0 if zt == et and zt else 0.0
    zs = norm(zh_labels[z].get('trans', zh_labels[z].get('label', z)))
    es = norm(en_labels[e].get('label', e))
    f['name_contains'] = 1.0 if (zs and es and (zs in es or es in zs)) else 0.0
    za, ea = zh_attrs.get(z, set()), en_attrs.get(e, set())
    f['attr_overlap'] = len(za & ea) / max(len(za | ea), 1) if (za or ea) else 0.0
    f['attr_count_diff'] = abs(len(za) - len(ea))
    zd, ed = len(zh_tris.get(z, [])), len(en_tris.get(e, []))
    f['degree_ratio'] = min(zd, ed) / max(zd, ed) if max(zd, ed) > 0 else 0.0
    f['degree_diff'] = abs(zd - ed)
    f['zh_degree'] = zd
    f['en_degree'] = ed
    zl = re.sub(r'%[0-9a-fA-F]{2}', '_', z.split('/')[-1].lower())
    el = re.sub(r'%[0-9a-fA-F]{2}', '_', e.split('/')[-1].lower())
    ut1 = set(re.split(r'[_\-]', zl)); ut2 = set(re.split(r'[_\-]', el))
    f['uri_overlap'] = len(ut1 & ut2) / max(len(ut1 | ut2), 1) if ut1 and ut2 else 0.0
    f['zh_attr_count'] = len(za)
    f['en_attr_count'] = len(ea)
    f['shared_attr_ratio'] = (len(za & ea) / min(len(za), len(ea))) if (za and ea) else 0.0
    f['l1_score'] = (f['name_overlap'] * 0.40 + f['attr_overlap'] * 0.35 +
                     f['degree_ratio'] * 0.15 + f['uri_overlap'] * 0.10)
    if f['attr_overlap'] == 0 and (zd > 5 or ed > 5):
        f['l1_score'] *= 0.7
    return f

feats = [extract_features(z, e) for z, e, _ in all_pairs]
y = np.array([l for _, _, l in all_pairs])
FN = ['name_overlap', 'name_exact', 'name_contains', 'attr_overlap', 'attr_count_diff',
      'degree_ratio', 'degree_diff', 'zh_degree', 'en_degree', 'uri_overlap',
      'zh_attr_count', 'en_attr_count', 'shared_attr_ratio', 'l1_score']
X = np.array([[f[k] for k in FN] for f in feats])

MERGE_TH, REJECT_TH = 0.5, 0.3
l1s = X[:, FN.index('l1_score')]
fixed_esc = [(i, all_pairs[i]) for i in range(len(y)) if REJECT_TH <= l1s[i] < MERGE_TH]
print(f"fixed escalated: {len(fixed_esc)}")

TRAIN = [f for f in FN if f != 'l1_score']
Xt = X[:, [FN.index(f) for f in TRAIN]]
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.full(len(y), np.nan)
for tr, te in kf.split(X):
    lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    lr.fit(Xt[tr], y[tr])
    oof[te] = lr.predict_proba(Xt[te])[:, 1]
assert not np.isnan(oof).any()

# verify against stored fold results
hier = json.load(open(os.path.join(BASE, 'results', 'hierarchy_feedback.json')))
ok = True
for fr in hier['level1_unified_lr']['fold_results']:
    pass  # fold index mapping unknown; verify aggregate instead
avg_acc = np.mean([np.mean([1.0]) ])  # placeholder; do direct replay comparison below

adapt_esc = [(i, all_pairs[i]) for i in range(len(y)) if REJECT_TH <= oof[i] < MERGE_TH]
print(f"adaptive escalated: {len(adapt_esc)}")

# stored aggregate check: stored level1 avg accuracy/interception
pred = np.where(oof >= MERGE_TH, 1, np.where(oof < REJECT_TH, 0, -1))
dec = pred != -1
acc = (pred[dec] == y[dec]).mean()
intercept = dec.mean()
st = hier['level1_unified_lr']
print(f"replay adaptive: acc={acc:.4f} intercept={intercept:.4f} | stored: acc={st['avg_accuracy'] if 'avg_accuracy' in st else st} ")
print("stored level1 keys:", list(st.keys()))

# fixed replay check
fpred = np.where(l1s >= MERGE_TH, 1, np.where(l1s < REJECT_TH, 0, -1))
fdec = fpred != -1
facc = (fpred[fdec] == y[fdec]).mean()
fint = fdec.mean()
l0 = hier['level0_fixed']
print(f"replay fixed: acc={facc:.4f} intercept={fint:.4f} esc={len(y)-fdec.sum()} | stored: acc={l0['accuracy']:.4f} int={l0['interception']:.4f} esc={l0['escalated']}")

# ---- checkpoint coverage ----
ck = json.load(open(os.path.join(BASE, 'results', 'dbp15k_glm5turbo_checkpoint.json')))
def parse(v):
    v = str(v).strip().lower()
    if v in ('same', '1', 'yes'): return 1
    if v in ('different', '0', 'no'): return 0
    return -1
# checkpoint structure: key unknown; inspect
k0 = list(ck)[0]
print("checkpoint sample entry:", k0[:80], '->', str(ck[k0])[:120])
