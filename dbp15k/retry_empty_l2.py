#!/usr/bin/env python3
"""retry_empty_l2.py — one retry pass for EMPTY L2 verdicts (temp 0.3)."""
import json, os, re, time, urllib.request
from pathlib import Path
from collections import defaultdict

DATA = Path(__file__).resolve().parent / 'JAPE' / 'data' / 'dbp15k' / 'zh_en'
_env = {}
for line in open(os.environ.get('ENV_FILE', '/root/.hermes/.env')):
    if '=' in line:
        k, v = line.strip().split('=', 1); _env[k] = v
API_KEY = _env['GLM_API_KEY']
API_URL = 'https://open.bigmodel.cn/api/coding/paas/v4/chat/completions'

def load_labels(fp):
    d = {}
    for line in open(fp):
        p = line.strip().split('\t')
        if len(p) >= 3: d[p[0]] = {'label': p[1], 'trans': p[2]}
        elif len(p) == 2: d[p[0]] = {'label': p[1], 'trans': p[1]}
    return d

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

zh_labels = load_labels(DATA / 's_labels'); en_labels = load_labels(DATA / 't_labels')
zh_attrs = load_attrs(DATA / 'training_attrs_1'); en_attrs = load_attrs(DATA / 'training_attrs_2')
zh_tris = load_triples(DATA / 's_triples'); en_tris = load_triples(DATA / 't_triples')

def build_prompt(z, e):
    zl = zh_labels.get(z, {}); el = en_labels.get(e, {})
    zt = zl.get('trans', zl.get('label', z)); el_name = el.get('label', e)
    za = zh_attrs.get(z, set()); ea = en_attrs.get(e, set())
    shared = za & ea; only_z = za - ea; only_e = ea - za
    zd = len(zh_tris.get(z, [])); ed = len(en_tris.get(e, []))
    return f"""You are an expert in cross-lingual entity alignment. Determine whether the following two entities from different language editions of DBpedia refer to the SAME real-world entity.

Entity A (Chinese DBpedia): {zt}
Entity B (English DBpedia): {el_name}

Shared attributes ({len(shared)}): {', '.join(sorted(list(shared))[:10])}
Attributes only in A ({len(only_z)}): {', '.join(sorted(list(only_z))[:10])}
Attributes only in B ({len(only_e)}): {', '.join(sorted(list(only_e))[:10])}
Relation degree: A={zd}, B={ed}

Answer with exactly one word: MATCH or DIFFERENT."""

def call(prompt):
    body = json.dumps({'model': 'glm-5-turbo', 'messages': [{'role': 'user', 'content': prompt}],
                       'temperature': 0.3, 'max_tokens': 4096}).encode()
    req = urllib.request.Request(API_URL, data=body)
    req.add_header('Authorization', f'Bearer {API_KEY}')
    req.add_header('Content-Type', 'application/json')
    with urllib.request.urlopen(req, timeout=180) as r:
        c = json.loads(r.read())['choices'][0]['message']['content'].strip()
        return re.sub(r'<think>.*?</think>', '', c, flags=re.DOTALL).strip()

CK = Path(__file__).resolve().parent / 'results' / 'dbp15k_rebuild_l2.json'
ck = json.load(open(CK))
empties = [k for k, v in ck.items() if v['raw'] == 'EMPTY']
print(f'retrying {len(empties)}', flush=True)
fixed = 0
for k in empties:
    z, e = k.split('||', 1)
    try:
        raw = call(build_prompt(z, e))
    except Exception as ex:
        raw = f'ERROR: {str(ex)[:60]}'
    if raw and not raw.startswith('ERROR'):
        ck[k]['raw'] = raw
        fixed += 1
        json.dump(ck, open(CK, 'w'), ensure_ascii=False)
    print(f'{fixed} recovered so far; last={raw[:30]!r}', flush=True)
json.dump(ck, open(CK, 'w'), ensure_ascii=False)
print(f'DONE recovered {fixed}/{len(empties)}', flush=True)
