#!/usr/bin/env python3
"""
DBP15K L2 GLM-5.2 — resilient version.
- 10s interval between calls
- 429: wait 60s, retry up to 10 times
- Checkpoint every 5 results to JSON
- Resume from checkpoint if interrupted
"""
import json, random, time, re, os, urllib.request, urllib.error
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent / "JAPE" / "data" / "dbp15k" / "zh_en"
RESULTS_DIR = Path(__file__).parent
CHECKPOINT = RESULTS_DIR / "dbp15k_l2_checkpoint.json"
FINAL = RESULTS_DIR / "dbp15k_l2_results.json"

# Load API key
_env = {}
with open("/root/.hermes/.env") as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            _env[k] = v
GLM_API_KEY = _env.get("GLM_API_KEY", "")

GLM_URL = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
GLM_MODEL = "glm-5.2"

# ─── Data ───
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

def norm(s):
    s = s.lower().strip()
    s = re.sub(r'[_\-\(\)]', ' ', s)
    return re.sub(r'\s+', ' ', s)

print("Loading data...")
zh_labels = load_labels(DATA_DIR / "s_labels")
en_labels = load_labels(DATA_DIR / "t_labels")
gold = load_ILLs(DATA_DIR / "ent_ILLs")
zh_attrs = load_attrs(DATA_DIR / "training_attrs_1")
en_attrs = load_attrs(DATA_DIR / "training_attrs_2")
zh_tris = load_triples(DATA_DIR / "s_triples")
en_tris = load_triples(DATA_DIR / "t_triples")

# Build eval set (same seed)
random.seed(42)
pos = random.sample(list(gold), 2000)
zh_uris = list(zh_labels.keys())
en_uris = list(en_labels.keys())
idx = defaultdict(list)
for u in en_uris:
    n = norm(en_labels.get(u, {}).get('label', ''))
    if n:
        pre = (n.split()[0][:3] if n.split() else n[:3])
        idx[pre].append(u)

neg = []
att = 0
while len(neg) < 2000 and att < 50000:
    att += 1
    z = random.choice(zh_uris)
    if z not in zh_labels: continue
    n = norm(zh_labels[z].get('trans', ''))
    if not n: continue
    pre = n.split()[0][:3] if n.split() else n[:3]
    cs = idx.get(pre, [])
    if not cs: continue
    e = random.choice(cs)
    if (z, e) not in gold: neg.append((z, e))

def l1_score(z, e):
    t = norm(zh_labels.get(z, {}).get('trans', ''))
    l = norm(en_labels.get(e, {}).get('label', ''))
    t1, t2 = set(t.split()), set(l.split())
    ns = len(t1 & t2) / max(len(t1 | t2), 1) if t1 and t2 else 0
    if t == l and t: ns = 1.0
    za, ea = zh_attrs.get(z, set()), en_attrs.get(e, set())
    as_ = len(za & ea) / max(len(za | ea), 1) if za and ea else 0
    zd, ed = len(zh_tris.get(z, [])), len(en_tris.get(e, []))
    ds = min(zd, ed) / max(zd, ed) if zd > 0 and ed > 0 else 0
    s = ns * 0.40 + as_ * 0.35 + ds * 0.15
    if as_ == 0 and (zd > 5 or ed > 5): s *= 0.7
    return s

esc = []
for z, e in pos:
    s = l1_score(z, e)
    if 0.3 <= s < 0.5: esc.append({'zh': z, 'en': e, 'label': 1, 'score': s})
for z, e in neg:
    s = l1_score(z, e)
    if 0.3 <= s < 0.5: esc.append({'zh': z, 'en': e, 'label': 0, 'score': s})

random.seed(123)
sample = random.sample(esc, min(200, len(esc)))
print(f"Sample: {len(sample)} ({sum(x['label'] for x in sample)} pos)")

def build_prompt(p):
    z, e = p['zh'], p['en']
    zl = zh_labels.get(z, {}).get('label', 'N/A')
    zt = zh_labels.get(z, {}).get('trans', 'N/A')
    el = en_labels.get(e, {}).get('label', 'N/A')
    za = sorted(zh_attrs.get(z, set()))[:8]
    ea = sorted(en_attrs.get(e, set()))[:8]
    zr = zh_tris.get(z, [])[:3]
    er = en_tris.get(e, [])[:3]
    return f"""You are an expert in cross-lingual entity alignment between Chinese and English DBpedia knowledge graphs.

Determine whether the following two entities refer to the same real-world entity.

Entity A (Chinese DBpedia):
- Label: {zl}
- English translation: {zt}
- Attributes: {', '.join(za) if za else 'none'}
- Relations: {', '.join(f'{r[0]}->{r[1]}' for r in zr) if zr else 'none'}

Entity B (English DBpedia):
- Label: {el}
- Attributes: {', '.join(ea) if ea else 'none'}
- Relations: {', '.join(f'{r[0]}->{r[1]}' for r in er) if er else 'none'}

Respond with ONLY one word: MATCH or DIFFERENT."""

def call_glm(prompt):
    body = json.dumps({"model": GLM_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": 16384}).encode()
    for attempt in range(10):
        req = urllib.request.Request(GLM_URL, data=body, method="POST")
        req.add_header("Authorization", f"Bearer {GLM_API_KEY}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                content = data['choices'][0]['message']['content'].strip()
                # Filter thinking tags
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                return content if content else "EMPTY"
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 60 if attempt < 5 else 120
                print(f"    429, waiting {wait}s (attempt {attempt+1}/10)", flush=True)
                time.sleep(wait)
            else:
                print(f"    HTTP {e.code}, retry in 10s", flush=True)
                time.sleep(10)
        except Exception as e:
            print(f"    Error: {e}, retry in 10s", flush=True)
            time.sleep(10)
    return "ERROR: max retries"

# ─── Resume from checkpoint ───
results = []
start_idx = 0
if CHECKPOINT.exists():
    with open(CHECKPOINT) as f:
        results = json.load(f)
    start_idx = len(results)
    print(f"Resuming from {start_idx}/{len(sample)}")

print(f"\nCalling GLM-5.2 ({len(sample) - start_idx} remaining, 10s interval)...")
for i in range(start_idx, len(sample)):
    p = sample[i]
    prompt = build_prompt(p)
    resp = call_glm(prompt)
    pred = 1 if 'MATCH' in resp.upper() and 'DIFFERENT' not in resp.upper() and 'ERROR' not in resp else 0

    results.append({
        'zh_uri': p['zh'], 'en_uri': p['en'],
        'label': p['label'], 'l1_score': round(p['score'], 4),
        'l2_prediction': pred, 'l2_raw': resp[:100],
    })

    if (i + 1) % 5 == 0:
        with open(CHECKPOINT, 'w') as f:
            json.dump(results, f, ensure_ascii=False)

    valid = [r for r in results if 'ERROR' not in r.get('l2_raw', '')]
    if valid:
        acc = sum(1 for r in valid if r['l2_prediction'] == r['label']) / len(valid)
    else:
        acc = 0

    print(f"  [{i+1}/{len(sample)}] label={p['label']} pred={pred} raw={resp[:30]} | running_acc={acc:.1%} ({len(valid)} valid)", flush=True)
    time.sleep(10)

# ─── Final evaluation ───
valid = [r for r in results if 'ERROR' not in r['l2_raw']]
errors = len(results) - len(valid)
correct = sum(1 for r in valid if r['l2_prediction'] == r['label'])
acc = correct / max(len(valid), 1)

tp = sum(1 for r in valid if r['l2_prediction'] == 1 and r['label'] == 1)
fp = sum(1 for r in valid if r['l2_prediction'] == 1 and r['label'] == 0)
tn = sum(1 for r in valid if r['l2_prediction'] == 0 and r['label'] == 0)
fn = sum(1 for r in valid if r['l2_prediction'] == 0 and r['label'] == 1)

print(f"\n=== L2 Results (GLM-5.2, n={len(valid)} valid / {errors} errors) ===")
print(f"Accuracy:  {acc:.1%}")
print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
print(f"Precision: {tp/max(tp+fp,1):.1%}  Recall: {tp/max(tp+fn,1):.1%}  F1: {2*tp/max(2*tp+fp+fn,1):.1%}")

output = {
    'dataset': 'DBP15K ZH-EN', 'l2_model': 'GLM-5.2',
    'l2_sample_size': len(valid), 'l2_errors': errors,
    'l2_accuracy': round(acc, 4),
    'l2_tp': tp, 'l2_fp': fp, 'l2_tn': tn, 'l2_fn': fn,
    'l2_precision': round(tp/max(tp+fp,1), 4),
    'l2_recall': round(tp/max(tp+fn,1), 4),
    'l2_f1': round(2*tp/max(2*tp+fp+fn,1), 4),
    'predictions': results,
}
with open(FINAL, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
CHECKPOINT.unlink(missing_ok=True)
print(f"\nSaved to {FINAL}")
