#!/usr/bin/env python3
"""
DBP15K L2 expanded: 500 pairs with all 227 negatives + 273 positives.
Prioritize harder pairs (higher L1 score = more ambiguous).
Uses GLM-5.2 with /api/coding/paas/v4 endpoint.
"""
import json, re, random, time, urllib.request, urllib.error
from pathlib import Path
from collections import Counter

OUT = Path("/root/workspace/tanshu_docs/experiments/dbp15k")
DATA = OUT / "JAPE" / "data" / "dbp15k" / "zh_en"

# ─── Load same data as L1 script ───
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
    from collections import defaultdict
    d = defaultdict(list)
    for line in open(fp):
        p = line.strip().split('\t')
        if len(p) == 3: d[p[0]].append((p[1].split('/')[-1], p[2].split('/')[-1]))
    return d

def norm(s):
    s = s.lower().strip()
    s = re.sub(r'[_\-\(\)]', ' ', s)
    return re.sub(r'\s+', ' ', s)

print("Loading DBP15K data...")
zh_labels = load_labels(DATA / "s_labels")
en_labels = load_labels(DATA / "t_labels")
gold = load_ILLs(DATA / "ent_ILLs")
zh_attrs = load_attrs(DATA / "training_attrs_1")
en_attrs = load_attrs(DATA / "training_attrs_2")
zh_tris = load_triples(DATA / "s_triples")
en_tris = load_triples(DATA / "t_triples")

random.seed(42)
pos = random.sample(list(gold), 2000)
zh_uris = list(zh_labels.keys())
en_uris = list(en_labels.keys())
from collections import defaultdict
idx = defaultdict(list)
for u in en_uris:
    n = norm(en_labels.get(u, {}).get('label', ''))
    if n:
        pre = n.split()[0][:3] if n.split() else n[:3]
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

# Score all pairs
print("Scoring all 4000 pairs...")
all_pairs = []
for z, e in pos:
    s = l1_score(z, e)
    all_pairs.append({'z': z, 'e': e, 'score': s, 'label': 1})
for z, e in neg:
    s = l1_score(z, e)
    all_pairs.append({'z': z, 'e': e, 'score': s, 'label': 0})

# Get escalated pairs (score between -0.3 and 0.5)
REJECT_TH = 0.3
MERGE_TH = 0.5
escalated = [p for p in all_pairs if p['score'] < MERGE_TH and p['score'] >= -REJECT_TH]
esc_pos = sorted([p for p in escalated if p['label'] == 1], key=lambda x: -x['score'])
esc_neg = sorted([p for p in escalated if p['label'] == 0], key=lambda x: -x['score'])

print(f"Escalated: {len(escalated)} ({len(esc_pos)} pos, {len(esc_neg)} neg)")

# Sample: 250 hardest negatives + 250 hardest positives (by L1 score, highest = most ambiguous)
N_PER = 250
sample_neg = esc_neg[:N_PER]
sample_pos = esc_pos[:N_PER]
sample = sample_neg + sample_pos
random.shuffle(sample)
print(f"Escalated: {len(esc_pos)} pos, {len(esc_neg)} neg")
print(f"Sampled: {len(sample)} ({sum(p['label'] for p in sample)} pos, {sum(1-p['label'] for p in sample)} neg)")

# ─── GLM API ───
_env = {}
with open("/root/.hermes/.env") as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            _env[k] = v
GLM_API_KEY = _env.get("GLM_API_KEY", "")
GLM_URL = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
GLM_MODEL = "glm-5.2"

def build_prompt(pair):
    z, e = pair['z'], pair['e']
    zl = zh_labels.get(z, {})
    el = en_labels.get(e, {})
    zt = zl.get('trans', zl.get('label', z))
    el_name = el.get('label', e)
    za = zh_attrs.get(z, set())
    ea = en_attrs.get(e, set())
    shared = za & ea
    only_z = za - ea
    only_e = ea - za
    zd = len(zh_tris.get(z, []))
    ed = len(en_tris.get(e, []))

    prompt = f"""You are an expert in cross-lingual entity alignment. Determine whether the following two entities from different language editions of DBpedia refer to the SAME real-world entity.

Entity A (Chinese DBpedia): {zt}
Entity B (English DBpedia): {el_name}

Shared attributes ({len(shared)}): {', '.join(sorted(list(shared))[:10])}
Attributes only in A ({len(only_z)}): {', '.join(sorted(list(only_z))[:10])}
Attributes only in B ({len(only_e)}): {', '.join(sorted(list(only_e))[:10])}
Relation degree: A={zd}, B={ed}

Answer with exactly one word: MATCH or DIFFERENT."""
    return prompt

def call_glm(prompt, retries=5):
    body = json.dumps({
        "model": GLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 16384
    }).encode()
    for attempt in range(retries):
        req = urllib.request.Request(GLM_URL, data=body)
        req.add_header("Authorization", f"Bearer {GLM_API_KEY}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                content = data['choices'][0]['message']['content'].strip()
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                return content if content else "EMPTY"
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 60 * (attempt + 1)
                print(f"    429, waiting {wait}s (attempt {attempt+1}/{retries})", flush=True)
                time.sleep(wait)
            else:
                return f"ERROR: HTTP {e.code}"
        except Exception as e:
            return f"ERROR: {str(e)[:50]}"
    return "ERROR: max retries exceeded"

def parse_response(raw):
    raw_lower = raw.lower().strip()
    if 'match' in raw_lower and 'different' not in raw_lower:
        return 1
    elif 'different' in raw_lower:
        return 0
    else:
        return -1

# ─── Run ───
CKPT = OUT / "dbp15k_l2_expanded_checkpoint.json"
RESULTS = OUT / "dbp15k_l2_expanded_results.json"

# Resume from checkpoint
results = []
if CKPT.exists():
    with open(CKPT) as f:
        results = json.load(f)
    print(f"Resuming from checkpoint: {len(results)}/{len(sample)} done")

remaining = len(sample) - len(results)
print(f"\nCalling GLM-5.2 ({remaining} remaining, 5s interval)...", flush=True)

for i in range(len(results), len(sample)):
    pair = sample[i]
    prompt = build_prompt(pair)
    raw = call_glm(prompt)
    pred = parse_response(raw)
    results.append({
        'z': pair['z'], 'e': pair['e'],
        'label': pair['label'],
        'l1_score': round(pair['score'], 4),
        'l2_raw': raw[:200],
        'l2_prediction': pred,
    })

    if (i + 1) % 5 == 0:
        with open(CKPT, 'w') as f:
            json.dump(results, f, ensure_ascii=False)

    valid = [r for r in results if r['l2_prediction'] != -1]
    acc = sum(1 for r in valid if r['l2_prediction'] == r['label']) / max(len(valid), 1) * 100
    if (i + 1) % 20 == 0:
        print(f"  [{i+1}/{len(sample)}] acc={acc:.1f}% ({len(valid)} valid)", flush=True)

    time.sleep(5)

# ─── Compute results ───
valid = [r for r in results if r['l2_prediction'] != -1]
errors = sum(1 for r in results if r['l2_prediction'] == -1)
tp = sum(1 for r in valid if r['l2_prediction'] == 1 and r['label'] == 1)
fp = sum(1 for r in valid if r['l2_prediction'] == 1 and r['label'] == 0)
tn = sum(1 for r in valid if r['l2_prediction'] == 0 and r['label'] == 0)
fn = sum(1 for r in valid if r['l2_prediction'] == 0 and r['label'] == 1)

pos_n = tp + fn
neg_n = fp + tn
acc = (tp + tn) / max(len(valid), 1) * 100
pos_acc = tp / max(pos_n, 1) * 100
neg_acc = tn / max(neg_n, 1) * 100
precision = tp / max(tp + fp, 1) * 100
recall = tp / max(tp + fn, 1) * 100
f1 = 2 * precision * recall / max(precision + recall, 1)

print(f"\n=== L2 Results (GLM-5.2, n={len(valid)} valid / {errors} errors) ===", flush=True)
print(f"Accuracy:  {acc:.1f}%", flush=True)
print(f"  Positive (MATCH): {tp}/{pos_n} = {pos_acc:.1f}%", flush=True)
print(f"  Negative (DIFF):  {tn}/{neg_n} = {neg_acc:.1f}%", flush=True)
print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}", flush=True)
print(f"  Precision: {precision:.1f}%  Recall: {recall:.1f}%  F1: {f1:.1f}%", flush=True)

summary = {
    'l2_accuracy': acc / 100,
    'l2_errors': errors,
    'total_pairs': len(sample),
    'valid_pairs': len(valid),
    'positive_pairs': pos_n,
    'negative_pairs': neg_n,
    'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    'positive_accuracy': pos_acc / 100,
    'negative_accuracy': neg_acc / 100,
    'precision': precision / 100,
    'recall': recall / 100,
    'f1': f1 / 100,
    'sampling_strategy': 'all_negatives + hardest_positives (by L1 score)',
    'model': 'glm-5.2',
}

with open(RESULTS, 'w') as f:
    json.dump({'summary': summary, 'predictions': results}, f, ensure_ascii=False, indent=2)

print(f"\nSaved to {RESULTS}", flush=True)
