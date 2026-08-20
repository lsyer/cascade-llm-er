#!/usr/bin/env python3
"""
DBP15K L2 with GLM-5-Turbo: run ALL 808 escalated pairs (not just 500 sample).
Uses same data loading and prompt as run_dbp15k_l2_expanded.py.
5 concurrent workers.
"""
import json, re, random, time, urllib.request, urllib.error
import os
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

OUT = Path(__file__).resolve().parent
DATA = Path(__file__).resolve().parent / "JAPE" / "data" / "dbp15k" / "zh_en"
CKPT = OUT / "results" / "dbp15k_glm5turbo_checkpoint.json"
RESULTS = OUT / "results" / "dbp15k_glm5turbo_results.json"
MAX_WORKERS = 5
MODEL = "glm-5-turbo"

# ─── Load API key ───
_env = {}
with open(os.environ.get("ENV_FILE", ".env")) as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            _env[k] = v
API_KEY = _env.get("GLM_API_KEY", "")
API_URL = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"

# ─── Load data (same as original script) ───
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
    s = re.sub(r'[_\-()]', ' ', s)
    return re.sub(r'\s+', ' ', s)

def l1_score(z, e):
    """Same L1 score as original script."""
    zl = zh_labels.get(z, {})
    el = en_labels.get(e, {})
    zt = norm(zl.get('trans', zl.get('label', z)))
    el_name = norm(el.get('label', e))
    za = zh_attrs.get(z, set())
    ea = en_attrs.get(e, set())
    overlap = len(za & ea) / max(len(za | ea), 1) if (za or ea) else 0
    name_sim = 1.0 if zt == el_name else (0.5 if (zt in el_name or el_name in zt) else 0.0)
    zd = len(zh_tris.get(z, []))
    ed = len(en_tris.get(e, []))
    deg_ratio = min(zd, ed) / max(zd, ed) if max(zd, ed) > 0 else 0
    return 0.5 * name_sim + 0.3 * overlap + 0.2 * deg_ratio

print("Loading DBP15K data...", flush=True)
zh_labels = load_labels(DATA / "s_labels")
en_labels = load_labels(DATA / "t_labels")
gold = load_ILLs(DATA / "ent_ILLs")
zh_attrs = load_attrs(DATA / "training_attrs_1")
en_attrs = load_attrs(DATA / "training_attrs_2")
zh_tris = load_triples(DATA / "s_triples")
en_tris = load_triples(DATA / "t_triples")
print(f"  zh: {len(zh_labels)}, en: {len(en_labels)}, gold: {len(gold)}", flush=True)

random.seed(42)
# sorted for determinism (set order varies per process via PYTHONHASHSEED)
pos = random.sample(sorted(gold), 2000)
zh_uris = list(zh_labels.keys())
en_uris = list(en_labels.keys())
idx = defaultdict(list)
for eu in en_uris:
    prefix = norm(en_labels[eu].get('label', eu))[:3]
    idx[prefix].append(eu)

neg = []
for z, e in pos:
    zt = norm(zh_labels[z].get('trans', zh_labels[z].get('label', z)))
    candidates = []
    for prefix_len in [3, 2, 1]:
        prefix = zt[:prefix_len]
        if prefix in idx:
            candidates = idx[prefix]
            break
    if not candidates:
        candidates = en_uris
    for _ in range(2):
        e_neg = random.choice(candidates)
        if (z, e_neg) not in gold and e_neg != e:
            neg.append((z, e_neg))

all_pairs = []
for z, e in pos:
    s = l1_score(z, e)
    all_pairs.append({'z': z, 'e': e, 'score': s, 'label': 1})
for z, e in neg:
    s = l1_score(z, e)
    all_pairs.append({'z': z, 'e': e, 'score': s, 'label': 0})

REJECT_TH = 0.3
MERGE_TH = 0.5
escalated = [p for p in all_pairs if p['score'] < MERGE_TH and p['score'] >= -REJECT_TH]
print(f"Total pairs: {len(all_pairs)}, Escalated: {len(escalated)}", flush=True)

# Run ALL escalated pairs (not just 500 sample)
sample = escalated
random.shuffle(sample)

# ─── Prompt builder (same as original) ───
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

    return f"""You are an expert in cross-lingual entity alignment. Determine whether the following two entities from different language editions of DBpedia refer to the SAME real-world entity.

Entity A (Chinese DBpedia): {zt}
Entity B (English DBpedia): {el_name}

Shared attributes ({len(shared)}): {', '.join(sorted(list(shared))[:10])}
Attributes only in A ({len(only_z)}): {', '.join(sorted(list(only_z))[:10])}
Attributes only in B ({len(only_e)}): {', '.join(sorted(list(only_e))[:10])}
Relation degree: A={zd}, B={ed}

Answer with exactly one word: MATCH or DIFFERENT."""

def call_glm5turbo(prompt, retries=5):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 4096,
    }).encode()
    for attempt in range(retries):
        req = urllib.request.Request(API_URL, data=body)
        req.add_header("Authorization", f"Bearer {API_KEY}")
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
            elif e.code == 400:
                err_body = e.read().decode()[:200]
                return f"ERROR_400: {err_body}"
            else:
                return f"ERROR: HTTP {e.code}"
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                return f"ERROR: {str(e)[:80]}"
    return "ERROR: max retries exceeded"

def parse_response(raw):
    raw_lower = raw.lower().strip()
    if 'match' in raw_lower and 'different' not in raw_lower:
        return 1
    elif 'different' in raw_lower:
        return 0
    else:
        return -1

# ─── Resume from checkpoint ───
results = {}
if CKPT.exists():
    with open(CKPT) as f:
        saved = json.load(f)
    for r in saved:
        key = f"{r['z']}|{r['e']}"
        results[key] = r
    print(f"Resuming: {len(results)} done, {len(sample) - len(results)} remaining", flush=True)

remaining = []
for pair in sample:
    key = f"{pair['z']}|{pair['e']}"
    if key not in results:
        remaining.append(pair)

print(f"To run: {len(remaining)} pairs with {MAX_WORKERS} workers", flush=True)

# ─── Run ───
lock = threading.Lock()
done_count = 0
total = len(remaining)
start_time = time.time()

def process_one(pair):
    global done_count
    prompt = build_prompt(pair)
    raw = call_glm5turbo(prompt)
    pred = parse_response(raw)
    return {
        'z': pair['z'], 'e': pair['e'],
        'label': pair['label'],
        'l1_score': round(pair['score'], 4),
        'l2_raw': raw[:200],
        'l2_prediction': pred,
    }

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_one, pair): pair for pair in remaining}
    
    for future in as_completed(futures):
        try:
            result = future.result()
            key = f"{result['z']}|{result['e']}"
            with lock:
                results[key] = result
                done_count += 1
                
                if done_count % 10 == 0:
                    all_results = list(results.values())
                    with open(CKPT, 'w') as f:
                        json.dump(all_results, f, ensure_ascii=False)
                    
                    elapsed = time.time() - start_time
                    rate = done_count / elapsed if elapsed > 0 else 0
                    eta = (total - done_count) / rate if rate > 0 else 0
                    
                    valid = [r for r in all_results if r['l2_prediction'] != -1]
                    acc = sum(1 for r in valid if r['l2_prediction'] == r['label']) / max(len(valid), 1) * 100
                    
                    print(f"  [{done_count}/{total}] {done_count/total*100:.1f}%  "
                          f"rate={rate:.1f}/s  eta={eta/60:.0f}min  "
                          f"acc={acc:.1f}% ({len(valid)} valid)", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)

# Final save
all_results = list(results.values())
with open(CKPT, 'w') as f:
    json.dump(all_results, f, ensure_ascii=False)

# ─── Compute results ───
valid = [r for r in all_results if r['l2_prediction'] != -1]
errors = sum(1 for r in all_results if r['l2_prediction'] == -1)
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

print(f"\n{'='*60}", flush=True)
print(f"DBP15K L2 GLM-5-Turbo Results (n={len(valid)} valid / {errors} errors)", flush=True)
print(f"{'='*60}", flush=True)
print(f"Accuracy:  {acc:.1f}%", flush=True)
print(f"  Positive (MATCH): {tp}/{pos_n} = {pos_acc:.1f}%", flush=True)
print(f"  Negative (DIFF):  {tn}/{neg_n} = {neg_acc:.1f}%", flush=True)
print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}", flush=True)
print(f"  Precision: {precision:.1f}%  Recall: {recall:.1f}%  F1: {f1:.1f}%", flush=True)

summary = {
    'l2_accuracy': acc / 100,
    'l2_errors': errors,
    'total_pairs': len(all_results),
    'valid_pairs': len(valid),
    'positive_pairs': pos_n,
    'negative_pairs': neg_n,
    'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    'positive_accuracy': pos_acc / 100,
    'negative_accuracy': neg_acc / 100,
    'precision': precision / 100,
    'recall': recall / 100,
    'f1': f1 / 100,
    'sampling_strategy': 'all_escalated',
    'model': MODEL,
}

with open(RESULTS, 'w') as f:
    json.dump({'summary': summary, 'predictions': all_results}, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {RESULTS}", flush=True)
