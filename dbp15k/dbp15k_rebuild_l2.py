#!/usr/bin/env python3
"""dbp15k_rebuild_l2.py — L2 verdicts for the 1,037 escalated pairs missing from
the existing checkpoint, for empirical pipeline replay (Table 5 fix).

Prompt, model, temperature, parsing: identical to run_dbp15k_glm5turbo.py.
Resume-safe: existing verdicts in dbp15k_rebuild_l2.json are kept.
"""
import json, os, re, time, urllib.request, urllib.error
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

OUT = Path(__file__).resolve().parent
DATA = OUT / "JAPE" / "data" / "dbp15k" / "zh_en"
RES = OUT / "results"
CKPT = RES / "dbp15k_rebuild_l2.json"
STATE = RES / "dbp15k_rebuild_state.json"
MODEL = "glm-5-turbo"
MAX_WORKERS = 5

_env = {}
with open(os.environ.get("ENV_FILE", "/root/.hermes/.env")) as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            _env[k] = v
API_KEY = _env.get("GLM_API_KEY", "")
API_URL = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"

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

zh_labels = load_labels(DATA / "s_labels")
en_labels = load_labels(DATA / "t_labels")
zh_attrs = load_attrs(DATA / "training_attrs_1")
en_attrs = load_attrs(DATA / "training_attrs_2")
zh_tris = load_triples(DATA / "s_triples")
en_tris = load_triples(DATA / "t_triples")

state = json.load(open(STATE))
missing = [tuple(m) for m in state['missing']]
print(f"missing pairs to run: {len(missing)}", flush=True)

try:
    ckpt = json.load(open(CKPT))
except FileNotFoundError:
    ckpt = {}
todo = [p for p in missing if f"{p[0]}||{p[1]}" not in ckpt]
print(f"already done: {len(missing)-len(todo)}, to run: {len(todo)}", flush=True)

def build_prompt(z, e):
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
                print(f"429, waiting {wait}s ({attempt+1}/{retries})", flush=True)
                time.sleep(wait)
            elif e.code == 400:
                return f"ERROR_400: {e.read().decode()[:200]}"
            else:
                return f"ERROR: HTTP {e.code}"
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                return f"ERROR: {str(e)[:80]}"
    return "ERROR: max retries exceeded"

lock = threading.Lock()
done = 0
def run_pair(p):
    global done
    z, e = p
    raw = call_glm5turbo(build_prompt(z, e))
    with lock:
        ckpt[f"{z}||{e}"] = {'raw': raw}
        done += 1
        if done % 25 == 0:
            json.dump(ckpt, open(CKPT, 'w'), ensure_ascii=False)
            print(f"progress {done}/{len(todo)}", flush=True)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = [ex.submit(run_pair, p) for p in todo]
    for f in as_completed(futures):
        f.result()

json.dump(ckpt, open(CKPT, 'w'), ensure_ascii=False)
errs = sum(1 for v in ckpt.values() if str(v['raw']).startswith('ERROR'))
print(f"DONE: {len(ckpt)} verdicts saved, errors: {errs}", flush=True)
