#!/usr/bin/env python3
"""qwen_full_l2.py — Full-set (2,591 MINEC pairs) Qwen3.6-35B-A3B-INT8 judge run.

Purpose (reviewer R2-4/R3-D4 external baselines):
  (a) pure SLM-judge baseline accuracy on the full GT
  (b) Qwen-as-L2 cascade replay (equal-budget routing curves with a local SLM)
Prompt identical to deployment L2 (run_glm5turbo_l2.build_prompt), thinking off,
temp 0.1, strict verdict parsing. Resume-safe incremental checkpoint.
Output: results/qwen_l2_full.json
"""
import json, os, re, sys, time
import urllib.request
import concurrent.futures as cf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('ENV_FILE', '/root/.hermes/.env')
from run_glm5turbo_l2 import build_prompt

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(BASE, 'results')
API = 'http://www.netforger.com:8009/v1/chat/completions'
MODEL = 'Qwen3.6-35B-A3B'
KEY = __import__('os').environ.get('QWEN_KEY', '')
OUT = os.path.join(RES, 'qwen_l2_full.json')

scored = json.load(open(os.path.join(RES, 'scored_pairs.json')))
gt = json.load(open(os.path.join(BASE, 'data', 'minec_ground_truth_v2.json')))
gtl = {r['pair_id']: r for r in (gt['records'] if isinstance(gt, dict) else gt)}
ds = json.load(open(os.path.join(BASE, 'data', 'dataset_v3_cleaned.json')))
pairdata = {p['pair_id']: p for p in ds['same_type_pairs'] + ds.get('cross_type_pairs', [])}
frag = json.load(open(os.path.join(BASE, 'data', 'entity_fragments.json')))['fragments']

def make_pd(pid):
    d = pairdata[pid]
    et = d.get('type_a') or d.get('type') or ''
    return {'type': et, 'name_a': d['name_a'], 'name_b': d['name_b'],
            'props_a': d['props_a'], 'props_b': d['props_b'],
            'frags_a': frag.get(d['vid_a'], []), 'frags_b': frag.get(d['vid_b'], [])}

def call_qwen(prompt, retries=4):
    body = json.dumps({'model': MODEL,
                       'messages': [{'role': 'user', 'content': prompt}],
                       'temperature': 0.1, 'max_tokens': 512,
                       'chat_template_kwargs': {'enable_thinking': False}}).encode()
    for attempt in range(retries):
        req = urllib.request.Request(API, data=body)
        req.add_header('Authorization', f'Bearer {KEY}')
        req.add_header('Content-Type', 'application/json')
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())['choices'][0]['message']['content'].strip()
        except Exception:
            if attempt == retries - 1:
                return 'ERROR'
            time.sleep(5 * (attempt + 1))

def strict(raw):
    s = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip().lower()
    return s if s in ('same', 'different') else 'unresolved'

done = {}
if os.path.exists(OUT):
    done = {r['pair_id']: r['verdict'] for r in json.load(open(OUT))['records']
            if r['verdict'] != 'unresolved'}
todo = [p['pair_id'] for p in scored if p['pair_id'] not in done]
print(f'total {len(scored)}, done {len(done)}, todo {len(todo)}', flush=True)

records = json.load(open(OUT))['records'] if os.path.exists(OUT) else []
t0 = time.time()
LOCK = __import__('threading').Lock()
def work(pid):
    v = strict(call_qwen(build_prompt(make_pd(pid))))
    with LOCK:
        records.append({'pair_id': pid, 'verdict': v})
        if len(records) % 200 == 0:
            json.dump({'records': records}, open(OUT, 'w'))
            print(f'{len(records)}/{len(scored)} ({time.time()-t0:.0f}s)', flush=True)
    return v

with cf.ThreadPoolExecutor(max_workers=3) as ex:
    list(ex.map(work, todo))
json.dump({'records': records}, open(OUT, 'w'))

ok = [r for r in records if r['verdict'] != 'unresolved']
def gold(pid):
    g = gtl.get(pid)
    return g['final_label'] if g else None
acc = sum(gold(r['pair_id']) == r['verdict'] for r in ok) / len(ok) * 100
overall = sum(gold(r['pair_id']) == r['verdict'] for r in records) / len(records) * 100
summary = {'model': MODEL, 'n': len(records), 'unresolved': len(records) - len(ok),
           'resolved_acc_pct': round(acc, 2), 'overall_strict_pct': round(overall, 2),
           'glm5turbo_full_ref': 91.6, 'elapsed_s': round(time.time() - t0)}
json.dump({'summary': summary, 'records': records}, open(OUT, 'w'), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1))
