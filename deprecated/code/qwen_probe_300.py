#!/usr/bin/env python3
"""qwen_probe_300.py — Probe Qwen3.6-35B-A3B-INT8 (local vLLM) as L2 judge on the
300-pair human-annotated subset.

Identity verified: /v1/models -> owned_by vllm, root /app/models/Apsara-Stack/
Qwen3___6-35B-A3B-INT8; self-report Alibaba Tongyi Lab Qwen.
Prompt: EXACTLY the deployment L2 prompt (imported from run_glm5turbo_l2.build_prompt)
— same prompt, different judge = clean comparison with GLM-5-Turbo (95.7% on this
subset). Thinking disabled (enable_thinking false), temperature 0.1 (same as L2).
Verdict parsing: STRICT — content must be exactly same/different (case-insensitive,
after whitespace strip); anything else = unresolved.
Output: results/qwen_probe_300.json
"""
import json, os, re, sys, time
import urllib.request
import concurrent.futures as cf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from run_glm5turbo_l2 import build_prompt, clean_props

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(BASE, 'results')
API = 'http://www.netforger.com:8009/v1/chat/completions'
MODEL = 'Qwen3.6-35B-A3B'
KEY = __import__('os').environ.get('QWEN_KEY', '')

h = json.load(open(os.path.join(BASE, 'data', 'minec_human_annotation_300.json')))
recs = h['records']

ds = json.load(open(os.path.join(BASE, 'data', 'dataset_v3_cleaned.json')))
pairdata = {}
for p in ds['same_type_pairs'] + ds.get('cross_type_pairs', []):
    pairdata[p['pair_id']] = p
frag = json.load(open(os.path.join(BASE, 'data', 'entity_fragments.json')))['fragments']

def make_pd(r):
    d = pairdata[r['pair_id']]
    return {'type': r['type'], 'name_a': r['name_a'], 'name_b': r['name_b'],
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
                data = json.loads(resp.read())
                return data['choices'][0]['message']['content'].strip()
        except Exception as e:
            if attempt == retries - 1:
                return f'ERROR:{type(e).__name__}'
            time.sleep(5 * (attempt + 1))

def strict(raw):
    s = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip().lower()
    if s == 'same':
        return 'same'
    if s == 'different':
        return 'different'
    return 'unresolved'

results = []
t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=3) as ex:
    futs = {ex.submit(call_qwen, build_prompt(make_pd(r))): r for r in recs}
    for i, fut in enumerate(cf.as_completed(futs), 1):
        r = futs[fut]
        raw = fut.result()
        v = strict(raw)
        results.append({'pair_id': r['pair_id'], 'human': r['human_label'],
                        'verdict': v, 'raw_head': raw[:120] if raw else 'EMPTY'})
        if i % 50 == 0:
            print(f'{i}/300 ({time.time()-t0:.0f}s)', flush=True)

ok = [x for x in results if x['verdict'] != 'unresolved']
acc = sum(x['verdict'] == x['human'] for x in ok) / len(ok) * 100
overall = sum(x['verdict'] == x['human'] for x in results) / len(results) * 100
comp = len(ok) / len(results) * 100
by_t = {}
for x in results:
    t = next(r['type'] for r in recs if r['pair_id'] == x['pair_id'])
    by_t.setdefault(t, [0, 0, 0])
    by_t[t][0] += x['verdict'] == x['human']
    by_t[t][1] += 1
    by_t[t][2] += x['verdict'] == 'unresolved'
summary = {'model': MODEL, 'n': len(results), 'compliance_pct': round(comp, 1),
           'resolved_acc_pct': round(acc, 2), 'overall_strict_pct': round(overall, 2),
           'unresolved': sum(x['verdict'] == 'unresolved' for x in results),
           'by_type': {k: {'acc': round(v[0] / max(v[1] - v[2], 1) * 100, 1),
                           'n': v[1], 'unres': v[2]} for k, v in by_t.items()},
           'glm5turbo_subset_ref_pct': 95.7,
           'elapsed_s': round(time.time() - t0)}
json.dump({'summary': summary, 'records': results},
          open(os.path.join(RES, 'qwen_probe_300.json'), 'w'), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False))
