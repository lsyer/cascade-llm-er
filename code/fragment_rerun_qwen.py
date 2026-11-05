#!/usr/bin/env python3
"""fragment_rerun_qwen.py — Re-run the fragment ablation for the VERIFIED local
judge (Qwen3.6-35B-A3B-INT8, vLLM) on the same 290-pair sample as
fragment_ablation_v2.py (same eligibility filter, same random.seed(42), same
per-type cap), in BOTH conditions (with / without fragments), because the
original experiment used an endpoint whose served model could not be verified.
Labels: ground-truth-v2 final_label. Output: results/fragment_rerun_qwen.json
"""
import json, os, re, sys, time, random
from collections import defaultdict, Counter
import urllib.request
import concurrent.futures as cf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('ENV_FILE', '/root/.hermes/.env')
from run_glm5turbo_l2 import build_prompt, clean_props, TYPE_RULES, DEFAULT_RULES

def build_prompt_no_fragment(etype, name_a, name_b, props_a, props_b):
    """Verbatim copy of fragment_ablation_v2.build_prompt_no_fragment (module has
    no __main__ guard so importing it would re-run the old experiment)."""
    cp_a = clean_props(props_a)
    cp_b = clean_props(props_b)
    rules = TYPE_RULES.get(etype, DEFAULT_RULES)
    props_str_a = json.dumps(cp_a, ensure_ascii=False, indent=2)
    props_str_b = json.dumps(cp_b, ensure_ascii=False, indent=2)
    return f"""You are an expert entity resolution system for a military/naval intelligence knowledge graph.
Determine if two entities are the SAME real-world entity or DIFFERENT entities.

Entity type: {etype}

{rules}

Entity A (name="{name_a}"):
Attributes:
{props_str_a}

Entity B (name="{name_b}"):
Attributes:
{props_str_b}

Based on the attributes above, are these the SAME entity or DIFFERENT entities?

Answer with exactly one word: "SAME" or "DIFFERENT" """

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(BASE, 'results')
API = 'http://www.netforger.com:8009/v1/chat/completions'
MODEL = 'Qwen3.6-35B-A3B'
KEY = __import__('os').environ.get('QWEN_KEY', '')
OUT = os.path.join(RES, 'fragment_rerun_qwen.json')

with open(os.path.join(BASE, 'data', 'dataset_v3_cleaned.json')) as f:
    ds = json.load(f)
entity_props = {}
for r in ds['same_type_pairs']:
    for side in ('a', 'b'):
        vid = r.get(f'vid_{side}', '')
        if vid and vid not in entity_props:
            entity_props[vid] = {'type': r.get(f'type_{side}', ''), 'props': r.get(f'props_{side}', {})}
with open(os.path.join(BASE, 'data', 'entity_fragments.json')) as f:
    fragments = json.load(f)['fragments']
with open(os.path.join(RES, 'scored_pairs.json')) as f:
    sp = json.load(f)
gt = json.load(open(os.path.join(BASE, 'data', 'minec_ground_truth_v2.json')))
recs = gt['records'] if isinstance(gt, dict) else gt
gold = {r['pair_id']: r['final_label'] for r in recs}

# identical sample reconstruction
eligible = []
for r in sp:
    if r.get('l1_decision_v2') != 'escalate':
        continue
    va, vb = r.get('vid_a', ''), r.get('vid_b', '')
    if va not in entity_props or vb not in entity_props:
        continue
    if len(fragments.get(va, [])) == 0 and len(fragments.get(vb, [])) == 0:
        continue
    eligible.append(r)
random.seed(42)
by_type = defaultdict(list)
for r in eligible:
    by_type[r.get('type_a', 'unknown')].append(r)
sample = []
for t in sorted(by_type.keys()):
    n = min(len(by_type[t]), 73)
    sample.extend(random.sample(by_type[t], n))
print(f"sampled {len(sample)} pairs: {Counter(r['type_a'] for r in sample)}", flush=True)

def call(prompt, retries=4):
    body = json.dumps({'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
                       'temperature': 0.1, 'max_tokens': 512,
                       'chat_template_kwargs': {'enable_thinking': False}}).encode()
    for a in range(retries):
        req = urllib.request.Request(API, data=body)
        req.add_header('Authorization', f'Bearer {KEY}')
        req.add_header('Content-Type', 'application/json')
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())['choices'][0]['message']['content'].strip()
        except Exception:
            if a == retries - 1:
                return 'ERROR'
            time.sleep(5 * (a + 1))

def strict(raw):
    s = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip().lower()
    return s if s in ('same', 'different') else 'unknown'

done = {}
if os.path.exists(OUT):
    done = {r['pair_id']: r for r in json.load(open(OUT))['records']}
todo = [r for r in sample if r['pair_id'] not in done]
print(f"todo {len(todo)}", flush=True)

records = list(done.values())
LOCK = __import__('threading').Lock()
def work(pair):
    pid = pair['pair_id']
    va, vb = pair['vid_a'], pair['vid_b']
    etype = pair.get('type_a') or entity_props[va]['type']
    with_f = build_prompt({'type': etype, 'name_a': pair['name_a'], 'name_b': pair['name_b'],
                           'props_a': entity_props[va]['props'], 'props_b': entity_props[vb]['props'],
                           'frags_a': fragments.get(va, []), 'frags_b': fragments.get(vb, [])})
    no_f = build_prompt_no_fragment(etype, pair['name_a'], pair['name_b'],
                                    entity_props[va]['props'], entity_props[vb]['props'])
    rec = {'pair_id': pid, 'type': pair['type_a'], 'gold': gold.get(pid),
           'with_frag': strict(call(with_f)), 'no_frag': strict(call(no_f))}
    with LOCK:
        records.append(rec)
        if len(records) % 50 == 0:
            json.dump({'records': records}, open(OUT, 'w'))
    return rec

with cf.ThreadPoolExecutor(max_workers=3) as ex:
    list(ex.map(work, todo))
json.dump({'records': records}, open(OUT, 'w'))

def acc(field):
    ok = [r for r in records if r[field] in ('same', 'different')]
    c = sum(r[field] == r['gold'] for r in ok)
    return c, len(records), round(c / len(records) * 100, 1), len(records) - len(ok)
for f in ('with_frag', 'no_frag'):
    print(f, acc(f))
summary = {f: {'correct': acc(f)[0], 'n': acc(f)[1], 'acc_pct': acc(f)[2], 'unknown': acc(f)[3]}
           for f in ('with_frag', 'no_frag')}
summary['delta_pp'] = round(summary['with_frag']['acc_pct'] - summary['no_frag']['acc_pct'], 1)
json.dump({'summary': summary, 'records': records}, open(OUT, 'w'), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1))
