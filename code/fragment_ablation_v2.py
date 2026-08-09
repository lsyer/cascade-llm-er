#!/usr/bin/env python3
"""
Fragment ablation v2: Run all 4 L2 models WITHOUT source text fragments.
Compare accuracy with the with-fragment checkpoints.

Uses the same prompt template as run_glm5turbo_l2.py / 17_l2_experiment_v3.py,
but removes the "Source text fragments" section from both entities.

Sample: 365 pairs stratified by type (73 per type), matching the paper's
without-fragment denominator.

Models: GLM-5-Turbo, GLM-4.5-Air, Qwen3.6, GLM-4.5
"""
import json, re, random, time, urllib.request, urllib.error
import os
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path(__file__).resolve().parents[1]
CKPT_DIR = BASE / "results" / "checkpoints_v3"
OUT = BASE / "results" / "fragment_ablation_v2_results.json"
CKPT_FILE = BASE / "results" / "fragment_ablation_v2_checkpoint.json"

# ─── Config ───
# Read API keys
_env = {}
env_paths = [os.environ.get("ENV_FILE", ".env"), ".env"]
for ep in env_paths:
    try:
        with open(ep) as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    _env[k] = v
    except FileNotFoundError:
        pass

GLM_KEY = _env.get('LLM_API_KEY', _env.get('GLM_API_KEY', ''))
GLM_URL = 'https://open.bigmodel.cn/api/coding/paas/v4/chat/completions'
QWEN_URL = 'http://www.netforger.com:8009/v1/chat/completions'
QWEN_KEY = _env.get('QWEN_KEY', 'token-abc123')

MODELS = {
    'glm-5-turbo': {'url': GLM_URL, 'key': GLM_KEY, 'model_id': 'glm-5-turbo'},
    'glm-4.5-air': {'url': GLM_URL, 'key': GLM_KEY, 'model_id': 'glm-4.5-air'},
    'glm-4.5':     {'url': GLM_URL, 'key': GLM_KEY, 'model_id': 'glm-4.5'},
    'qwen3.6':     {'url': QWEN_URL, 'key': QWEN_KEY, 'model_id': 'Qwen3.6'},
}

# ─── Load data ───
print("Loading data...", flush=True)

# Read from dataset_v3_cleaned.json — same source as run_glm5turbo_l2.py
# This covers all 2639 pairs with props_a/props_b inline
with open(BASE / "data" / "dataset_v3_cleaned.json") as f:
    ds = json.load(f)

entity_props = {}
for r in ds['same_type_pairs']:
    vid = r.get('vid_a', '')
    if vid and vid not in entity_props:
        entity_props[vid] = {'type': r.get('type_a', ''), 'props': r.get('props_a', {})}
    vid = r.get('vid_b', '')
    if vid and vid not in entity_props:
        entity_props[vid] = {'type': r.get('type_b', ''), 'props': r.get('props_b', {})}

with open(BASE / "data" / "entity_fragments.json") as f:
    fragments = json.load(f)['fragments']

with open(BASE / "results" / "scored_pairs.json") as f:
    sp = json.load(f)

# With-fragment checkpoints (for comparison)
with_frag_ckpts = {}
for model_name in MODELS:
    ckpt_path = CKPT_DIR / f"{model_name}_checkpoint.json"
    with open(ckpt_path) as f:
        with_frag_ckpts[model_name] = json.load(f)
    print(f"  Loaded {model_name} checkpoint: {len(with_frag_ckpts[model_name])} entries", flush=True)

# ─── Prompt template (NO fragments) ───
USELESS_VALUES = {'', 'None', '__EMPTY__', 'N/A', 'null', '[]'}

def clean_props(props):
    cleaned = {}
    for k, v in props.items():
        s = str(v).strip().strip('"')
        if s in USELESS_VALUES:
            continue
        cleaned[k] = s
    return cleaned

TYPE_RULES = {
    "equipment": """Judging rules for EQUIPMENT (ships, aircraft, vehicles, weapons, radar):
- Hull number / designation (e.g. CVN-78, DDG-51) is the strongest identifier. Same designation = same equipment.
- Same ship/aircraft with different name spellings = same (e.g. "USS Gerald Ford" vs "USS Gerald R. Ford").
- Different hull numbers = different equipment, even if same class.
- Same name but different type (e.g. a ship vs an aircraft both called "Enterprise") = different.""",

    "person": """Judging rules for PERSON:
- Same full name + same rank/role + same organization = same person.
- Same name + different nationality or different era (decades apart) = likely different.
- Name variations (e.g. "Pete Hegseth" vs "Pete Brian Hegseth") = same if context matches.
- Generic titles (e.g. "Secretary of Defense") without name = different placeholder entities.""",

    "organization": """Judging rules for ORGANIZATION (military units, companies, government agencies):
- Same unit designation/number (e.g. "VFA-27", "Carrier Air Wing 8") = same organization.
- Same name + same region/industry = same.
- Parent and subsidiary (e.g. "BAE Systems" vs "BAE Systems Naval Ships") = different entities.
- Different level of the same hierarchy (e.g. "US Navy" vs "7th Fleet") = different.""",

    "location": """Judging rules for LOCATION (ports, bases, cities, regions):
- Same coordinates = same location.
- Same name + same region/country = likely same.
- Same name but different country (e.g. "Springfield, VA" vs "Springfield, IL") = different.
- A facility within a city (e.g. "Norfolk Naval Base" vs "Norfolk") = different if distinct.""",

    "event": """Judging rules for EVENT (exercises, operations, summits, incidents):
- Same name + same time period + same location = same event.
- Same name but different year or different participants = different event instance.
- Recurring events (e.g. "RIMPAC 2024" vs "RIMPAC 2026") = different.
- Generic names (e.g. "naval exercise") need strong contextual evidence to match.""",
}

DEFAULT_RULES = """Judging rules:
- Compare all available attributes systematically.
- Same name alone is not sufficient — need corroborating evidence.
- When in doubt, lean towards DIFFERENT (false merge is worse than false split)."""

def build_prompt_no_fragment(etype, name_a, name_b, props_a, props_b):
    """Build L2 prompt WITHOUT source text fragments."""
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

def call_model(model_config, prompt, retries=5):
    body = json.dumps({
        "model": model_config['model_id'],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 4096,
    }).encode()

    for attempt in range(retries):
        req = urllib.request.Request(model_config['url'], data=body)
        req.add_header("Authorization", f"Bearer {model_config['key']}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                content = data['choices'][0]['message']['content'].strip()
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                content = re.sub(r"Here's a thinking process:.*?</think>\s*", '', content, flags=re.DOTALL).strip()
                return content if content else "EMPTY"
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 60 * (attempt + 1)
                print(f"    429, waiting {wait}s (attempt {attempt+1}/{retries})", flush=True)
                time.sleep(wait)
            else:
                return f"ERROR: HTTP {e.code}"
        except Exception as e:
            return f"ERROR: {str(e)[:80]}"
    return "ERROR: max retries exceeded"

def parse_verdict(raw):
    if not raw:
        return 'unknown'
    raw_lower = raw.lower().strip()
    if 'same' in raw_lower and 'different' not in raw_lower:
        return 'same'
    elif 'different' in raw_lower or 'differ' in raw_lower:
        return 'different'
    return 'unknown'

# ─── Select sample (365 pairs, 73 per type) ───
eligible = []
for r in sp:
    if r.get('l1_decision_v2') != 'escalate':
        continue
    vid_a = r.get('vid_a', '')
    vid_b = r.get('vid_b', '')
    if vid_a not in entity_props or vid_b not in entity_props:
        continue
    frag_a = fragments.get(vid_a, [])
    frag_b = fragments.get(vid_b, [])
    if len(frag_a) == 0 and len(frag_b) == 0:
        continue
    eligible.append(r)

print(f"Eligible pairs (escalated, with ≥1 fragment): {len(eligible)}", flush=True)

random.seed(42)
by_type = defaultdict(list)
for r in eligible:
    by_type[r.get('type_a', 'unknown')].append(r)

sample = []
TARGET_PER_TYPE = 73  # 73 × 5 = 365
for t in sorted(by_type.keys()):
    pairs = by_type[t]
    n = min(len(pairs), TARGET_PER_TYPE)
    sample.extend(random.sample(pairs, n))
random.shuffle(sample)
print(f"Sampled: {len(sample)} pairs", flush=True)
print(f"Type distribution: {Counter(r.get('type_a','?') for r in sample)}", flush=True)

# ─── Resume from checkpoint ───
results = {}  # {pair_id: {model: {verdict, raw}}}
if CKPT_FILE.exists():
    with open(CKPT_FILE) as f:
        results = json.load(f)
    print(f"Resuming: {len(results)} pairs already done", flush=True)

remaining = [r for r in sample if r['pair_id'] not in results]
print(f"Remaining: {len(remaining)} pairs × {len(MODELS)} models = {len(remaining) * len(MODELS)} API calls", flush=True)

# ─── Run ───
total_calls = len(remaining) * len(MODELS)
calls_done = 0

for i, pair in enumerate(remaining):
    pid = pair['pair_id']
    vid_a = pair['vid_a']
    vid_b = pair['vid_b']
    gold = 'same' if pair['label'] == 1 else 'different'
    etype = entity_props[vid_a]['type']

    results[pid] = {'type': pair['type_a'], 'gold': gold}

    for model_name, model_cfg in MODELS.items():
        calls_done += 1
        prompt = build_prompt_no_fragment(
            etype,
            pair['name_a'], pair['name_b'],
            entity_props[vid_a]['props'], entity_props[vid_b]['props']
        )
        raw = call_model(model_cfg, prompt)
        verdict = parse_verdict(raw)
        results[pid][model_name] = {'verdict': verdict, 'raw': raw[:200]}

    # Checkpoint every 5 pairs
    if (i + 1) % 5 == 0:
        with open(CKPT_FILE, 'w') as f:
            json.dump(results, f, ensure_ascii=False)

        done_pairs = len(results)
        print(f"  [{i+1}/{len(remaining)}] {calls_done}/{total_calls} calls done, "
              f"{done_pairs} pairs complete", flush=True)

    time.sleep(1)  # rate limit

# ─── Save final ───
with open(CKPT_FILE, 'w') as f:
    json.dump(results, f, ensure_ascii=False)

# ─── Compute results ───
print(f"\n{'='*60}", flush=True)
print(f"FRAGMENT ABLATION v2 RESULTS (n={len(results)} pairs)", flush=True)
print(f"{'='*60}", flush=True)

summary = {'total_pairs': len(results), 'models': {}}

for model_name in MODELS:
    # With fragment accuracy (from checkpoint)
    wf_correct = wf_unknown = 0
    for pid, r in results.items():
        gold = r['gold']
        wf_verdict = with_frag_ckpts[model_name].get(pid, {}).get('verdict', 'unknown')
        if wf_verdict == 'unknown':
            wf_unknown += 1
        elif wf_verdict == gold:
            wf_correct += 1

    # Without fragment accuracy (from this run)
    nf_correct = nf_unknown = 0
    for pid, r in results.items():
        gold = r['gold']
        nf_data = r.get(model_name, {})
        nf_verdict = nf_data.get('verdict', 'unknown')
        if nf_verdict == 'unknown':
            nf_unknown += 1
        elif nf_verdict == gold:
            nf_correct += 1

    total = len(results)
    wf_acc = wf_correct / total * 100
    nf_acc = nf_correct / total * 100
    delta = wf_acc - nf_acc

    print(f"\n{model_name}:", flush=True)
    print(f"  With fragment:    {wf_correct}/{total} = {wf_acc:.1f}% ({wf_unknown} unknown)", flush=True)
    print(f"  Without fragment: {nf_correct}/{total} = {nf_acc:.1f}% ({nf_unknown} unknown)", flush=True)
    print(f"  Delta: {delta:+.1f} pp", flush=True)

    summary['models'][model_name] = {
        'with_fragment_accuracy_pct': round(wf_acc, 1),
        'with_fragment_correct': wf_correct,
        'with_fragment_unknown': wf_unknown,
        'without_fragment_accuracy_pct': round(nf_acc, 1),
        'without_fragment_correct': nf_correct,
        'without_fragment_unknown': nf_unknown,
        'delta_pp': round(delta, 1),
        'total_pairs': total,
    }

    # Per-type
    per_type = {}
    for t in sorted(set(r['type'] for r in results.values())):
        tr_pids = [pid for pid, r in results.items() if r['type'] == t]
        n_t = len(tr_pids)
        wf_t = sum(1 for pid in tr_pids if with_frag_ckpts[model_name].get(pid, {}).get('verdict', '') == results[pid]['gold'])
        nf_t = sum(1 for pid in tr_pids if results[pid].get(model_name, {}).get('verdict', '') == results[pid]['gold'])
        per_type[t] = {
            'n': n_t,
            'with_frag_acc': round(wf_t / n_t * 100, 1) if n_t else 0,
            'no_frag_acc': round(nf_t / n_t * 100, 1) if n_t else 0,
        }
    summary['models'][model_name]['per_type'] = per_type

output = {'summary': summary}
with open(OUT, 'w') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {OUT}", flush=True)
