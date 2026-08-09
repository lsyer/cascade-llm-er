#!/usr/bin/env python3
"""
GLM-5-Turbo L2 experiment: run all 2639 pairs with independent model.
Uses the EXACT same prompt template as 17_l2_experiment_v3.py.
5 concurrent workers, checkpoint resumption.
"""
import json, re, time, urllib.request, urllib.error
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import threading

BASE = Path(__file__).resolve().parents[1]
CKPT_DIR = BASE / "results" / "checkpoints_v3"
OUT_CKPT = CKPT_DIR / "glm-5-turbo_checkpoint.json"
API_URL = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
MODEL = "glm-5-turbo"
MAX_WORKERS = 5

# ─── Load API key ───
_env = {}
with open(os.environ.get("ENV_FILE", ".env")) as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            _env[k] = v
API_KEY = _env.get("GLM_API_KEY", "")

# ─── Load data from dataset_v3_cleaned.json (same source as original script) ───
with open(BASE / "data" / "dataset_v3_cleaned.json") as f:
    ds = json.load(f)

with open(BASE / "system" / "nebula_data.json") as f:
    nebula = json.load(f)

entity_props = {}
for etype in ['equipment', 'person', 'location', 'event', 'organization']:
    for entity in nebula.get(etype, []):
        entity_props[entity['vid']] = {'type': etype, 'props': entity['props']}

with open(BASE / "data" / "entity_fragments.json") as f:
    fragments = json.load(f)['fragments']

# Build pair data for ALL 2639 pairs
pair_data = {}
for r in ds['same_type_pairs']:
    pid = r['pair_id']
    pair_data[pid] = {
        'pair_id': pid,
        'vid_a': r.get('vid_a', ''),
        'vid_b': r.get('vid_b', ''),
        'name_a': r.get('name_a', ''),
        'name_b': r.get('name_b', ''),
        'type': r.get('type_a', ''),
        'props_a': r.get('props_a', {}),
        'props_b': r.get('props_b', {}),
        'frags_a': fragments.get(r.get('vid_a', ''), []),
        'frags_b': fragments.get(r.get('vid_b', ''), []),
    }

print(f"Total pairs to evaluate: {len(pair_data)}", flush=True)
missing_props = sum(1 for p in pair_data.values() if not p.get('props_a'))
print(f"Pairs missing props_a: {missing_props}", flush=True)

# ─── Exact same prompt template as 17_l2_experiment_v3.py ───
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

def build_prompt(pd):
    """Build L2 prompt — IDENTICAL to 17_l2_experiment_v3.py build_prompt."""
    etype = pd.get('type', '')
    rules = TYPE_RULES.get(etype, DEFAULT_RULES)
    
    cp_a = clean_props(pd.get('props_a', {}))
    cp_b = clean_props(pd.get('props_b', {}))
    
    def fmt_fragments(frags, name):
        if not frags:
            return f"  (No source text available for this entity)"
        lines = []
        for i, f in enumerate(frags[:2]):
            lines.append(f"  [{i+1}] {f}")
        return "\n".join(lines)
    
    props_str_a = json.dumps(cp_a, ensure_ascii=False, indent=2)
    props_str_b = json.dumps(cp_b, ensure_ascii=False, indent=2)
    
    return f"""You are an expert entity resolution system for a military/naval intelligence knowledge graph. 
Determine if two entities are the SAME real-world entity or DIFFERENT entities.

Entity type: {etype}

{rules}

Entity A (name="{pd['name_a']}"):
Attributes:
{props_str_a}
Source text fragments (where this entity appeared in original articles):
{fmt_fragments(pd.get('frags_a', []), pd['name_a'])}

Entity B (name="{pd['name_b']}"):
Attributes:
{props_str_b}
Source text fragments (where this entity appeared in original articles):
{fmt_fragments(pd.get('frags_b', []), pd['name_b'])}

Based on the attributes AND the source text evidence above, are these the SAME entity or DIFFERENT entities?

Answer with exactly one word: "SAME" or "DIFFERENT" """

def call_glm5turbo(prompt, retries=5):
    """Call GLM-5-Turbo API."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 4096,  # thinking model needs more tokens
    }).encode()
    
    for attempt in range(retries):
        req = urllib.request.Request(API_URL, data=body)
        req.add_header("Authorization", f"Bearer {API_KEY}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                content = data['choices'][0]['message']['content'].strip()
                # Strip thinking content if present
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

def parse_verdict(raw):
    """Same parser as original script."""
    if not raw:
        return 'unknown'
    raw_lower = raw.lower().strip()
    if 'same' in raw_lower and 'different' not in raw_lower:
        return 'same'
    elif 'different' in raw_lower or 'differ' in raw_lower:
        return 'different'
    return 'unknown'

# ─── Resume from checkpoint ───
results = {}
if OUT_CKPT.exists():
    with open(OUT_CKPT) as f:
        results = json.load(f)
    print(f"Resuming: {len(results)} done, {len(pair_data) - len(results)} remaining", flush=True)

remaining = [(pid, pd) for pid, pd in pair_data.items() if pid not in results]
print(f"To run: {len(remaining)} pairs with {MAX_WORKERS} workers", flush=True)

# ─── Run with thread pool ───
lock = threading.Lock()
done_count = 0
total = len(remaining)
start_time = time.time()

def process_one(item):
    global done_count
    pid, pd = item
    
    prompt = build_prompt(pd)
    raw = call_glm5turbo(prompt)
    verdict = parse_verdict(raw)
    
    return pid, {'raw': raw[:500], 'verdict': verdict}

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_one, item): item for item in remaining}
    
    for future in as_completed(futures):
        try:
            pid, result = future.result()
            with lock:
                results[pid] = result
                done_count += 1
                
                if done_count % 10 == 0:
                    # Save checkpoint
                    with open(OUT_CKPT, 'w') as f:
                        json.dump(results, f, ensure_ascii=False)
                    
                    elapsed = time.time() - start_time
                    rate = done_count / elapsed if elapsed > 0 else 0
                    eta = (total - done_count) / rate if rate > 0 else 0
                    
                    correct = sum(1 for r in results.values() 
                                  if r['verdict'] != 'unknown')
                    unknown = sum(1 for r in results.values() 
                                  if r['verdict'] == 'unknown')
                    
                    print(f"  [{done_count}/{total}] {done_count/total*100:.1f}%  "
                          f"rate={rate:.1f}/s  eta={eta/60:.0f}min  "
                          f"unknown={unknown}", flush=True)
        except Exception as e:
            pid = futures[future][0]
            print(f"  FAILED: {pid}: {e}", flush=True)
            results[pid] = {'raw': f'EXCEPTION: {e}', 'verdict': 'unknown'}

# Final save
with open(OUT_CKPT, 'w') as f:
    json.dump(results, f, ensure_ascii=False)

# ─── Summary ───
verdicts = Counter(r['verdict'] for r in results.values())
print(f"\n{'='*60}", flush=True)
print(f"GLM-5-Turbo L2 experiment complete", flush=True)
print(f"{'='*60}", flush=True)
print(f"Total: {len(results)}", flush=True)
print(f"Verdicts: {dict(verdicts)}", flush=True)
print(f"Saved to: {OUT_CKPT}", flush=True)
