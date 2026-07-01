#!/usr/bin/env python3
"""
Experiment 32: Full annotation of all 2639 MINEC pairs with Claude Opus 4.8
using the canonical annotation prompt.

Uses the exact same prompt as GLM-5.2 re-annotation (experiment 31) for
fair comparison. No bias instruction, unified TYPE_JUDGING_RULES, both
entities get source text fragments, single-word output.

Features:
  - Checkpoint every 10 pairs
  - Auto-resume from checkpoint
  - Rate-limit retry with exponential backoff
  - Progress reporting to stderr
"""
import json, sys, time, requests
from pathlib import Path
from collections import Counter

# ── Config ──────────────────────────────────────────────────────────────

RELEASE = Path("/root/workspace/tanshu_docs/experiments/release")

with open("/root/.kksj_key") as f:
    KEY = f.read().strip()

API_URL = "https://api.kksj.org/v1/chat/completions"
MODEL = "claude-opus-4-8"

CKPT_PATH = RELEASE / "results/claude_opus_canonical_checkpoint.json"
FINAL_PATH = RELEASE / "results/claude_opus_canonical_annotation.json"

# ── Load data ───────────────────────────────────────────────────────────

with open(RELEASE / "data/dataset_v3_cleaned.json") as f:
    ds = json.load(f)
pairs = ds["same_type_pairs"]

with open(RELEASE / "data/entity_fragments.json") as f:
    frags_data = json.load(f)
fragments = frags_data["fragments"]

print(f"Loaded {len(pairs)} pairs", file=sys.stderr)

# ── Canonical prompt components (identical to GLM-5.2 script) ───────────

NOISE_FIELDS = {"confidence", "created_at", "updated_at", "aliases", "vid", "id", "source_pk"}

TYPE_JUDGING_RULES = {
    "person": (
        "1. Timeline: Do their service periods overlap?\n"
        "2. Location: Are they associated with the same locations?\n"
        "3. Organization: Do they belong to the same military unit or organization?\n"
        "4. Social circle: Do they share connections to the same people?\n"
        "Key: Same name + different organization + different role → likely DIFFERENT."
    ),
    "equipment": (
        "1. Model/Designation: Same hull number or designation = same equipment.\n"
        "   Different hull number = ALWAYS different, regardless of name similarity.\n"
        "2. Deployment: Same home port or operating area?\n"
        "3. Parent unit: Same squadron, fleet, or command?\n"
        "4. Technical specs: Same class, variant, or model series?\n"
        "Key: Designation is the strongest identifier; KC-46 ≈ KC-46A but EA-18 ≠ EA-18G."
    ),
    "location": (
        "1. Coordinates: Same or very close geographic coordinates?\n"
        "2. Administrative hierarchy: Is one a sub-area or parent of the other?\n"
        "3. Containment: Could they be different names for the same place?\n"
        "4. Alias mapping: Are the names known synonyms?\n"
        "Key: Arabian Gulf = Persian Gulf, Formosa Strait = Taiwan Strait."
    ),
    "event": (
        "1. Time period: Same dates or overlapping timeframe?\n"
        "2. Location: Same region or geographic area?\n"
        "3. Participants: Same military units, ships, or personnel involved?\n"
        "4. Event type: Same category (operation, exercise, incident)?\n"
        "Key: Generic event names may refer to DIFFERENT occurrences.\n"
        "Different dates/locations/participants = ALWAYS different events."
    ),
    "organization": (
        "1. Organizational entity: Same institution or unit?\n"
        "2. Hierarchy level: Same level in command structure?\n"
        "3. Location: Same registered address or operating region?\n"
        "4. Business scope: Same industry or functional area?\n"
        "Key: Abbreviations may refer to different organizations.\n"
        "VFA-27 ≠ VA-27; check parent organization and mission."
    ),
}


def clean_attributes(props: dict) -> dict:
    if not props:
        return {}
    cleaned = {}
    for k, v in props.items():
        if k in NOISE_FIELDS:
            continue
        v_str = str(v).strip() if v else ""
        if v_str and v_str not in ("None", "__EMPTY__", "__NULL__", "{}"):
            cleaned[k] = v
    return cleaned


def format_attributes(props: dict) -> str:
    lines = []
    for k, v in sorted(props.items()):
        v_str = str(v).strip() if v else ""
        if v_str:
            lines.append(f"  {k}: {v_str}")
    return "\n".join(lines) if lines else "  (no attributes)"


def format_fragments(frags: list) -> str:
    if not frags:
        return "  (none)"
    lines = []
    for i, f in enumerate(frags[:2], 1):
        if isinstance(f, str):
            lines.append(f'  Fragment {i}: "{f[:300]}"')
        elif isinstance(f, dict):
            text = f.get("text", f.get("fragment", ""))
            lines.append(f'  Fragment {i}: "{str(text)[:300]}"')
    return "\n".join(lines) if lines else "  (none)"


def build_canonical_prompt(pair: dict) -> str:
    etype = pair.get("type_a", "")
    pa = clean_attributes(pair.get("props_a", {}))
    pb = clean_attributes(pair.get("props_b", {}))
    fa = fragments.get(pair.get("vid_a", ""), [])
    fb = fragments.get(pair.get("vid_b", ""), [])
    rules = TYPE_JUDGING_RULES.get(etype.lower(), "Compare all available attributes.")

    return f"""You are an expert intelligence analyst performing entity resolution.
Determine whether the two entities below refer to the SAME real-world entity or DIFFERENT entities.

Entity Type: {etype}

Entity A:
{format_attributes(pa)}

Entity B:
{format_attributes(pb)}

Source text for Entity A:
{format_fragments(fa)}

Source text for Entity B:
{format_fragments(fb)}

Judging procedure for {etype}:
{rules}

Based on ALL available evidence, output EXACTLY one word:
  "same" — if they are the same real-world entity
  "different" — if they are different entities
  "unknown" — if there is genuinely insufficient information to decide"""


# ── API call with retry ─────────────────────────────────────────────────

def call_claude(prompt: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.0,
                },
                timeout=120,
            )

            if resp.status_code == 429:
                wait = min(10 * (2 ** attempt), 120)
                print(f"    429 rate limited, waiting {wait}s (attempt {attempt+1})", file=sys.stderr)
                time.sleep(wait)
                continue

            if resp.status_code >= 500:
                wait = min(5 * (2 ** attempt), 60)
                print(f"    HTTP {resp.status_code}, retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")

            raw = resp.json()["choices"][0]["message"]["content"].strip().lower()
            return raw

        except requests.exceptions.Timeout:
            wait = min(5 * (2 ** attempt), 60)
            print(f"    Timeout, retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = min(5 * (2 ** attempt), 60)
                print(f"    Error: {e}, retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
            else:
                raise

    raise Exception(f"Failed after {max_retries} retries")


def parse_verdict(raw: str) -> str:
    raw = raw.lower().strip()
    if raw.startswith("same") or raw == "same":
        return "same"
    if raw.startswith("different") or raw.startswith("differ"):
        return "different"
    if raw.startswith("unknown"):
        return "unknown"
    if "different" in raw and "same" not in raw:
        return "different"
    if "same" in raw and "different" not in raw:
        return "same"
    if "unknown" in raw or "insufficient" in raw or "uncertain" in raw:
        return "unknown"
    return "unknown"


# ── Main loop ───────────────────────────────────────────────────────────

if CKPT_PATH.exists():
    with open(CKPT_PATH) as f:
        results = json.load(f)
    print(f"Resuming: {len(results)} already annotated", file=sys.stderr)
else:
    results = {}

total = len(pairs)
todo = [p for p in pairs if p["pair_id"] not in results]
print(f"To annotate: {len(todo)}/{total}", file=sys.stderr)

errors = 0
start = time.time()

for i, pair in enumerate(todo):
    pid = pair["pair_id"]

    try:
        prompt = build_canonical_prompt(pair)
        raw = call_claude(prompt)
        verdict = parse_verdict(raw)

        results[pid] = {
            "verdict": verdict,
            "raw": raw[:200],
            "type": pair.get("type_a", "unknown"),
        }

        done = len(results)
        if done % 10 == 0:
            with open(CKPT_PATH, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            elapsed = time.time() - start
            completed_this_run = done - (total - len(todo))
            rate = completed_this_run / max(elapsed, 1)
            eta = (total - done) / max(rate, 0.01)
            print(f"  [{done}/{total}] {pid} → {verdict} "
                  f"({elapsed:.0f}s, ETA {eta/60:.0f}min, {errors} errors)",
                  file=sys.stderr)

        time.sleep(0.3)

    except Exception as e:
        done = len(results)
        print(f"  [{done}/{total}] {pid} FAILED: {e}", file=sys.stderr)
        errors += 1
        results[pid] = {
            "verdict": "error",
            "raw": str(e)[:200],
            "type": pair.get("type_a", "unknown"),
        }
        time.sleep(2)

# Save final
with open(CKPT_PATH, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
with open(FINAL_PATH, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# ── Summary ─────────────────────────────────────────────────────────────

print(f"\n{'='*60}", file=sys.stderr)
print(f"DONE: {len(results)} annotated, {errors} errors", file=sys.stderr)
print(f"Output: {FINAL_PATH}", file=sys.stderr)

valid = {k: v for k, v in results.items() if v["verdict"] in ("same", "different", "unknown")}
dist = Counter(v["verdict"] for v in valid.values())
print(f"\nVerdict distribution: {dict(dist)}", file=sys.stderr)

print(f"\nPer-type:", file=sys.stderr)
for t in ["organization", "person", "location", "equipment", "event"]:
    tv = Counter(v["verdict"] for v in valid.values() if v["type"] == t)
    n = sum(tv.values())
    if n > 0:
        print(f"  {t:<15}: same={tv.get('same',0):4d} diff={tv.get('different',0):4d} "
              f"unk={tv.get('unknown',0):4d} (n={n})", file=sys.stderr)
