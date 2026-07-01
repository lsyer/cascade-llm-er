#!/usr/bin/env python3
"""
Experiment 30: Probe annotation with Claude Opus 4.8 via kksj.org API.
Annotate 50 random pairs (same seed as GPT-5.4 experiment) to check
agreement with GLM-5.2 before committing to full 2639-pair annotation.
"""
import json, time, random, requests
from pathlib import Path
from collections import Counter

# Load key
with open("/root/.kksj_key") as f:
    KEY = f.read().strip()

API_URL = "https://api.kksj.org/v1/chat/completions"
MODEL = "claude-opus-4-8"

BASE = Path("/root/workspace/tanshu_docs/experiments/release")
with open(BASE / "data/dataset_v3_cleaned.json") as f:
    ds = json.load(f)
with open(BASE / "results/scored_pairs.json") as f:
    scored = json.load(f)
with open(BASE / "results/checkpoints_v3/glm-5.2_checkpoint.json") as f:
    glm52_ckpt = json.load(f)

pairs = ds["same_type_pairs"]
scored_map = {p["pair_id"]: p for p in scored}

# Load fragments
with open(BASE / "data/entity_fragments.json") as f:
    frags_data = json.load(f)
fragments = frags_data["fragments"]

# Same seed as GPT-5.4 experiment for overlap comparability
random.seed(42)
valid_pairs = [p for p in pairs if p["pair_id"] in glm52_ckpt]
all_sample = random.sample(valid_pairs, 300)  # same 300 as GPT-5.4
probe_pairs = all_sample[:50]  # first 50 for probe

type_dist = Counter(p.get("type_a", "unknown") for p in probe_pairs)
print(f"Probe: {len(probe_pairs)} pairs. Type dist: {dict(type_dist)}")

# ── Prompt helpers (same structure as GPT-5.4 experiment) ──────────────

def clean_props(props):
    skip = {"confidence", "created_at", "updated_at", "aliases"}
    out = {}
    for k, v in props.items():
        if k in skip:
            continue
        val = str(v).strip()
        if val and val != "{}" and val != "__EMPTY__":
            out[k] = val
    return out


def get_fragments(vid):
    return fragments.get(vid, [])


def format_frags(frags):
    if not frags:
        return "  (no source text available)"
    lines = []
    for i, f in enumerate(frags[:2], 1):
        if isinstance(f, str):
            lines.append(f'  Fragment {i}: "{f[:300]}"')
        elif isinstance(f, dict):
            text = f.get("text", f.get("fragment", ""))
            lines.append(f'  Fragment {i}: "{str(text)[:300]}"')
    return "\n".join(lines) if lines else "  (no source text available)"


RULES = {
    "person": "Key identifiers: phone/ID number, social media. Supporting: timeline (career history), location, organization affiliation, rank. Two persons are SAME if they share a unique identifier OR if name+org+timeline all align. DIFFERENT if same name but different organization in same time period.",
    "equipment": "Key identifiers: hull number/designation code. Supporting: model, deployment location, affiliation, technical parameters. Two equipment are SAME if they share a designation/hull number. DIFFERENT if different model series (e.g., P4 vs P5) even if same manufacturer.",
    "location": "Key identifiers: coordinates (lat/lon). Supporting: administrative code, containment hierarchy, alias. Two locations are SAME if coordinates match or one is within the other AND names refer to the same place. DIFFERENT if different administrative hierarchy levels.",
    "event": "Key identifiers: case number. Supporting: time, location, participants, event type. Two events are SAME only if time+location+type all align. DIFFERENT if different year, different region, or different participants even if similar description.",
    "organization": "Key identifiers: unified credit code. Supporting: hierarchy level, registered address, time window, business scope. Two orgs are SAME if credit code matches. DIFFERENT if registered addresses are completely different.",
}


def build_prompt(pair):
    etype = pair.get("type_a", "")
    pa = clean_props(pair.get("props_a", {}))
    pb = clean_props(pair.get("props_b", {}))
    fa = get_fragments(pair.get("vid_a", ""))
    fb = get_fragments(pair.get("vid_b", ""))
    rule = RULES.get(etype.lower(), "Compare all available attributes.")
    return f"""You are an expert entity resolver for a knowledge graph. Determine whether the two entities below refer to the SAME real-world entity or DIFFERENT entities.

Entity Type: {etype}

Entity A attributes:
{json.dumps(pa, indent=2, ensure_ascii=False)}

Entity B attributes:
{json.dumps(pb, indent=2, ensure_ascii=False)}

Source text for Entity A:
{format_frags(fa)}

Source text for Entity B:
{format_frags(fb)}

Judging rules for {etype}: {rule}

Based on ALL available evidence, output EXACTLY one word: "same" or "different".
If there is genuinely insufficient information to decide, output "unknown"."""


# ── Run probe ───────────────────────────────────────────────────────────

ckpt_path = BASE / "results/claude_opus_probe_50.json"
results = {}
errors = 0
start = time.time()

for i, pair in enumerate(probe_pairs):
    pid = pair["pair_id"]
    if pid in results:
        continue

    try:
        resp = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": build_prompt(pair)}],
                "max_tokens": 100,
                "temperature": 0.0,
            },
            timeout=120,
        )

        if resp.status_code == 429:
            print(f"  [{i+1}] Rate limited, sleeping 10s...")
            time.sleep(10)
            continue

        if resp.status_code != 200:
            print(f"  [{i+1}] HTTP {resp.status_code}: {resp.text[:150]}")
            errors += 1
            if errors > 10:
                print("Too many errors, stopping")
                break
            time.sleep(3)
            continue

        raw = resp.json()["choices"][0]["message"]["content"].strip().lower()
        # Parse verdict — Claude is usually clean but handle edge cases
        if "different" in raw and "same" not in raw:
            verdict = "different"
        elif "same" in raw and "different" not in raw:
            verdict = "same"
        elif "different" in raw:
            verdict = "different"  # "not the same" → different
        elif "same" in raw:
            verdict = "same"
        elif "unknown" in raw:
            verdict = "unknown"
        else:
            verdict = "unknown"

        glm_label = glm52_ckpt[pid].get("verdict", glm52_ckpt[pid].get("raw", "")).lower().strip()

        results[pid] = {
            "verdict": verdict,
            "raw": raw[:200],
            "glm_label": glm_label,
            "type": pair.get("type_a", "unknown"),
        }

        elapsed = time.time() - start
        print(f"  [{i+1}/{len(probe_pairs)}] {pid} type={pair.get('type_a','')} "
              f"claude={verdict} glm={glm_label} {'✓' if verdict==glm_label else '✗'} "
              f"({elapsed:.0f}s)")

        # Save checkpoint every 10
        if (i + 1) % 10 == 0:
            with open(ckpt_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        time.sleep(0.5)

    except Exception as e:
        print(f"  [{i+1}] ERROR: {e}")
        errors += 1
        time.sleep(3)

# Save final
with open(ckpt_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# ── Analysis ────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"PROBE COMPLETE: {len(results)} annotated, {errors} errors")
print(f"Elapsed: {time.time()-start:.0f}s")

# Verdict distribution
print(f"\nClaude verdict distribution: {Counter(r['verdict'] for r in results.values())}")

# Agreement with GLM-5.2
valid = [(r["verdict"], r["glm_label"]) for r in results.values()
         if r["verdict"] in ("same", "different") and r["glm_label"] in ("same", "different")]

if valid:
    agree = sum(1 for c, g in valid if c == g)
    print(f"\nAgreement (excl unknown): {len(valid)} pairs")
    print(f"  Raw agreement: {agree}/{len(valid)} = {agree/len(valid)*100:.1f}%")

    tp = sum(1 for c, g in valid if c == "same" and g == "same")
    tn = sum(1 for c, g in valid if c == "different" and g == "different")
    fp = sum(1 for c, g in valid if c == "same" and g == "different")
    fn = sum(1 for c, g in valid if c == "different" and g == "same")

    print(f"\nConfusion matrix (Claude vs GLM-5.2):")
    print(f"                  GLM=same  GLM=diff")
    print(f"  Claude=same     {tp:>5}      {fp:>5}")
    print(f"  Claude=diff     {fn:>5}      {tn:>5}")

    # Per-type
    print(f"\nPer-type agreement:")
    for t in ["organization", "person", "location", "equipment", "event"]:
        tv = [(r["verdict"], r["glm_label"]) for r in results.values()
              if r["type"] == t and r["verdict"] in ("same", "different")
              and r["glm_label"] in ("same", "different")]
        if tv:
            ta = sum(1 for c, g in tv if c == g) / len(tv) * 100
            print(f"  {t:<15}: {ta:.1f}% ({len(tv)} pairs)")

    unknown_count = sum(1 for r in results.values() if r["verdict"] == "unknown")
    print(f"\nUnknown rate: {unknown_count}/{len(results)} = {unknown_count/len(results)*100:.1f}%")
