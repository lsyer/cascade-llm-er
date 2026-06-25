#!/usr/bin/env python3
"""
Experiment 26: Independent annotation with GPT-5.4 via kksj.org API.
Annotate 300 random pairs to check label agreement with GLM-5.2 (Cohen's Kappa).
"""
import json, os, time, random, requests
from pathlib import Path

# Load key
with open("/root/.kksj_key") as f:
    KEY = f.read().strip()

API_URL = "https://api.kksj.org/v1/chat/completions"
MODEL = "gpt-5.4"

BASE = Path("/root/workspace/tanshu_docs/experiments")
with open(BASE / "data/dataset_v3_cleaned.json") as f:
    ds = json.load(f)
with open(BASE / "results/checkpoints_v3/glm-5_checkpoint.json") as f:
    ckpt = json.load(f)
pairs = ds["same_type_pairs"]

# Select 300 random pairs (same seed as before for reproducibility)
random.seed(42)
valid_pairs = [p for p in pairs if p["pair_id"] in ckpt]
sample_pairs = random.sample(valid_pairs, 300)

from collections import Counter
type_dist = Counter(p.get("type_a","unknown") for p in sample_pairs)
print(f"Selected {len(sample_pairs)} pairs. Type dist: {dict(type_dist)}")

# Check for existing checkpoint
ckpt_path = BASE / "results/gpt54_annotation_300_checkpoint.json"
if ckpt_path.exists():
    with open(ckpt_path) as f:
        results = json.load(f)
    print(f"Resuming: {len(results)} already annotated")
else:
    results = {}

def clean_props(props):
    skip = {"confidence", "created_at", "updated_at", "aliases"}
    out = {}
    for k, v in props.items():
        if k in skip: continue
        val = str(v).strip()
        if val and val != "{}" and val != "__EMPTY__":
            out[k] = val
    return out

def format_frags(frags):
    if not frags: return "  (no source text available)"
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
    fa = pair.get("fragments_a", [])
    fb = pair.get("fragments_b", [])
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

errors = 0
start = time.time()

for i, pair in enumerate(sample_pairs):
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
            timeout=90,
        )

        if resp.status_code == 429:
            print(f"  [{i+1}] Rate limited, sleeping 10s...")
            time.sleep(10)
            continue

        if resp.status_code != 200:
            print(f"  [{i+1}] HTTP {resp.status_code}: {resp.text[:100]}")
            errors += 1
            if errors > 15:
                print("Too many errors, stopping")
                break
            time.sleep(3)
            continue

        raw = resp.json()["choices"][0]["message"]["content"].strip().lower()
        if "same" in raw and "different" not in raw:
            verdict = "same"
        elif "different" in raw:
            verdict = "different"
        elif "unknown" in raw:
            verdict = "unknown"
        else:
            verdict = "unknown"

        results[pid] = {
            "verdict": verdict,
            "raw": raw[:200],
            "glm_label": ckpt[pid].get("verdict", ckpt[pid].get("raw", "")),
            "type": pair.get("type_a", "unknown"),
        }

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(sample_pairs)}] done, {elapsed:.0f}s, {errors} errors")
            with open(ckpt_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        time.sleep(0.5)

    except Exception as e:
        print(f"  [{i+1}] ERROR: {e}")
        errors += 1
        time.sleep(3)

# Save final
output_path = BASE / "results/gpt54_annotation_300.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"DONE: {len(results)} annotated, {errors} errors")

# Agreement analysis
from sklearn.metrics import cohen_kappa_score

glm_labels, gpt_labels = [], []
for pid, r in results.items():
    glm = r["glm_label"].lower().strip()
    gpt = r["verdict"]
    if glm in ("same", "different") and gpt in ("same", "different"):
        glm_labels.append(glm)
        gpt_labels.append(gpt)

if glm_labels:
    kappa = cohen_kappa_score(glm_labels, gpt_labels)
    agreement = sum(1 for g, p in zip(glm_labels, gpt_labels) if g == p) / len(glm_labels) * 100
    print(f"\nAgreement (excl unknown): {len(glm_labels)} pairs")
    print(f"  Raw agreement: {agreement:.1f}%")
    print(f"  Cohen's Kappa: {kappa:.3f}")

    tp = sum(1 for g, p in zip(glm_labels, gpt_labels) if g == "same" and p == "same")
    tn = sum(1 for g, p in zip(glm_labels, gpt_labels) if g == "different" and p == "different")
    fp = sum(1 for g, p in zip(glm_labels, gpt_labels) if g == "different" and p == "same")
    fn = sum(1 for g, p in zip(glm_labels, gpt_labels) if g == "same" and p == "different")
    print(f"\nConfusion matrix (GLM-5.2 vs GPT-5.4):")
    print(f"                 GPT=same  GPT=diff")
    print(f"  GLM-5.2=same    {tp:>5}      {fn:>5}")
    print(f"  GLM-5.2=diff    {fp:>5}      {tn:>5}")

    print(f"\nPer-type agreement:")
    for t in ["organization", "person", "location", "equipment", "event"]:
        tg, tp2 = [], []
        for pid, r in results.items():
            if r["type"] != t: continue
            glm = r["glm_label"].lower().strip()
            gpt = r["verdict"]
            if glm in ("same", "different") and gpt in ("same", "different"):
                tg.append(glm)
                tp2.append(gpt)
        if tg:
            ta = sum(1 for g, p in zip(tg, tp2) if g == p) / len(tg) * 100
            print(f"  {t:<15}: {ta:.1f}% ({len(tg)} pairs)")
