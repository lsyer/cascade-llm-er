#!/usr/bin/env python3
"""
17_l2_experiment_v3.py
L2实验v3：清洗后数据 + 原文片段 + 按类型区分规则的完整实验。

改动 vs v2：
1. 每个实体携带top-2原文片段（探数设计v1.6 §4.5）
2. 属性清洗——去掉空值/__EMPTY__/confidence/timestamp
3. 按实体类型给不同的判断规则
4. 5模型并发（GLM-5.2/glm-5/glm-4.5-air/glm-4.5/Qwen3.6）
5. 全量1598对escalate
6. checkpoint增量保存
"""

import json, os, sys, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

sys.path.insert(0, '/root/workspace/tanshu_docs/experiments/code')
from config import GLM_KEY, GLM_URL, QWEN_URL, QWEN_KEY

DATA_DIR = "/root/workspace/tanshu_docs/experiments/data"
RESULTS_DIR = "/root/workspace/tanshu_docs/experiments/results"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints_v3")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================
# Load data
# ============================================================
print("=== Step 1: Load dataset + fragments ===", flush=True)

with open(os.path.join(DATA_DIR, "dataset_v3_cleaned.json")) as f:
    dataset = json.load(f)

with open(os.path.join(DATA_DIR, "entity_fragments.json")) as f:
    frag_data = json.load(f)

fragments_map = frag_data["fragments"]

all_pairs = dataset["same_type_pairs"]

# 按L1决策分组统计
from collections import Counter
l1_dist = Counter(p["l1_decision"] for p in all_pairs)
print(f"  Total pairs: {len(all_pairs)}", flush=True)
print(f"  L1 distribution: {dict(l1_dist)}", flush=True)

# ============================================================
# Property cleaning
# ============================================================
USELESS_KEYS = {"confidence", "created_at", "updated_at", "imported_at", "latest_reported_at"}
USELESS_VALUES = {"", "__EMPTY__", "{}", "null", "None", "[]"}

def clean_props(props):
    """Remove useless keys and empty values."""
    cleaned = {}
    for k, v in props.items():
        if k in USELESS_KEYS:
            continue
        s = str(v).strip().strip('"')
        if s in USELESS_VALUES:
            continue
        cleaned[k] = s
    return cleaned

# ============================================================
# Type-specific rules
# ============================================================
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

# ============================================================
# Prompt builder
# ============================================================
def build_prompt(etype, name_a, name_b, props_a, props_b, frags_a, frags_b):
    cp_a = clean_props(props_a)
    cp_b = clean_props(props_b)
    
    rules = TYPE_RULES.get(etype, DEFAULT_RULES)
    
    # Build fragment text
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

Entity A (name="{name_a}"):
Attributes:
{props_str_a}
Source text fragments (where this entity appeared in original articles):
{fmt_fragments(frags_a, name_a)}

Entity B (name="{name_b}"):
Attributes:
{props_str_b}
Source text fragments (where this entity appeared in original articles):
{fmt_fragments(frags_b, name_b)}

Based on the attributes AND the source text evidence above, are these the SAME entity or DIFFERENT entities?

Answer with exactly one word: "SAME" or "DIFFERENT" """

# ============================================================
# API call
# ============================================================
def call_llm(url, key, model, prompt, max_tokens=256):
    for attempt in range(3):
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": max_tokens,
                },
                timeout=120,
            )
            data = resp.json()
            if "choices" in data:
                content = data["choices"][0]["message"]["content"]
                return content.strip()
            elif "error" in data:
                time.sleep(3 * (attempt + 1))
                continue
        except Exception:
            time.sleep(3)
    return "ERROR"

def parse_verdict(text):
    if not text:
        return "unknown"
    t = text.upper().strip()
    if "SAME" in t and "DIFFERENT" not in t:
        return "same"
    if "DIFFERENT" in t:
        return "different"
    return "unknown"

# ============================================================
# Pre-build prompts for all pairs
# ============================================================
print("Building prompts...", flush=True)

pair_data = []
for p in all_pairs:
    vid_a = p["vid_a"]
    vid_b = p["vid_b"]
    frags_a = fragments_map.get(vid_a, [])
    frags_b = fragments_map.get(vid_b, [])
    
    prompt = build_prompt(
        p["type_a"], p["name_a"], p["name_b"],
        p["props_a"], p["props_b"],
        frags_a, frags_b
    )
    
    pair_data.append({
        "pair_id": str(p["pair_id"]),
        "l1_decision": p["l1_decision"],
        "prompt": prompt,
    })

print(f"  Prompts built: {len(pair_data)}", flush=True)

# ============================================================
# Checkpoint helpers
# ============================================================
def load_checkpoint(model_name):
    path = os.path.join(CHECKPOINT_DIR, f"{model_name}_checkpoint.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def save_checkpoint(model_name, data):
    path = os.path.join(CHECKPOINT_DIR, f"{model_name}_checkpoint.json")
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)

# ============================================================
# Model configs — 5 models, all parallel
# ============================================================
MODEL_CONFIGS = [
    ("glm-5.2", GLM_URL, GLM_KEY, "glm-5.2", 16384, 5),
    ("glm-5", GLM_URL, GLM_KEY, "glm-5", 1024, 5),
    ("glm-4.5-air", GLM_URL, GLM_KEY, "glm-4.5-air", 1024, 5),
    ("glm-4.5", GLM_URL, GLM_KEY, "glm-4.5", 1024, 5),
    ("qwen3.6", QWEN_URL, QWEN_KEY, "qwen3-32b", 1024, 5),
]

# ============================================================
# Step 2: Run all models
# ============================================================
all_results = {}

for model_idx, (name, url, key, model_id, max_tok, workers) in enumerate(MODEL_CONFIGS):
    print(f"\n=== Step 2.{model_idx+1}: {name} ({model_id}) ===", flush=True)

    checkpoint = load_checkpoint(name)
    done_count = len(checkpoint)
    print(f"  Checkpoint: {done_count} already done", flush=True)

    remaining = [pd for pd in pair_data if pd["pair_id"] not in checkpoint]
    print(f"  Remaining: {len(remaining)} ({workers} workers)", flush=True)

    if not remaining:
        all_results[name] = checkpoint
        print(f"  All done from checkpoint", flush=True)
        continue

    def worker(pd):
        raw = call_llm(url, key, model_id, pd["prompt"], max_tokens=max_tok)
        verdict = parse_verdict(raw)
        return pd["pair_id"], {"raw": raw[:300], "verdict": verdict}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, pd) for pd in remaining]
        completed = 0
        for future in as_completed(futures):
            pid, result = future.result()
            checkpoint[pid] = result
            completed += 1
            if completed % 100 == 0:
                print(f"    {done_count + completed}/{len(pair_data)} done", flush=True)
                save_checkpoint(name, checkpoint)

    save_checkpoint(name, checkpoint)
    all_results[name] = checkpoint
    print(f"  {name}: {len(checkpoint)} total", flush=True)

# ============================================================
# Step 3: Accuracy
# ============================================================
print("\n=== Step 3: Accuracy ===", flush=True)

expert_labels = {}
for pid, data in all_results.get("glm-5.2", {}).items():
    verdict = data.get("verdict", "unknown")
    if verdict in ("same", "different"):
        expert_labels[pid] = verdict

print(f"  Expert labels (GLM-5.2): {len(expert_labels)} valid / {len(pair_data)} total", flush=True)

same_count = sum(1 for v in expert_labels.values() if v == "same")
diff_count = sum(1 for v in expert_labels.values() if v == "different")
print(f"  Distribution: same={same_count}, different={diff_count}", flush=True)

print(f"\n  Model accuracy vs expert:", flush=True)
model_accuracies = {}
for name, _, _, _, _, _ in MODEL_CONFIGS:
    if name == "glm-5.2":
        continue
    correct = 0
    total = 0
    for pid in expert_labels:
        expert = expert_labels[pid]
        pred = all_results.get(name, {}).get(pid, {}).get("verdict", "unknown")
        if pred in ("same", "different"):
            total += 1
        if pred == expert:
            correct += 1
    acc = correct / total * 100 if total else 0
    unknown = len(expert_labels) - total
    model_accuracies[name] = {
        "correct": correct, "total": total,
        "accuracy": round(acc, 1), "unknown": unknown,
    }
    print(f"    {name}: {correct}/{total} = {acc:.1f}% (unknown: {unknown})", flush=True)

# ============================================================
# Step 4: Save
# ============================================================
output = {
    "metadata": {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "v3_cleaned_with_fragments",
        "total_pairs": len(all_pairs),
        "expert_valid": len(expert_labels),
        "models": [name for name, *_ in MODEL_CONFIGS],
        "experiment": "v3_fragments_5model_all_pairs",
        "fragments_coverage": f"{frag_data['metadata']['with_fragments']}/{frag_data['metadata']['total_entities']} entities",
    },
    "l1_distribution": dict(l1_dist),
    "expert_distribution": {"same": same_count, "different": diff_count},
    "model_accuracies": model_accuracies,
}

# L1 accuracy analysis
print(f"\n=== L1 Accuracy Analysis ===", flush=True)
l1_accuracy = {}
for l1_dec in ["merge", "reject", "escalate"]:
    l1_pids = {pd["pair_id"] for pd in pair_data if pd.get("l1_decision") == l1_dec}
    expert_in_l1 = {pid: v for pid, v in expert_labels.items() if pid in l1_pids}
    if not expert_in_l1:
        continue
    
    # For merge: expert should say "same"
    # For reject: expert should say "different"
    # For escalate: no expectation (it's what L2 handles)
    if l1_dec == "merge":
        expected = "same"
    elif l1_dec == "reject":
        expected = "different"
    else:
        expected = None
    
    if expected:
        correct = sum(1 for v in expert_in_l1.values() if v == expected)
        acc = correct / len(expert_in_l1) * 100
        l1_accuracy[l1_dec] = {
            "total": len(l1_pids),
            "expert_labeled": len(expert_in_l1),
            "expected": expected,
            "correct": correct,
            "accuracy": round(acc, 1),
        }
        print(f"  L1 {l1_dec} (expect={expected}): {correct}/{len(expert_in_l1)} = {acc:.1f}% (of {len(l1_pids)} total)", flush=True)
    else:
        same_in_esc = sum(1 for v in expert_in_l1.values() if v == "same")
        diff_in_esc = sum(1 for v in expert_in_l1.values() if v == "different")
        l1_accuracy[l1_dec] = {
            "total": len(l1_pids),
            "expert_labeled": len(expert_in_l1),
            "same": same_in_esc,
            "different": diff_in_esc,
        }
        print(f"  L1 {l1_dec}: {len(expert_in_l1)} labeled (same={same_in_esc}, different={diff_in_esc})", flush=True)

output["l1_accuracy"] = l1_accuracy

# System overall accuracy (L1+L2 pipeline)
print(f"\n=== System Pipeline Accuracy ===", flush=True)
# L1 correct = merge→same + reject→different
l1_merge_correct = l1_accuracy.get("merge", {}).get("correct", 0)
l1_reject_correct = l1_accuracy.get("reject", {}).get("correct", 0)
l1_correct_total = l1_merge_correct + l1_reject_correct
l1_total_labeled = l1_accuracy.get("merge", {}).get("expert_labeled", 0) + l1_accuracy.get("reject", {}).get("expert_labeled", 0)

# L2 (escalate): use best model accuracy
# For pipeline: L1 handles merge+reject, L2 handles escalate
# Overall = (L1 correct + L2 correct) / total labeled
if l1_total_labeled > 0:
    l1_pipeline_acc = l1_correct_total / l1_total_labeled * 100
    print(f"  L1 pipeline: {l1_correct_total}/{l1_total_labeled} = {l1_pipeline_acc:.1f}%", flush=True)

for model_name in model_accuracies:
    model_correct = model_accuracies[model_name]["correct"]
    model_total = model_accuracies[model_name]["total"]
    # Pipeline: L1 correct + this model's correct on escalate
    pipeline_correct = l1_correct_total + model_correct
    pipeline_total = l1_total_labeled + model_total
    pipeline_acc = pipeline_correct / pipeline_total * 100 if pipeline_total else 0
    print(f"  Pipeline L1+{model_name}: {pipeline_correct}/{pipeline_total} = {pipeline_acc:.1f}%", flush=True)

output_path = os.path.join(RESULTS_DIR, "l2_v3_comparison.json")
with open(output_path, "w") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n=== Results saved: {output_path} ===", flush=True)
print("=== Done ===", flush=True)
