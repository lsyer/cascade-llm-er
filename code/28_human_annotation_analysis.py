#!/usr/bin/env python3
"""Extract human annotations from pending_entities and analyze 3-way agreement."""
import json, psycopg2

PREFIX = "POSTGRES_PASSW"
SUFFIX = "ORD="
with open("/root/workspace/usn-monitor/.env") as f:
    for line in f:
        if line.startswith(PREFIX + SUFFIX):
            PASS = line.strip().split(SUFFIX, 1)[1]
            break

conn = psycopg2.connect(host="127.0.0.1", port=15432, dbname="usn_monitor", user="usn", password=PASS)
cur = conn.cursor()

# Get all cross-model records with their resolution
cur.execute("""
    SELECT id, entity_type, vid_a, vid_b, name_a, name_b,
           status, llm_verdict, resolved_vid, notes,
           raw_data, candidates
    FROM pending_entities 
    WHERE notes LIKE '%Cross-model%'
    ORDER BY id
""")
rows = cur.fetchall()
col_names = [desc[0] for desc in cur.description]

# Load GPT-5.4 annotation results for cross-reference
with open("/root/workspace/tanshu_docs/experiments/release/results/gpt54_annotation_300.json") as f:
    gpt54 = json.load(f)

results = []
for row in rows:
    d = dict(zip(col_names, row))
    rd = d["raw_data"] if isinstance(d["raw_data"], dict) else json.loads(d["raw_data"] or "{}")
    pair_id = rd.get("pair_id", "")
    
    # Parse candidates (GLM-5.2 vs GPT-5.4 labels)
    cands = d["candidates"] if isinstance(d["candidates"], list) else json.loads(d["candidates"] or "[]")
    glm_label = ""
    gpt_label = ""
    for c in cands:
        if isinstance(c, dict):
            if c.get("model") == "glm-5.2":
                glm_label = c.get("verdict", "")
            elif c.get("model") == "gpt-5.4":
                gpt_label = c.get("verdict", "")
    
    # Also check GPT-5.4 results file
    if pair_id in gpt54:
        gpt_label = gpt54[pair_id]["verdict"]
        glm_label = gpt54[pair_id]["glm_label"]
    
    # Determine human verdict from resolution
    status = d["status"]
    resolved_vid = d["resolved_vid"]
    
    if status == "merged":
        human_verdict = "same"
    elif status == "disambig":
        human_verdict = "different"
    elif status == "discarded":
        human_verdict = "discard"  # means the pending record itself is meaningless
    elif status == "resolved":
        human_verdict = "resolved_other"
    else:
        human_verdict = f"unknown({status})"
    
    results.append({
        "pair_id": pair_id,
        "id": d["id"],
        "type": d["entity_type"],
        "name_a": d["name_a"],
        "name_b": d["name_b"],
        "glm_label": glm_label,
        "gpt_label": gpt_label,
        "human_verdict": human_verdict,
        "status": status,
        "resolved_vid": resolved_vid,
    })

print(f"=== Human annotation results for cross-model records ===")
print(f"Total: {len(results)}")

# Status distribution
from collections import Counter
status_dist = Counter(r["status"] for r in results)
print(f"\nStatus distribution:")
for s, c in status_dist.most_common():
    print(f"  {s}: {c}")

# Map human verdicts to same/different for comparison
human_map = {"merged": "same", "disambig": "different"}
valid = [r for r in results if r["status"] in ("merged", "disambig")]
print(f"\nValid for 3-way comparison: {len(valid)} (merged/disambig)")

# 3-way agreement
all_agree = 0
glm_human_agree = 0
gpt_human_agree = 0
glm_gpt_agree = 0
glm_correct = 0
gpt_correct = 0
total_valid = len(valid)

type_results = {}  # per-type analysis

for r in valid:
    glm = r["glm_label"].lower().strip()
    gpt = r["gpt_label"].lower().strip()
    human = human_map[r["status"]]
    
    etype = r["type"]
    if etype not in type_results:
        type_results[etype] = {"total": 0, "glm_correct": 0, "gpt_correct": 0, "glm_gpt_agree": 0, "all_agree": 0}
    type_results[etype]["total"] += 1
    
    if glm == human:
        glm_correct += 1
        type_results[etype]["glm_correct"] += 1
    if gpt == human:
        gpt_correct += 1
        type_results[etype]["gpt_correct"] += 1
    if glm == gpt:
        glm_gpt_agree += 1
        type_results[etype]["glm_gpt_agree"] += 1
    if glm == human and gpt == human:
        all_agree += 1
        type_results[etype]["all_agree"] += 1

print(f"\n=== 3-WAY AGREEMENT (human as ground truth) ===")
print(f"Total valid: {total_valid}")
print(f"GLM-5.2 vs Human agreement: {glm_correct}/{total_valid} = {glm_correct/total_valid*100:.1f}%")
print(f"GPT-5.4 vs Human agreement: {gpt_correct}/{total_valid} = {gpt_correct/total_valid*100:.1f}%")
print(f"GLM-5.2 vs GPT-5.4 agreement: {glm_gpt_agree}/{total_valid} = {glm_gpt_agree/total_valid*100:.1f}%")
print(f"All three agree: {all_agree}/{total_valid} = {all_agree/total_valid*100:.1f}%")

# On disagreement cases, who was right?
print(f"\n=== ON GLM-GPT DISAGREEMENTS, WHO WAS RIGHT? ===")
disagree_cases = [r for r in valid if r["glm_label"].lower().strip() != r["gpt_label"].lower().strip()]
print(f"Total disagreements: {len(disagree_cases)}")
glm_right = sum(1 for r in disagree_cases if human_map[r["status"]] == r["glm_label"].lower().strip())
gpt_right = sum(1 for r in disagree_cases if human_map[r["status"]] == r["gpt_label"].lower().strip())
neither = sum(1 for r in disagree_cases if human_map[r["status"]] != r["glm_label"].lower().strip() and human_map[r["status"]] != r["gpt_label"].lower().strip())
print(f"GLM-5.2 right: {glm_right} ({glm_right/len(disagree_cases)*100:.1f}%)")
print(f"GPT-5.4 right: {gpt_right} ({gpt_right/len(disagree_cases)*100:.1f}%)")

print(f"\n=== PER-TYPE BREAKDOWN ===")
for etype in sorted(type_results.keys()):
    t = type_results[etype]
    print(f"  {etype:<15} total={t['total']:>3}  GLM_acc={t['glm_correct']}/{t['total']} ({t['glm_correct']/t['total']*100:.0f}%)  GPT_acc={t['gpt_correct']}/{t['total']} ({t['gpt_correct']/t['total']*100:.0f}%)")

# Show discarded records
discarded = [r for r in results if r["status"] == "discarded"]
print(f"\n=== Discarded records (not comparable): {len(discarded)} ===")
for r in discarded[:5]:
    print(f"  id={r['id']} {r['type']} | {str(r['name_a'])[:25]} vs {str(r['name_b'])[:25]}")

# Save results
output_path = "/root/workspace/tanshu_docs/experiments/release/results/human_annotation_analysis.json"
with open(output_path, "w") as f:
    json.dump({
        "summary": {
            "total_cross_model": len(results),
            "valid_for_comparison": total_valid,
            "glm_vs_human": f"{glm_correct}/{total_valid} ({glm_correct/total_valid*100:.1f}%)",
            "gpt_vs_human": f"{gpt_correct}/{total_valid} ({gpt_correct/total_valid*100:.1f}%)",
            "on_disagreements_glm_right": glm_right,
            "on_disagreements_gpt_right": gpt_right,
            "on_disagreements_total": len(disagree_cases),
        },
        "per_type": {k: v for k, v in type_results.items()},
        "details": results,
    }, f, indent=2, ensure_ascii=False)

print(f"\nSaved to {output_path}")

cur.close()
conn.close()
