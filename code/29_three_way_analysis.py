#!/usr/bin/env python3
"""3-way agreement analysis: GLM-5.2 vs GPT-5.4 vs Human."""
import json, psycopg2
from collections import Counter

PREFIX = "POSTGRES_PASSW"
SUFFIX = "ORD="
with open("/root/workspace/usn-monitor/.env") as f:
    for line in f:
        if line.startswith(PREFIX + SUFFIX):
            PASS = line.strip().split(SUFFIX, 1)[1]
            break

conn = psycopg2.connect(host="127.0.0.1", port=15432, dbname="usn_monitor", user="usn", password=PASS)
cur = conn.cursor()

# Get all records with pair_id (our cross-model ones)
cur.execute("""
    SELECT id, entity_type, status, notes, raw_data::text, candidates::text
    FROM pending_entities 
    WHERE raw_data::text LIKE '%pair_id%'
    ORDER BY id
""")
rows = cur.fetchall()

# Load GPT-5.4 annotation for reference
with open("/root/workspace/tanshu_docs/experiments/results/gpt54_annotation_300.json") as f:
    gpt54 = json.load(f)

results = []
for row in rows:
    rd = json.loads(row[4]) if isinstance(row[4], str) else row[4]
    pair_id = rd.get("pair_id", "")
    cands = json.loads(row[5]) if isinstance(row[5], str) else (row[5] or [])
    
    glm_label = ""
    gpt_label = ""
    for c in cands:
        if isinstance(c, dict):
            if c.get("model") == "glm-5.2": glm_label = c.get("verdict", "")
            if c.get("model") == "gpt-5.4": gpt_label = c.get("verdict", "")
    # Fallback to GPT-5.4 results file
    if pair_id in gpt54:
        gpt_label = gpt_label or gpt54[pair_id]["verdict"]
        glm_label = glm_label or gpt54[pair_id]["glm_label"]
    
    status = row[2]
    if status == "merged":
        human = "same"
    elif status == "disambig":
        human = "different"
    elif status == "discarded":
        human = "discard"  # not a real judgment on same/different
    else:
        human = f"other({status})"
    
    results.append({
        "pair_id": pair_id, "type": row[1], "status": status,
        "glm": glm_label.lower().strip(), "gpt": gpt_label.lower().strip(),
        "human": human,
    })

print(f"=== TOTAL: {len(results)} cross-model records ===\n")
status_dist = Counter(r["status"] for r in results)
print("Resolution status:")
for s, c in status_dist.most_common():
    print(f"  {s}: {c}")

# Filter to comparable: human said same/different
valid = [r for r in results if r["human"] in ("same", "different")]
discarded = [r for r in results if r["human"] == "discard"]
print(f"\nComparable (same/different): {len(valid)}")
print(f"Discarded (no judgment): {len(discarded)}")

# 3-way analysis
glm_correct = sum(1 for r in valid if r["glm"] == r["human"])
gpt_correct = sum(1 for r in valid if r["gpt"] == r["human"])
glm_gpt_agree = sum(1 for r in valid if r["glm"] == r["gpt"])
total = len(valid)

print(f"\n{'='*60}")
print(f"3-WAY AGREEMENT (Human as Ground Truth, n={total})")
print(f"{'='*60}")
print(f"GLM-5.2 accuracy: {glm_correct}/{total} = {glm_correct/total*100:.1f}%")
print(f"GPT-5.4 accuracy: {gpt_correct}/{total} = {gpt_correct/total*100:.1f}%")
print(f"GLM-GPT raw agree: {glm_gpt_agree}/{total} = {glm_gpt_agree/total*100:.1f}%")

# On disagreements, who was right?
disagree = [r for r in valid if r["glm"] != r["gpt"]]
print(f"\nOn GLM-GPT disagreements (n={len(disagree)}):")
glm_right = sum(1 for r in disagree if r["glm"] == r["human"])
gpt_right = sum(1 for r in disagree if r["gpt"] == r["human"])
neither = sum(1 for r in disagree if r["glm"] != r["human"] and r["gpt"] != r["human"])
print(f"  GLM-5.2 right: {glm_right} ({glm_right/len(disagree)*100:.0f}%)")
print(f"  GPT-5.4 right: {gpt_right} ({gpt_right/len(disagree)*100:.0f}%)")
print(f"  Neither:       {neither}")

# Per-type
print(f"\n{'='*60}")
print("PER-TYPE BREAKDOWN")
print(f"{'='*60}")
for etype in sorted(set(r["type"] for r in valid)):
    tv = [r for r in valid if r["type"] == etype]
    if not tv: continue
    ga = sum(1 for r in tv if r["glm"] == r["human"])
    pa = sum(1 for r in tv if r["gpt"] == r["human"])
    print(f"  {etype:<15} n={len(tv):>3}  GLM_acc={ga/len(tv)*100:.0f}%  GPT_acc={pa/len(tv)*100:.0f}%")

# GPT-5.4 unknowns
unknowns = [r for r in results if r["gpt"] == "unknown"]
print(f"\nGPT-5.4 returned 'unknown': {len(unknowns)}")
for r in unknowns:
    print(f"  {r['pair_id']:<12} {r['type']:<13} GLM={r['glm']:<10} Human={r['human']:<10}")

# Discarded detail
print(f"\n{'='*60}")
print(f"DISCARDED RECORDS (n={len(discarded)})")
print(f"{'='*60}")
for r in discarded:
    print(f"  {r['pair_id']:<12} {r['type']:<13} GLM={r['glm']:<10} GPT={r['gpt']:<10}")

# Save full analysis
output = {
    "summary": {
        "total": len(results),
        "comparable": total,
        "discarded": len(discarded),
        "glm_accuracy": f"{glm_correct}/{total} ({glm_correct/total*100:.1f}%)",
        "gpt_accuracy": f"{gpt_correct}/{total} ({gpt_correct/total*100:.1f}%)",
        "on_disagreements": {"glm_right": glm_right, "gpt_right": gpt_right, "total": len(disagree)},
    },
    "details": results,
}
with open("/root/workspace/tanshu_docs/experiments/results/human_annotation_analysis.json", "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

cur.close()
conn.close()
