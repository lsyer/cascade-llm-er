"""一次性去重脚本：基于 Nebula Graph 查询候选，LLM 判断合并，执行合并"""
import asyncio
import json
import httpx
import os
import sys
import logging

log = logging.getLogger("usn.dedup")

sys.path.insert(0, '/app')
from app.nebula_service import get_nebula

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4-flash")

DEDUP_PROMPT = """You are a military entity deduplication expert. For each pair, determine if they are the SAME entity.

Entity type: {entity_type}

Pairs to evaluate:
{pairs_text}

Rules:
- Same hull number = same ship, regardless of prefix (USS, USNS)
- Rank variants: Adm. = Admiral = ADM
- Name variants: Rob=Robert, Mike=Michael, Bill=William, Tom=Thomas
- Location synonyms: Arabian Gulf = Persian Gulf, Strait of Hormuz = Hormuz Strait
- "Naval Air Station X" ≠ "Naval Station X" (unless clearly same facility)
- Different cities/locations are ALWAYS different
- "Gulf" alone is ambiguous, do NOT merge with named gulfs

Return ONLY a JSON object mapping each pair key:
{{"pair_0": {{"merge": true, "keep_vid": "<vid>", "reason": "..."}}, "pair_1": {{"merge": false, "reason": "..."}}, ...}}
"""


def _esc(s):
    if s is None:
        return ""
    return str(s).replace('\\', '\\\\').replace('"', '\\"')


def _nb():
    return get_nebula()


def _fetch_props(nb, tag: str, vid: str) -> dict:
    """取一个节点的所有属性"""
    rows = nb.query_rows(f'FETCH PROP ON {tag} "{vid}" YIELD {tag}.*')
    if not rows:
        return {}
    return rows[0]


def _find_similar_pairs(nb, tag: str, name_prefix: str = "") -> list[tuple]:
    """查 Nebula 找名称相似的候选对"""
    # 取所有该类型节点
    rows = nb.query_rows(f'LOOKUP ON {tag} YIELD id(vertex) AS vid, {tag}.name AS name, {tag}.aliases AS aliases')
    entities = []
    for r in rows:
        vid = r.get("vid", "")
        name = r.get("name", "")
        aliases_str = r.get("aliases", "")
        aliases = aliases_str.split("||") if aliases_str else []
        entities.append({"vid": vid, "name": name, "aliases": aliases})

    # 找名称相似对（简单：共享词、别名交叉）
    pairs = []
    seen = set()
    for i, a in enumerate(entities):
        for j, b in enumerate(entities):
            if i >= j:
                continue
            key = (a["vid"], b["vid"])
            if key in seen:
                continue
            score = _name_similarity(a["name"], b["name"], a["aliases"], b["aliases"])
            if score > 0.3:
                seen.add(key)
                pairs.append((a, b, score))

    pairs.sort(key=lambda x: -x[2])
    return pairs


def _name_similarity(name_a: str, name_b: str, aliases_a: list, aliases_b: list) -> float:
    """简单名称相似度"""
    na = name_a.lower().strip()
    nb = name_b.lower().strip()

    # 完全相同
    if na == nb:
        return 1.0

    # 别名匹配
    all_a = {na} | {a.lower().strip() for a in aliases_a if a}
    all_b = {nb} | {b.lower().strip() for b in aliases_b if b}
    if all_a & all_b:
        return 0.95

    # 共享关键词
    words_a = set(na.split())
    words_b = set(nb.split())
    common = words_a & words_b
    if not common:
        return 0.0
    # Jaccard
    union = words_a | words_b
    return len(common) / len(union) if union else 0.0


async def llm_dedup(entity_type: str, pairs: list):
    if not LLM_API_KEY or not pairs:
        return {}

    pairs_text = ""
    for i, (a, b, sim) in enumerate(pairs):
        pairs_text += f'  pair_{i}: [{a["vid"]}] "{a["name"]}" vs [{b["vid"]}] "{b["name"]}" (similarity={sim:.2f})\n'

    prompt = DEDUP_PROMPT.format(entity_type=entity_type, pairs_text=pairs_text)

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        return json.loads(content)


def merge_entity(nb, tag: str, keep_vid: str, remove_vid: str):
    """在 Nebula 中合并两个实体"""
    print(f"  ✅ Merging {tag} {remove_vid} → {keep_vid}")

    # 1. 合并别名
    keep_props = _fetch_props(nb, tag, keep_vid)
    remove_props = _fetch_props(nb, tag, remove_vid)
    keep_aliases = (keep_props.get("aliases") or "").split("||") if keep_props.get("aliases") else []
    remove_name = remove_props.get("name", "")
    remove_aliases = (remove_props.get("aliases") or "").split("||") if remove_props.get("aliases") else []

    merged_aliases = list(set(keep_aliases + remove_aliases + [remove_name]))
    nb.query(f'UPDATE VERTEX ON {tag} "{keep_vid}" SET aliases = "{_esc("||".join(merged_aliases))}"')

    # 2. 重连 8 种独立边（remove 的边改连到 keep）
    _EDGE_TYPES = ["social", "employ", "locate", "participate", "org_relation", "use_edge", "spatial", "trace"]
    for et in _EDGE_TYPES:
        # 正向 remove→dst
        edges_from = nb.query_rows(f'GO FROM "{remove_vid}" OVER {et} YIELD {et}._dst AS dst, {et}.sub_type AS st, {et}.confidence AS conf, {et}.source AS src_new, {et}.created_at AS cat_new')
        for e in edges_from:
            dst = e.get("dst", "")
            if dst == keep_vid:
                continue
            st = e.get("st", "")
            conf = e.get("conf", "1.0")
            src_new = e.get("src_new", "")
            cat_new = e.get("cat_new", "")
            nb.query(f'INSERT EDGE {et}(sub_type,confidence,source,created_at) VALUES "{keep_vid}"->"{dst}":("{_esc(st)}",{conf},"{_esc(src_new)}","{_esc(cat_new)}")')
        # 反向 src→remove
        edges_to = nb.query_rows(f'GO FROM "{remove_vid}" OVER {et} REVERSELY YIELD {et}._src AS src, {et}.sub_type AS st, {et}.confidence AS conf, {et}.source AS src_new, {et}.created_at AS cat_new')
        for e in edges_to:
            src = e.get("src", "")
            if src == keep_vid:
                continue
            st = e.get("st", "")
            conf = e.get("conf", "1.0")
            src_new = e.get("src_new", "")
            cat_new = e.get("cat_new", "")
            nb.query(f'INSERT EDGE {et}(sub_type,confidence,source,created_at) VALUES "{src}"->"{keep_vid}":("{_esc(st)}",{conf},"{_esc(src_new)}","{_esc(cat_new)}")')

    # 3. 溯源图 trace 边重连（旧 generate 边已迁移为 trace）
    gen_to = nb.query_rows(
        f'GO FROM "{remove_vid}" OVER trace REVERSELY YIELD trace._src AS src_vid',
        space="usn_main"
    )
    for g in gen_to:
        src_vid = g.get("src_vid", "")
        if not src_vid:
            continue
        nb.query(f'INSERT EDGE trace(sub_type,confidence,source,created_at) VALUES "{src_vid}"->"{keep_vid}":("merge",1.0,"dedup_merge","")', space="usn_main")

    # 4. 删 remove 节点（含边）
    nb.query(f'DELETE VERTEX "{remove_vid}" WITH EDGE')
    print(f"  ✅ Deleted {remove_vid}")


async def dedup_tag(nb, tag: str, entity_type: str):
    print(f"\n{'='*50}")
    print(f"Deduplicating {tag} ({entity_type})")
    print(f"{'='*50}")

    pairs = _find_similar_pairs(nb, tag)
    if not pairs:
        print("  No similar pairs found.")
        return 0

    print(f"  Found {len(pairs)} similar pairs")

    merged = 0
    batch_size = 5
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        # 过滤已删除
        valid_batch = []
        for a, b, sim in batch:
            pa = _fetch_props(nb, tag, a["vid"])
            pb = _fetch_props(nb, tag, b["vid"])
            if pa and pb:
                valid_batch.append((a, b, sim))

        if not valid_batch:
            continue

        results = await llm_dedup(entity_type, valid_batch)
        for j, (a, b, sim) in enumerate(valid_batch):
            key = f"pair_{j}"
            result = results.get(key, {})
            if result.get("merge"):
                keep_vid = result["keep_vid"]
                if keep_vid not in (a["vid"], b["vid"]):
                    print(f"  ⚠️ LLM returned invalid keep_vid={keep_vid}, using {a['vid']}")
                    keep_vid = a["vid"]
                remove_vid = b["vid"] if keep_vid == a["vid"] else a["vid"]
                merge_entity(nb, tag, keep_vid, remove_vid)
                merged += 1
            else:
                reason = result.get("reason", "different")
                print(f"  Keep separate: [{a['vid']}] {a['name']} ↔ [{b['vid']}] {b['name']}")

    return merged


async def main():
    nb = _nb()
    total = 0

    # 去重 person 和 location（equipment 有 designation 精确匹配，通常不需要）
    total += await dedup_tag(nb, "person", "person")
    total += await dedup_tag(nb, "location", "location")
    total += await dedup_tag(nb, "event", "activity")

    print(f"\n{'='*50}")
    print(f"Total merged: {total}")
    print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())
