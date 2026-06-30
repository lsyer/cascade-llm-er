"""全量融合消歧服务 — 从 Nebula 取所有实体，使用经 MINEC 验证的 L1 scorer 计算相似度，写入 PG 暂存队列"""
from __future__ import annotations

import os
import json
import logging
import httpx
from datetime import datetime

# 使用经离线实验验证的生产级 L1 scorer
from app.services.l1_scorer import (
    l1_score_with_name as _l1_score_fn,
    MERGE_THRESHOLD as L1_MERGE,
    REJECT_THRESHOLD as L1_REJECT,
)
from app.services.l2_judge import l2_judge as _l2_judge_fn

log = logging.getLogger("usn.fusion")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4-flash")

# 同义词表
SYNONYM_GROUPS = [
    {"Arabian Gulf", "Persian Gulf"},
    {"Formosa Strait", "Taiwan Strait"},
    {"Strait of Hormuz", "Hormuz Strait"},
    {"South China Sea", "SCS"},
    {"Sea of Japan", "East Sea"},
]

# 反向索引：name → set of synonyms
SYNONYM_MAP: dict[str, set[str]] = {}
for _grp in SYNONYM_GROUPS:
    _lower_grp = {m.lower().strip() for m in _grp}
    for _m in _grp:
        SYNONYM_MAP[_m.lower().strip()] = _lower_grp


def _esc(s):
    if s is None:
        return ""
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def _parse_aliases(val) -> list[str]:
    """解析 Nebula aliases 字段"""
    if isinstance(val, list):
        return val
    if not val or not isinstance(val, str):
        return []
    s = val.strip()
    # Nebula set format
    if s.startswith("{") and s.endswith("}"):
        inner = s[1:-1]
        if not inner:
            return []
        items = []
        for item in inner.split(","):
            item = item.strip().strip('"').strip("'")
            if item:
                items.append(item)
        return items
    # "||" format
    if "||" in s:
        return [x.strip() for x in s.split("||") if x.strip()]
    # JSON
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return [s] if s else []


def _tag_for_type(entity_type: str) -> str:
    return {
        "equipment": "equipment",
        "person": "person",
        "location": "location",
        "activity": "event",
        "organization": "organization",
    }.get(entity_type, entity_type)


# ─── Entity fetch from Nebula ───

def fetch_all_entities(nb, tag: str) -> list[dict]:
    """Fetch all entities of a given tag from Nebula."""
    nql = f'LOOKUP ON {tag} YIELD id(vertex) AS vid, {tag}.name AS name, {tag}.aliases AS aliases'
    rows = nb.query_rows(nql)
    entities = []
    for r in rows:
        vid = r.get("vid", "")
        name = r.get("name", "")
        aliases = _parse_aliases(r.get("aliases"))
        entities.append({"vid": vid, "name": name, "aliases": aliases})
    return entities


def fetch_entity_props(nb, tag: str, vid: str) -> dict:
    """Fetch all properties of a single entity."""
    row = nb.query_one(f'FETCH PROP ON {tag} "{vid}" YIELD properties(vertex) AS props')
    if row and isinstance(row, dict) and "props" in row:
        props = row["props"]
        if isinstance(props, str):
            import json
            try:
                props = json.loads(props)
            except Exception:
                pass
        return props if isinstance(props, dict) else row
    return row or {}


# ─── Similarity scoring (per entity type) ───

def _name_contains_score(name_a: str, name_b: str) -> float:
    """One name contains the other → partial score."""
    a = name_a.lower().strip()
    b = name_b.lower().strip()
    if a == b:
        return 1.0
    if a in b or b in a:
        shorter = min(len(a), len(b))
        longer = max(len(a), len(b))
        return 0.4 + 0.3 * (shorter / longer) if longer else 0.0
    return 0.0


def _alias_cross_score(aliases_a: list[str], name_a: str, aliases_b: list[str], name_b: str) -> float:
    """Alias cross-match score."""
    set_a = {name_a.lower().strip()} | {a.lower().strip() for a in aliases_a if a}
    set_b = {name_b.lower().strip()} | {b.lower().strip() for b in aliases_b if b}
    if set_a & set_b:
        return 0.3
    # Check synonym table
    for item in set_a:
        syns = SYNONYM_MAP.get(item, set())
        if syns & set_b:
            return 0.25
    return 0.0


def _designation_match(props_a: dict, props_b: dict) -> float:
    """Equipment designation match — strongest signal."""
    # "name" field in Nebula equipment stores the designation when it's the hull number
    # Check multiple possible fields
    desig_a = (props_a.get("name") or "").strip()
    desig_b = (props_b.get("name") or "").strip()
    if desig_a and desig_b and desig_a.lower() == desig_b.lower():
        return 0.40

    # Check equip_type and category
    score = 0.0
    cat_a = props_a.get("category", "") or ""
    cat_b = props_b.get("category", "") or ""
    if cat_a and cat_b and cat_a.lower() == cat_b.lower():
        score += 0.15

    et_a = props_a.get("equip_type", "") or ""
    et_b = props_b.get("equip_type", "") or ""
    if et_a and et_b and et_a.lower() == et_b.lower():
        score += 0.10

    return score


def _attribute_match(props_a: dict, props_b: dict, fields: list[tuple[str, float]]) -> float:
    """Generic attribute matching with weights."""
    score = 0.0
    for field, weight in fields:
        va = (props_a.get(field) or "").strip().lower()
        vb = (props_b.get(field) or "").strip().lower()
        if va and vb and va == vb:
            score += weight
    return score


def compute_similarity(entity_type: str, ent_a: dict, ent_b: dict,
                       props_a: dict, props_b: dict) -> tuple[float, str]:
    """使用经 MINEC 验证的 universal property overlap scorer 计算相似度。

    返回 (score, decision):
      score ∈ [-1.0, +1.0]
      decision: 'merge' | 'reject' | 'escalate'
        merge   → score ≥ 0.6
        reject  → score ≤ -0.4
        escalate → 中间区间
    """
    name_a = ent_a.get("name", "")
    name_b = ent_b.get("name", "")

    score, decision, detail = _l1_score_fn(
        name_a, name_b, props_a, props_b, entity_type,
    )
    return score, decision


# ─── Quick pre-filter ───

def _quick_related(name_a: str, name_b: str, aliases_a: list[str], aliases_b: list[str]) -> bool:
    """Quick check: are these two entities potentially related at all?
    Used to skip obviously unrelated pairs."""
    na = name_a.lower().strip()
    nb = name_b.lower().strip()

    if na == nb:
        return True

    # One name contains the other
    if na in nb or nb in na:
        return True

    # Alias overlap
    set_a = {na} | {a.lower().strip() for a in aliases_a if a}
    set_b = {nb} | {b.lower().strip() for b in aliases_b if b}
    if set_a & set_b:
        return True

    # Synonym table
    for item in set_a:
        syns = SYNONYM_MAP.get(item, set())
        if syns & set_b:
            return True

    # Shared significant words (>2 chars, excluding common stop words)
    stop = {"the", "and", "for", "uss", "usns", "hmas", "hmcs", "uss", "in", "of", "on", "at", "to"}
    words_a = {w for w in na.split() if len(w) > 2 and w not in stop}
    words_b = {w for w in nb.split() if len(w) > 2 and w not in stop}
    if words_a & words_b:
        return True

    return False


# ─── LLM judgment ───

FUSION_PROMPT = """You are a military entity fusion and disambiguation expert. For each pair, decide if they are the SAME entity (merge) or DIFFERENT entities (disambiguate).

Entity type: {entity_type}

Pairs to evaluate:
{pairs_text}

Rules:
- Same hull number/designation = same equipment. USS Abraham Lincoln (CVN-72) = Abraham Lincoln (CVN-72)
- Different hull number = different equipment. CVN-72 ≠ CVN-71
- KC-46 ≈ KC-46A (same base model), but EA-18 ≠ EA-18G (different variant)
- Name variants: Adm. = Admiral = ADM, Rob=Robert, Mike=Michael
- Location synonyms: Arabian Gulf = Persian Gulf, Formosa Strait = Taiwan Strait
- "Naval Air Station X" ≠ "Naval Station X" (unless clearly same facility)
- Different cities/locations are ALWAYS different
- Same person name + same org = likely same person
- Same person name + different org + different role = likely different person
- Same event name + same region + overlapping time = likely same event
- Same event name + different era/region = different events

Return ONLY a JSON object:
{{"pair_0": {{"action": "merge", "keep_vid": "<vid>", "reason": "..."}}, "pair_1": {{"action": "disambiguate", "reason": "..."}}, ...}}
"""


async def llm_judge_pairs(entity_type: str, pairs: list[dict]) -> dict:
    """Send pairs to LLM for merge/disambiguate judgment."""
    if not LLM_API_KEY or not pairs:
        return {}

    pairs_text = ""
    for i, p in enumerate(pairs):
        pairs_text += f'  pair_{i}: [{p["vid_a"]}] "{p["name_a"]}" vs [{p["vid_b"]}] "{p["name_b"]}" (similarity={p["similarity"]:.2f})\n'
        if p.get("props_a"):
            pairs_text += f'    A attrs: {json.dumps({k: v for k, v in p["props_a"].items() if v}, ensure_ascii=False)[:200]}\n'
        if p.get("props_b"):
            pairs_text += f'    B attrs: {json.dumps({k: v for k, v in p["props_b"].items() if v}, ensure_ascii=False)[:200]}\n'

    prompt = FUSION_PROMPT.format(entity_type=entity_type, pairs_text=pairs_text)

    try:
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
    except Exception as e:
        log.error(f"[Fusion] LLM judge failed: {e}")
        return {}


# ─── Merge execution ───

def merge_entities(nb, tag: str, keep_vid: str, remove_vid: str):
    """Merge remove_vid into keep_vid in Nebula."""
    log.info(f"[Fusion] Merging {tag} {remove_vid} → {keep_vid}")

    # 1. Merge aliases
    keep_props = fetch_entity_props(nb, tag, keep_vid)
    remove_props = fetch_entity_props(nb, tag, remove_vid)
    keep_aliases = _parse_aliases(keep_props.get("aliases"))
    remove_name = remove_props.get("name", "")
    remove_aliases = _parse_aliases(remove_props.get("aliases"))

    merged_aliases = list(set(keep_aliases + remove_aliases + [remove_name]))
    nb.query(f'UPDATE VERTEX ON {tag} "{keep_vid}" SET aliases = "{_esc("||".join(merged_aliases))}"')

    # 2. Reconnect 8 independent edges (remove → keep)
    _EDGE_TYPES = ["social", "employ", "locate", "participate", "org_relation", "use_edge", "spatial", "trace"]
    for et in _EDGE_TYPES:
        # Forward remove→dst
        edges_from = nb.query_rows(
            f'GO FROM "{remove_vid}" OVER {et} YIELD {et}._dst AS dst, {et}.sub_type AS st, {et}.confidence AS conf, {et}.source AS src_new, {et}.created_at AS cat_new'
        )
        for e in edges_from:
            dst = e.get("dst", "")
            if dst == keep_vid:
                continue
            st = e.get("st", "")
            conf = e.get("conf", "1.0")
            src_new = e.get("src_new", "")
            cat_new = e.get("cat_new", "")
            nb.query(f'INSERT EDGE {et}(sub_type,confidence,source,created_at) VALUES "{keep_vid}"->"{dst}":("{_esc(st)}",{conf},"{_esc(src_new)}","{_esc(cat_new)}")')
        # Reverse src→remove
        edges_to = nb.query_rows(
            f'GO FROM "{remove_vid}" OVER {et} REVERSELY YIELD {et}._src AS src, {et}.sub_type AS st, {et}.confidence AS conf, {et}.source AS src_new, {et}.created_at AS cat_new'
        )
        for e in edges_to:
            src = e.get("src", "")
            if src == keep_vid:
                continue
            st = e.get("st", "")
            conf = e.get("conf", "1.0")
            src_new = e.get("src_new", "")
            cat_new = e.get("cat_new", "")
            nb.query(f'INSERT EDGE {et}(sub_type,confidence,source,created_at) VALUES "{src}"->"{keep_vid}":("{_esc(st)}",{conf},"{_esc(src_new)}","{_esc(cat_new)}")')

    # 3. Reconnect trace edges (旧 generate 边已迁移为 trace)
    gen_to = nb.query_rows(
        f'GO FROM "{remove_vid}" OVER trace REVERSELY YIELD trace._src AS src_vid',
        space="usn_main"
    )
    for g in gen_to:
        src_vid = g.get("src_vid", "")
        if not src_vid:
            continue
        nb.query(
            f'INSERT EDGE trace(sub_type,confidence,source,created_at) '
            f'VALUES "{src_vid}"->"{keep_vid}":("merge",1.0,"fusion_merge","")',
            space="usn_main"
        )

    # 4. Delete remove node
    nb.query(f'DELETE VERTEX "{remove_vid}" WITH EDGE')
    log.info(f"[Fusion] Deleted {remove_vid}")
