"""LLM 实体抽取器 — 图优先架构，写 Nebula Graph"""
from __future__ import annotations

import asyncio
import os
import json
import httpx

import threading

# 全局锁：确保 LLM 调用完全串行（一次只有一个请求在飞）
_sync_llm_lock = threading.Lock()
# 异步任务级锁：确保 extract 任务不并发
_extract_semaphore = asyncio.Semaphore(1)
# L2 不确定累积器：_disambig_entity 写入，save_to_nebula 读取后清空
_l2_uncertain: list[dict] = []
import logging
import re
from datetime import date, datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

log = logging.getLogger("usn.extractor")

LLM_BASE_URL = os.environ["LLM_BASE_URL"]
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ["LLM_MODEL"]

EXTRACT_PROMPT = """你是一个军事信息抽取专家。从以下新闻文本中提取所有与美国海军/军事相关的实体。

请严格返回 JSON，包含以下五类实体和它们之间的关系：

{{
  "equipment": [
    {{"name": "USS Gerald R. Ford", "designation": "CVN-78", "equipment_type": "ship", "category": "aircraft_carrier", "status": "active", "aliases": ["福特号"]}},
    {{"name": "F-35C Lightning II", "designation": "", "equipment_type": "aircraft", "category": "fighter", "status": "", "aliases": []}},
    {{"name": "SM-6 missile", "designation": "", "equipment_type": "missile", "category": "surface_to_air", "status": "", "aliases": []}}
  ],
  "organizations": [
    {{"name": "VFA-27", "org_type": "military_unit", "country": "United States", "description": "Strike Fighter Squadron 27"}},
    {{"name": "NAVSEA", "org_type": "government_agency", "country": "United States", "description": "Naval Sea Systems Command"}},
    {{"name": "Boeing", "org_type": "company", "country": "United States", "description": ""}}
  ],
  "persons": [
    {{"name": "Adm. John Aquilino", "rank": "Admiral", "position": "Commander, USINDOPACOM", "service_branch": "US Navy"}}
  ],
  "locations": [
    {{"name": "South China Sea", "location_type": "sea_area", "country": "", "region": "INDOPACOM"}},
    {{"name": "Naval Station Norfolk", "location_type": "naval_station", "country": "United States", "region": "HOMECOM", "lat": 36.94, "lng": -76.33}}
  ],
  "activities": [
    {{"name": "Freedom of Navigation Operation", "activity_type": "operation", "start_date": "2026-05-01", "end_date": "", "region": "South China Sea", "description": ""}},
    {{"name": "RIMPAC 2026", "activity_type": "exercise", "start_date": "2026-06-01", "end_date": "2026-07-31", "region": "Pacific", "description": ""}},
    {{"name": "USS Gerald R. Ford Port Visit Singapore", "activity_type": "port_visit", "start_date": "2026-05-03", "end_date": "", "region": "Singapore", "description": ""}}
  ],
  "relations": [
    {{"subject_type": "equipment", "subject_name": "USS Gerald R. Ford", "relation": "locate", "sub_type": "located_at", "object_type": "location", "object_name": "South China Sea"}},
    {{"subject_type": "equipment", "subject_name": "USS Gerald R. Ford", "relation": "participate", "sub_type": "join", "object_type": "activity", "object_name": "Freedom of Navigation Operation"}},
    {{"subject_type": "person", "subject_name": "Adm. John Aquilino", "relation": "participate", "sub_type": "command", "object_type": "activity", "object_name": "Freedom of Navigation Operation"}},
    {{"subject_type": "equipment", "subject_name": "USS Higgins", "relation": "org_relation", "sub_type": "equipped_with", "object_type": "equipment", "object_name": "USS Gerald R. Ford"}},
    {{"subject_type": "equipment", "subject_name": "SM-6 missile", "relation": "participate", "sub_type": "used_in", "object_type": "activity", "object_name": "Freedom of Navigation Operation"}}
  ]
}}

实体说明（严格分类，不要混淆）：
- **equipment(器装)**: 只有具体的物理装备——舰船、飞机、导弹、车辆、武器系统、雷达、无人机等。**军事编制单位（如 VFA-27、Carrier Air Wing 8、3rd Marine Division）不是 equipment，必须归入 organizations**。军事行动名称（如"Operation Freedom"、"Project Freedom"）也不是 equipment，必须归入 activities。造船厂/公司/机构也不要放入此类。
  equipment_type: ship/aircraft/missile/vehicle/weapon/uav/radar/system/other
- **organizations(组织机构)**: 军事编制单位（舰队、联队、中队、大队，如"VFA-27"、"7th Fleet"、"Carrier Strike Group 12"）、政府机构（如"NAVSEA"、"NATO"、"CENTCOM"）、公司企业（如"Boeing"、"Austal USA"、"Helsing"）、军事联盟（如"AUKUS"、"Five Eyes"）。只要是一个有组织的集体，不是具体个人、地点、装备或事件，就归此类。
  org_type: military_unit/company/government_agency/military_alliance/research_institution/media/other
- **persons(人员)**: 军官、官员、分析师等具体个人。提取军衔(rank)和职务(position)。注意区分同名不同人——检查军衔和职务。职务名称（如"First Sea Lord"、"SACEUR"）不是个人，应归 organizations。
- **locations(地理位置)**: 军事基地、港口、海域、海峡、空域、城市、国家等地理实体。**海军/军事组织（如"Royal Navy"、"NATO"）不是 location**。只有你确信坐标准确时才填 lat/lng（如知名基地、主要城市、海峡），不确定就留空。
  location_type: naval_station/naval_air_station/sea_area/strait/port/region/city/country/airbase/shipyard/other
- **activities(活动)**: 军事行动、作战计划、部署、演习、港口访问、事故、对峙事件、护航任务、巡逻、封锁等。任何有名称的军事行动/计划/任务都归此类。
  **命名规范（严格遵守）**：name 字段必须是规范化的实体名称，不是句子片段或动词短语。
  - 正确："RIMPAC 2026"、"Operation Epic Fury"、"Gerald R. Ford CSG Deployment 2025-2026"、"Tomahawk Strike on Yemen"、"Maduro Capture Operation"
  - 错误："USS Ford arrives in Singapore"（句子）、"blockade"（太泛）、"carrier strike group deployment beyond the First Island Chain"（过长短语）
  - 规则：(1) 首字母大写（Title Case）;(2) 如果文章没有给出正式名称，用"主体+动作+对象"格式构造一个名词短语（如"US Strikes on Iranian Facilities"而不是"strikes against Iranian military facilities"）;(3) 不要超过6个词;(4) 不要用动词开头（如"deploying"、"launching"）。
  活动有层级概念：大的作战行动（如"Operation Epic Fury"）包含多个子活动（空袭、封锁、护航等）。如果文章提到了大行动和子活动，都要提取，并用 participate 的 part_of 关系关联。
  抽取粒度：只提取文章中有明确描述、值得单独记录的活动。不要过度拆分（如"舰船启动引擎"不算），但一个有名称的子行动（如"Tomahawk Strike on Yemen"）值得提取。
  activity_type: operation/deployment/exercise/port_visit/transit/incident/surveillance/patrol/blockade/strike/escort/other

**relations 关系（严格遵守 8 大类 + sub_type 分类体系）：**

每条关系必须同时指定 `relation`（大类）和 `sub_type`（子类）：

| relation | 含义 | sub_type 枚举 |
|----------|------|--------------|
| social | 人际关系 | family, associate, friend, contact, other |
| employ | 职属关系 | member_of, lead, found, invest, other |
| locate | 位置关系 | located_at, visited, other |
| participate | 参与关系 | command, join, used_in, other |
| org_relation | 组织间/装备间关系 | part_of, partner, rival, supply, equipped_with, variant_of, other |
| use | 使用码址 | phone, email, social_media, id_card, other |
| spatial | 空间关系 | contains, adjacent, near, other |
| trace | 溯源/数据血缘 | generate, contain, other |

**分类指导：**
- 装备→地点（部署、驻扎、停靠、位于）→ locate
- 装备/人员→活动（参与、指挥、被使用于）→ participate
- 装备→装备（护航、配备、变体）→ org_relation
- 人员→机构（就职、领导、指挥）→ employ
- 活动→活动（子活动属于大行动）→ participate, sub_type=other
- 不要提取 "mentioned"/"related_to" 等无意义关系
- 拿不准的 sub_type 统一用 other

如果某类实体没有，返回空数组。只返回 JSON，不要其他文字。

新闻标题：{title}

新闻内容：
{content}
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Nebula 服务（同步，在 extractor 里直接用）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_nebula_pool = None


def _get_pool():
    global _nebula_pool
    if _nebula_pool is None:
        from nebula3.gclient.net import ConnectionPool
        from nebula3.Config import Config
        _nebula_pool = ConnectionPool()
        _nebula_pool.init([("usn-nebula-graphd", 9669)], Config())
        log.info("[Nebula] Connection pool initialized for extractor")
    return _nebula_pool


def _nb_query(nql: str, space: str = "usn_main"):
    pool = _get_pool()
    sess = pool.get_session("root", "nebula")
    sess.execute(f"USE {space}")
    r = sess.execute(nql)
    sess.release()
    return r


def _nb_rows(nql: str, space: str = "usn_main") -> list[dict]:
    r = _nb_query(nql, space)
    if not r.is_succeeded():
        return []
    keys = r.keys()
    rows = []
    for i in range(r.row_size()):
        row = {}
        for j, key in enumerate(keys):
            val = r.row_values(i)[j]
            if val.is_empty():
                row[key] = None
                continue
            try:
                row[key] = val.as_string()
            except Exception:
                try:
                    row[key] = val.as_int()
                except Exception:
                    try:
                        row[key] = val.as_double()
                    except Exception:
                        # Last resort: check for __NULL__ sentinel
                        s = str(val)
                        row[key] = None if '__NULL__' in s else s
        rows.append(row)
    return rows


def _esc(s):
    if s is None:
        return ""
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def _vid_prefix(entity_type: str) -> str:
    return {
        "equipment": "equip",
        "person": "person",
        "location": "loc",
        "activity": "event",
        "organization": "org",
    }.get(entity_type, "entity")


def _tag_name(entity_type: str) -> str:
    return {
        "equipment": "equipment",
        "person": "person",
        "location": "location",
        "activity": "event",
        "organization": "organization",
    }.get(entity_type, "unknown")


def _type_from_vid(vid: str) -> str:
    prefix = vid.split("_")[0]
    return {
        "equip": "equipment", "person": "person", "loc": "location",
        "event": "activity", "org": "organization",
    }.get(prefix, "unknown")


def _id_from_vid(vid: str) -> str:
    parts = vid.split("_", 1)
    return parts[1] if len(parts) > 1 else vid


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 两层消歧：L1 规则层（双向评分）+ L2 LLM 判断
# 使用经 MINEC 离线实验验证的生产级模块（l1_scorer.py + l2_judge.py）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from app.services.l1_scorer import (
    l1_score_with_name as _l1_score_fn,
    l1_score_adaptive as _l1_score_adaptive_fn,
    MERGE_THRESHOLD as _L1_CONFIRM_SAME,
    REJECT_THRESHOLD as _L1_CONFIRM_DIFF,
)
from app.services.l2_judge import l2_judge as _l2_judge_fn


def _normalize(s: str) -> str:
    """统一比较：去空格、转小写"""
    return (s or "").strip().lower()


def _years_close(date_a: str, date_b: str, threshold: int = 1) -> bool:
    """两个日期字符串的年份差是否在阈值内"""
    try:
        ya = int(str(date_a)[:4])
        yb = int(str(date_b)[:4])
        return abs(ya - yb) <= threshold
    except (ValueError, TypeError):
        return False


def _l1_exact_match(entity_type: str, field: str, value: str) -> str | None:
    """L1 规则层：Nebula LOOKUP 精确匹配，返回 VID 或 None"""
    tag = _tag_name(entity_type)
    nql = f'LOOKUP ON {tag} WHERE {tag}.{field} == "{_esc(value)}" YIELD id(vertex) AS vid'
    rows = _nb_rows(nql)
    return rows[0]["vid"] if rows else None


def _l1_name_candidates(entity_type: str, name: str) -> list[dict]:
    """L1 规则层：按名称查找候选，返回 [{vid, name, ...}]

    过滤掉 VID prefix 不匹配的候选——防止跨标签脏数据污染
    （如 equip_1245 出现在 location 候选里）。
    """
    tag = _tag_name(entity_type)
    expected_prefix = _vid_prefix(entity_type)
    nql = f'LOOKUP ON {tag} WHERE {tag}.name == "{_esc(name)}" YIELD id(vertex) AS vid, {tag}.name AS name'
    rows = _nb_rows(nql)
    # VID prefix 过滤：equip_* 不应出现在 location 候选里
    filtered = [r for r in rows if r.get("vid", "").startswith(expected_prefix + "_")]
    if len(filtered) < len(rows):
        dropped = [r.get("vid") for r in rows if r not in filtered]
        log.warning(f"[L1] VID prefix filter dropped {len(rows) - len(filtered)} cross-tag candidates: {dropped}")
    return filtered


def _l1_rule_score(entity_type: str, new_entity: dict, candidate_vid: str) -> tuple:
    """L1 双向属性评分（使用经离线实验验证的 universal property overlap 算法）

    返回 (score, decision):
      score ∈ [-1.0, +1.0]
      decision: 'merge' | 'reject' | 'escalate'
      merge   → score ≥ 0.6 (跳过 L2)
      reject  → score ≤ -0.4 (直接新建)
      escalate → 中间区间 (→ L2 LLM)
    """
    tag = _tag_name(entity_type)
    # 取候选实体完整属性
    field_map = {
        "equipment": ["name", "equip_type", "category", "state", "parent_unit", "home_location", "aliases"],
        "person":    ["name", "occupation", "org_name", "nationality", "gender", "aliases"],
        "location":  ["name", "loc_type", "region", "coordinates", "aliases"],
        "activity":  ["name", "event_type", "start_date", "end_date", "location_name", "aliases"],
        "organization": ["name", "org_type", "region", "industry", "aliases"],
    }
    fields = field_map.get(entity_type, ["name"])
    field_str = ", ".join([f'{tag}.{f} AS {f}' for f in fields])
    rows = _nb_rows(f'FETCH PROP ON {tag} "{candidate_vid}" YIELD {field_str}')
    if not rows:
        return 0.0, 'escalate'

    cand = rows[0]
    # 提取候选实体名称
    cand_name = cand.get("name", "")
    # 使用 adaptive scorer：先查 learned model，无则用 fixed L1
    result = _l1_score_adaptive_fn(
        new_entity, cand, entity_type,
        name_a=new_entity.get("name", ""), name_b=cand_name,
    )
    score = result.score
    decision = result.decision
    return score, decision


def _disambig_entity(entity_type: str, name: str, designation: str = None,
                     extra: dict = None,
                     article_title: str = "", article_excerpt: str = "",
                     db_conn=None) -> str | None:
    """两层消歧，返回匹配的 VID 或 None（新建）

    L1 规则层：双向属性评分 → 三态判断
      merge (score ≥ 0.6)  → 直接用已有 VID
      reject (score ≤ -0.4) → 直接新建（返回 None）
      escalate             → L2 LLM 判断
    L2 LLM 层：带源文本片段的语义判断 → 不确定的写 pending 等人工
    """
    extra = extra or {}

    # L1: designation 精确匹配（equipment 特有）
    if designation:
        vid = _l1_exact_match(entity_type, "name", designation)
        if vid:
            return vid

    # L1: 名称候选 + 双向评分
    candidates = _l1_name_candidates(entity_type, name)
    if not candidates:
        return None  # 无同名实体，直接新建

    # 对每个候选双向评分（使用新 universal overlap scorer）
    scored = []
    for c in candidates:
        s, decision = _l1_rule_score(entity_type, extra, c["vid"])
        scored.append((c, s, decision))
    scored.sort(key=lambda x: x[1], reverse=True)

    best_cand, best_score, best_decision = scored[0]

    # L1 路由
    if best_decision == 'merge' or best_score >= _L1_CONFIRM_SAME:
        log.info(f"[Disambig L1] '{name}' → {best_cand['vid']} "
                 f"(score={best_score:.2f}, merge)")
        return best_cand["vid"]
    if best_decision == 'reject' or best_score <= _L1_CONFIRM_DIFF:
        log.info(f"[Disambig L1] '{name}' → new "
                 f"(score={best_score:.2f}, reject)")
        return None

    # 唯一候选 + 灰色区间：保守策略，仍送 L2
    # （旧代码此处会直接接受，但这违反了 precision priority 原则）

    # L1 不确定 → L2 LLM 判断
    log.info(f"[Disambig] L1 uncertain for '{name}' "
             f"(score={best_score:.2f}, {len(scored)} candidates) → L2")

    # 准备候选属性列表给 L2
    tag = _tag_name(entity_type)
    l2_candidates = []
    for c, s, d in scored:
        vid = c["vid"]
        props_rows = _nb_rows(f'FETCH PROP ON {tag} "{vid}" YIELD properties(vertex) AS props')
        cand_props = {}
        if props_rows and "props" in props_rows[0]:
            raw = props_rows[0]["props"]
            if isinstance(raw, str):
                try:
                    import json as _json
                    cand_props = _json.loads(raw)
                except Exception:
                    cand_props = {}
            elif isinstance(raw, dict):
                cand_props = raw
        # 取候选关联实体
        related = _nb_rows(
            f'GO FROM "{vid}" OVER employ, locate, social BIDIRECT '
            f'YIELD employ._dst AS employ_dst, locate._dst AS locate_dst, social._dst AS social_dst'
        )
        related_names = []
        if related:
            seen = set()
            for row in related:
                for k in ("employ_dst", "locate_dst", "social_dst"):
                    dst_vid = row.get(k, "")
                    if dst_vid and dst_vid not in seen:
                        seen.add(dst_vid)
                        dst_tag = "person" if k == "social_dst" else (
                            "organization" if k == "employ_dst" else "location")
                        dst_rows = _nb_rows(f'FETCH PROP ON {dst_tag} "{dst_vid}" YIELD {dst_tag}.name AS name')
                        if dst_rows:
                            related_names.append(f"{dst_tag}:{dst_rows[0].get('name', '?')}")
        if related_names:
            cand_props["_related"] = ", ".join(related_names[:5])

        l2_candidates.append({
            "vid": vid,
            "name": c.get("name", ""),
            "props": cand_props,
        })

    # 调用 L2 judge（带源文本片段提取 + 人工 few-shot）
    l2_result = _l2_judge_fn(
        entity_type=entity_type,
        name_new=name,
        props_new=extra,
        candidates=l2_candidates,
        article_title=article_title,
        article_excerpt=article_excerpt,
        entity_vid_new="",  # 抽取时模式，新实体尚未写入图
        db_conn=db_conn,  # 传入 DB 连接以获取人工 few-shot 样本
    )

    match_vid = l2_result.get('match_vid')
    is_uncertain = l2_result.get('is_uncertain', False)

    if match_vid:
        return match_vid

    # L2 不确定 — 记录候选对，后续写入 pending 等人工处理
    if is_uncertain:
        _l2_uncertain.append({
            "entity_type": entity_type,
            "name": name,
            "candidates": [{"vid": c["vid"], "name": c.get("name", "")} for c, _, _ in scored],
            "source": "l2_uncertain",
        })
    log.info(f"[Disambig] L2 result for '{name}': "
             f"{'uncertain' if is_uncertain else 'different'}, creating new")
    return None


def _l2_llm_disambiguate(entity_type: str, name: str, extra: dict,
                         candidates: list[dict],
                         article_title: str, article_excerpt: str) -> str | None:
    """L2: LLM 判断，返回匹配的 VID 或 None"""
    if not LLM_API_KEY or not candidates:
        return None

    tag = _tag_name(entity_type)

    # 构造候选文本
    candidates_text = ""
    valid_vids = set()
    for c in candidates:
        vid = c.get("vid", "")
        valid_vids.add(vid)
        # 取候选属性
        props = _nb_rows(f'FETCH PROP ON {tag} "{vid}" YIELD {tag}.name AS name')
        pname = props[0].get("name", "?") if props else "?"
        line = f"  [{vid}] {pname}"
        # 按类型取额外属性
        extra_fields = {
            "equipment": ["equip_type", "category", "state", "aliases"],
            "person": ["occupation", "org_name", "aliases"],
            "location": ["loc_type", "region", "aliases"],
            "activity": ["event_type", "start_date", "location_name", "aliases"],
            "organization": ["org_type", "region", "industry", "aliases"],
        }
        fields = extra_fields.get(entity_type, [])
        if fields:
            field_str = ", ".join([f'{tag}.{f} AS {f}' for f in fields])
            det = _nb_rows(f'FETCH PROP ON {tag} "{vid}" YIELD {field_str}')
            if det:
                attrs = []
                for f in fields:
                    v = det[0].get(f, "")
                    if v:
                        attrs.append(f"{f}={v}")
                if attrs:
                    line += f"\n    Attributes: {', '.join(attrs)}"
        # 取候选关联实体（employ→组织, locate→地点, social→人物）
        related = _nb_rows(
            f'GO FROM "{vid}" OVER employ, locate, social BIDIRECT '
            f'YIELD employ._dst AS employ_dst, locate._dst AS locate_dst, social._dst AS social_dst'
        )
        if related:
            related_names = []
            seen = set()
            for row in related:
                for k in ("employ_dst", "locate_dst", "social_dst"):
                    dst_vid = row.get(k, "")
                    if dst_vid and dst_vid not in seen:
                        seen.add(dst_vid)
                        # 取关联实体名
                        dst_tag = "person" if k == "social_dst" else ("organization" if k == "employ_dst" else "location")
                        dst_rows = _nb_rows(f'FETCH PROP ON {dst_tag} "{dst_vid}" YIELD {dst_tag}.name AS name')
                        if dst_rows:
                            related_names.append(f"{dst_tag}:{dst_rows[0].get('name', '?')}")
            if related_names:
                line += f"\n    Related entities: {', '.join(related_names[:5])}"  # 最多5个，防prompt太长
        candidates_text += line + "\n"

    # 取原文中包含实体名的句子作为上下文
    entity_context = ""
    if article_excerpt:
        sentences = article_excerpt.replace('\n', ' ').split('.')
        entity_sentences = [s.strip() for s in sentences if name.lower() in s.lower()]
        if entity_sentences:
            entity_context = entity_sentences[0][:200]

    prompt = f"""You are a military entity resolution expert. Determine if the NEW entity matches any EXISTING entity.

Entity type: {entity_type}

Source article: {article_title or 'N/A'}
Source context (sentence mentioning this entity): {entity_context or 'N/A'}

NEW entity:
  Name: {name}
  Properties: {extra or {}}

EXISTING candidates:
{candidates_text}

Resolution rules:
- Same hull number/designation = same equipment. Different = different.
- Name variants: Adm. = Admiral, Rob=Robert, Mike=Michael
- Location synonyms: Arabian Gulf = Persian Gulf, Formosa Strait = Taiwan Strait
- "Arabian Sea" ≠ "Persian Gulf"
- Same name + same org/role = very likely same person
- Same name + different org + different role = likely different person
- For events: same name + same region + overlapping time = same event. Different era/region = different.

Return ONLY: {{"match_vid": "<vid or null>", "reason": "<brief>"}}"""

    # 同步锁：确保 L2 消歧和主 extract LLM 调用不并发
    with _sync_llm_lock:
        try:
            resp = httpx.post(
                f"{LLM_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {LLM_API_KEY}",
                    "User-Agent": "OpenAI/Python 1.30.0",
                    "X-Title": "hermes-agent",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                    "thinking": {"type": "disabled"},
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            result = json.loads(content)
            match_vid = result.get("match_vid")
            reason = result.get("reason", "")
            if match_vid and match_vid in valid_vids:
                log.info(f"[Disambig L2] '{name}' → {match_vid} ({reason})")
                return match_vid
            elif match_vid:
                log.warning(f"[Disambig L2] Invalid vid '{match_vid}', ignoring")
            return None
        except Exception as e:
            log.warning(f"[Disambig L2] LLM failed: {e}")
            return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Nebula 写入
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _resolve_existing_vid(entity_type: str, name: str) -> str | None:
    """从 Nebula 按名称查找已有实体的 VID"""
    tag = _tag_name(entity_type)
    if not tag or not name:
        return None
    rows = _nb_rows(f'LOOKUP ON {tag} WHERE {tag}.name == "{_esc(name)}" YIELD id(vertex) AS vid')
    if rows:
        return rows[0].get("vid")
    # 试试别名
    rows = _nb_rows(f'LOOKUP ON {tag} YIELD id(vertex) AS vid, {tag}.aliases AS aliases')
    for r in rows:
        aliases_str = r.get("aliases", "")
        if aliases_str and name in aliases_str.split("||"):
            return r.get("vid")
    return None


def _insert_entity(entity_type: str, vid: str, attrs: dict):
    """INSERT VERTEX 到 Nebula"""
    tag = _tag_name(entity_type)
    # 显式补齐旧字段默认值（ALTER TAG ADD 后旧字段仍存在于 Schema）
    defaults = {
        "equipment": {"latest_reported_at": ""},
        "event": {"occurred_at": ""},  # 已迁移到 start_date/end_date，旧字段补空
    }
    for k, v in defaults.get(tag, {}).items():
        if k not in attrs:
            attrs[k] = v
    fields = ", ".join(attrs.keys())
    vals = ", ".join([f'"{_esc(v)}"' if isinstance(v, str) else str(v) for v in attrs.values()])
    nql = f'INSERT VERTEX {tag}({fields}) VALUES "{vid}":({vals})'
    r = _nb_query(nql)
    if not r.is_succeeded() and "Existed" not in r.error_msg():
        log.warning(f"[Nebula] INSERT {tag} failed: {r.error_msg()[:80]}")


def _update_aliases(tag: str, vid: str, new_aliases: list):
    """UPDATE aliases 字段"""
    if not new_aliases:
        return
    # 读现有 aliases
    rows = _nb_rows(f'FETCH PROP ON {tag} "{vid}" YIELD {tag}.aliases AS aliases')
    existing = rows[0].get("aliases", "") if rows else ""
    existing_list = [a for a in (existing.split("||") if existing else []) if a]
    changed = False
    for a in new_aliases:
        if a and a not in existing_list:
            existing_list.append(a)
            changed = True
    if changed:
        new_val = "||".join(existing_list)
        nql = f'UPDATE VERTEX ON {tag} "{vid}" SET aliases = "{_esc(new_val)}"'
        _nb_query(nql)



# relation 大类 → 独立边名映射
_RELATION_TO_EDGE = {
    "social": "social",
    "employ": "employ",
    "locate": "locate",
    "participate": "participate",
    "org_relation": "org_relation",
    "use": "use_edge",
    "use_edge": "use_edge",
    "spatial": "spatial",
    "trace": "trace",
}


def _insert_related(src_vid: str, dst_vid: str, relation: str, article_id: int, confidence: float = 0.9, sub_type: str = ""):
    """INSERT EDGE <独立边类型>（v1.6: 8种独立边替代旧 related 统一边）"""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    st = _esc(sub_type) if sub_type else ""
    edge_name = _RELATION_TO_EDGE.get(relation.lower().strip(), "social")
    nql = f'INSERT EDGE {edge_name}(sub_type, confidence, source, created_at) VALUES "{src_vid}"->"{dst_vid}":("{st or _esc(relation)}", {confidence}, "{article_id}", "{now}")'
    r = _nb_query(nql)
    if not r.is_succeeded() and "Existed" not in r.error_msg():
        log.warning(f"[Nebula] INSERT EDGE {edge_name} failed: {r.error_msg()[:80]}")


def _insert_trace_generate(record_vid: str, target_vid: str, entity_type: str, confidence: float = 0.9):
    """INSERT EDGE trace(sub_type='generate') — 溯源边"""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    nql = f'INSERT EDGE trace(sub_type, confidence, source, created_at) VALUES "{record_vid}"->"{target_vid}":("generate", {confidence}, "article", "{now}")'
    r = _nb_query(nql, space="usn_main")
    if not r.is_succeeded() and "Existed" not in r.error_msg():
        log.warning(f"[Nebula] INSERT EDGE trace.generate failed: {r.error_msg()[:80]}")


def _fetch_location_coords(loc_vid: str) -> tuple[float, float] | None:
    """从 Nebula location 节点取坐标"""
    r = _nb_query(f'FETCH PROP ON location "{loc_vid}" YIELD location.coordinates AS coords')
    if r.is_succeeded() and r.row_size() > 0:
        raw = r.row_values(0)[0]
        s = str(raw).strip('"').strip()
        if "," in s:
            parts = s.split(",", 1)
            try:
                return (float(parts[0].strip()), float(parts[1].strip()))
            except ValueError:
                pass
    return None


def _vid_to_numeric(vid: str) -> int | None:
    """equip_37 → 37"""
    if "_" in vid:
        try:
            return int(vid.split("_", 1)[1])
        except ValueError:
            pass
    return None


def _insert_trace_record(article_id: int, title: str, url: str = "", published_at: str = ""):
    """INSERT VERTEX datarecord（溯源定位节点）

    严格遵循探数 v1.2 datarecord 设计：只存溯源定位三要素 + summary。
    title/url/published_at 不入图，原始数据在 PG。
    """
    vid = f"record_article_{article_id}"
    now = datetime.utcnow().strftime("%Y-%m-%d")
    nql = f'INSERT VERTEX datarecord(source_db, source_table, source_pk, summary, imported_at) VALUES "{vid}":("usn_monitor", "articles", "{article_id}", "{_esc(title[:200])}", "{_esc(now)}")'
    r = _nb_query(nql, space="usn_main")
    if not r.is_succeeded() and "Existed" not in r.error_msg():
        log.warning(f"[Nebula] INSERT datarecord failed: {r.error_msg()[:80]}")
    return vid


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 自增 ID（简单方案：查现有最大 ID + 1）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _next_id(entity_type: str) -> int:
    prefix = _vid_prefix(entity_type)
    rows = _nb_rows(f'LOOKUP ON {_tag_name(entity_type)} YIELD id(vertex) AS vid')
    max_id = 0
    for r in rows:
        vid = r.get("vid", "")
        if vid.startswith(f"{prefix}_"):
            try:
                nid = int(vid.split("_", 1)[1])
                if nid > max_id:
                    max_id = nid
            except ValueError:
                pass
    return max_id + 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def extract_pending_articles(db: AsyncSession) -> int:
    """处理所有待抽取的文章 — 串行，全局锁保证不并发"""
    # 全局锁：防止多个 extract 任务并发
    if _extract_semaphore.locked():
        log.info("[Extractor] Another extraction is running, skipping")
        return 0

    async with _extract_semaphore:
        return await _do_extract(db)


async def _do_extract(db: AsyncSession) -> int:
    """实际抽取逻辑 — 每次取 5 篇，一篇一篇串行调 LLM"""
    rows = await db.execute(text("""
        SELECT id, title, content FROM articles
        WHERE processing_status = 'pending' AND content IS NOT NULL
        ORDER BY published_at DESC LIMIT 5
    """))
    articles = rows.mappings().all()

    if not articles:
        r = await db.execute(text("""
            UPDATE articles SET processing_status = 'failed'
            WHERE processing_status = 'pending' AND (content IS NULL OR content = '')
        """))
        await db.commit()
        log.info("[Extractor] No pending articles with content")
        return 0

    log.info(f"[Extractor] Processing {len(articles)} pending articles (serial)")
    processed = 0
    for i, article in enumerate(articles):
        try:
            log.info(f"[Extractor] [{i+1}/{len(articles)}] Extracting: {article['title'][:60]}...")
            await db.execute(text("""
                UPDATE articles SET processing_status = 'processing' WHERE id = :id
            """), {"id": article["id"]})
            await db.commit()

            result = await extract_with_llm(article["title"], article["content"])
            if result:
                counts = {k: len(v) for k, v in result.items() if isinstance(v, list)}
                log.info(f"[Extractor] Found: {counts}")
                save_result = save_to_nebula(article["id"], article["title"], result,
                                 article_excerpt=(article.get("content") or "")[:500],
                                 db_conn=db)
                uncertain_pairs = save_result.get("uncertain_pairs", []) if isinstance(save_result, dict) else save_result
                location_rels = save_result.get("location_rels", []) if isinstance(save_result, dict) else []
                # L2 不确定的候选对 → PG pending_entities
                if uncertain_pairs:
                    for pair in uncertain_pairs:
                        # 取包含实体名的原文句子作为 context
                        excerpt = (article.get("content") or "")[:500]
                        sentences = excerpt.replace('\n', ' ').split('.')
                        ctx_sentences = [s.strip() for s in sentences if pair["name_a"].lower() in s.lower()]
                        pair_context = ctx_sentences[0][:300] if ctx_sentences else excerpt[:200]

                        await db.execute(text("""
                            INSERT INTO pending_entities (entity_type, vid_a, vid_b, name_a, name_b, similarity, raw_data, context, source_article, status, llm_verdict)
                            VALUES (:etype, :vid_a, :vid_b, :name_a, :name_b, 0.5, CAST(:source AS jsonb), :ctx, :article_id, 'pending', 'l2_uncertain')
                        """), {
                            "etype": pair["entity_type"],
                            "vid_a": pair["vid_a"], "vid_b": pair["vid_b"],
                            "name_a": pair["name_a"], "name_b": pair["name_b"],
                            "source": json.dumps({"source": pair["source"]}),
                            "ctx": pair_context,
                            "article_id": article["id"],
                        })
                    log.info(f"[Extractor] {len(uncertain_pairs)} uncertain pairs written to pending")
                # 位置记录 → PG equipment_positions
                if location_rels:
                    from datetime import datetime as dt
                    pos_imported = 0
                    for lr in location_rels:
                        try:
                            await db.execute(text("""
                                INSERT INTO equipment_positions (equipment_vid, location, reported_at, source_article_id, source_type, notes)
                                VALUES (:vid, ST_SetSRID(ST_MakePoint(:lng, :lat), 4326), :reported, :aid, :stype, :notes)
                                ON CONFLICT DO NOTHING
                            """), {
                                "vid": lr["equipment_vid"],
                                "lng": lr["lng"],
                                "lat": lr["lat"],
                                "reported": dt.utcnow(),
                                "aid": article["id"],
                                "stype": "news_extract",
                                "notes": f"{lr['relation']} → {lr['location_name']}",
                            })
                            pos_imported += 1
                        except Exception as e:
                            log.warning(f"[Extractor] Position insert failed: {e}")
                    log.info(f"[Extractor] {pos_imported} position records written for article {article['id']}")
                await db.execute(text("""
                    UPDATE articles SET processing_status = 'done', entities_extracted = true WHERE id = :id
                """), {"id": article["id"]})
                await db.commit()
                processed += 1
                # coding plan RPM 较高，每篇之间等待 15s
                if i < len(articles) - 1:
                    log.info(f"[Extractor] Waiting 15s before next article...")
                    await asyncio.sleep(15)
            else:
                log.warning(f"[Extractor] LLM returned no result for article {article['id']}")
                await db.execute(text("""
                    UPDATE articles SET processing_status = 'pending', entities_extracted = false WHERE id = :id
                """), {"id": article["id"]})
                await db.commit()

        except Exception as e:
            log.error(f"[Extractor] Error on article {article['id']}: {e}", exc_info=True)
            try:
                await db.rollback()
            except Exception:
                pass
            try:
                await db.execute(text("""
                    UPDATE articles SET processing_status = 'failed' WHERE id = :id
                """), {"id": article["id"]})
                await db.commit()
            except Exception as e2:
                log.error(f"[Extractor] Failed to mark article {article['id']} as failed: {e2}")
                try:
                    await db.rollback()
                except Exception:
                    pass

    log.info(f"[Extractor] Done: {processed}/{len(articles)} articles processed")
    return processed


def _sync_llm_call(prompt: str):
    """同步 LLM 调用（带 response_format），受 _sync_llm_lock 保护"""
    with _sync_llm_lock:
        return httpx.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "User-Agent": "OpenAI/Python 1.30.0",
                "X-Title": "hermes-agent",
            },
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "thinking": {"type": "disabled"},
            },
            timeout=120.0,
        )


def _sync_llm_call_raw(prompt: str):
    """同步 LLM 调用（无 response_format），受 _sync_llm_lock 保护"""
    with _sync_llm_lock:
        return httpx.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "User-Agent": "OpenAI/Python 1.30.0",
                "X-Title": "hermes-agent",
            },
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "thinking": {"type": "disabled"},
            },
            timeout=120.0,
        )


async def extract_with_llm(title: str, content: str) -> dict | None:
    """调用 LLM 抽取实体 — 全局串行锁，一次只有一个请求在飞"""
    if not LLM_API_KEY:
        log.warning("[LLM] No API key, skipping extraction")
        return None

    prompt = EXTRACT_PROMPT.format(title=title, content=content[:4000])

    return await _call_llm(prompt)


async def _call_llm(prompt: str) -> dict | None:
    """实际 LLM 调用 — 通过 run_in_executor 跑同步函数，共享 _sync_llm_lock 保证全局串行"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, _sync_llm_call, prompt)
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                log.warning(f"[LLM] 429 rate limited, waiting {wait}s...")
                await asyncio.sleep(wait)
                continue
            if resp.status_code == 400:
                log.warning(f"[LLM] 400 on attempt {attempt+1}, retrying without response_format")
                resp = await loop.run_in_executor(None, _sync_llm_call_raw, prompt)
            resp.raise_for_status()
            data = resp.json()
            text_content = data["choices"][0]["message"]["content"].strip()
            if text_content.startswith("```"):
                lines = text_content.split("\n")
                text_content = "\n".join(lines[1:-1]) if len(lines) > 2 else text_content
            return json.loads(text_content)
        except json.JSONDecodeError as e:
            log.warning(f"[LLM] JSON parse failed on attempt {attempt+1}: {e}")
        except httpx.TimeoutException:
            log.warning(f"[LLM] Timeout on attempt {attempt+1}")
        except Exception as e:
            log.warning(f"[LLM] Attempt {attempt+1} failed: {e}")
        if attempt < max_retries - 1:
            await asyncio.sleep(5 * (attempt + 1))
    log.error(f"[LLM] All {max_retries} attempts failed")
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 保存到 Nebula（同步，在 extractor 内调用）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_to_nebula(article_id: int, article_title: str, entities: dict,
                 article_excerpt: str = "", db_conn=None) -> list[dict]:
    """将抽取结果写入 Nebula Graph（主图 + 溯源图），返回 L2 不确定的候选对列表"""
    global _l2_uncertain
    _l2_uncertain = []  # 重置累积器

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # 溯源图：创建 record 节点
    record_vid = _insert_trace_record(article_id, article_title)

    # 实体 ID 映射（name → vid），给后面建关系用
    entity_map = {}  # (type, name) → vid

    # ── 1. equipment ──
    for eq in entities.get("equipment", []):
        name = eq.get("name", "")
        desig = eq.get("designation", "")
        if not name and not desig:
            continue

        vid = _disambig_entity("equipment", name, desig, eq,
                                 article_title=article_title, article_excerpt=article_excerpt, db_conn=db_conn)
        if vid:
            _update_aliases("equipment", vid, eq.get("aliases", []))
        else:
            new_id = _next_id("equipment")
            vid = f"equip_{new_id}"
            _insert_entity("equipment", vid, {
                "name": name,
                "aliases": "||".join(eq.get("aliases", [])),
                "equip_type": eq.get("equipment_type", "other"),
                "category": eq.get("category", ""),
                "state": eq.get("status", "active"),
                "parent_unit": "",
                "home_location": "",
                "description": "",
                "confidence": 1.0,
                "created_at": now,
                "updated_at": now,
            })
        entity_map[("equipment", name)] = vid
        entity_map[("equipment", desig)] = vid if desig else None

        # 溯源图 generate 边
        _insert_trace_generate(record_vid, vid, "equipment")

    # ── 2. persons ──
    for p in entities.get("persons", []):
        name = p.get("name", "")
        if not name:
            continue

        vid = _disambig_entity("person", name, extra={
            "rank": p.get("rank", ""),
            "position": p.get("position", ""),
            "service_branch": p.get("service_branch", ""),
        }, article_title=article_title, article_excerpt=article_excerpt, db_conn=db_conn)
        if vid:
            _update_aliases("person", vid, p.get("aliases", []))
        else:
            new_id = _next_id("person")
            vid = f"person_{new_id}"
            _insert_entity("person", vid, {
                "name": name,
                "aliases": "||".join(p.get("aliases", [])),
                "gender": "",
                "nationality": "",
                "occupation": p.get("position", ""),
                "org_name": p.get("service_branch", ""),
                "description": "",
                "labels": "",
                "confidence": 1.0,
                "created_at": now,
                "updated_at": now,
            })
        entity_map[("person", name)] = vid

        _insert_trace_generate(record_vid, vid, "person")

    # ── 3. locations ──
    for loc in entities.get("locations", []):
        name = loc.get("name", "")
        if not name:
            continue

        lat = loc.get("lat")
        lng = loc.get("lng")
        coords = f"{lat},{lng}" if lat and lng else ""

        vid = _disambig_entity("location", name, extra={
            "region": loc.get("region", ""),
        }, article_title=article_title, article_excerpt=article_excerpt, db_conn=db_conn)
        if not vid:
            new_id = _next_id("location")
            vid = f"loc_{new_id}"
            _insert_entity("location", vid, {
                "name": name,
                "aliases": "",
                "loc_type": loc.get("location_type", "other"),
                "region": loc.get("region", ""),
                "coordinates": coords,
                "description": "",
                "confidence": 1.0,
                "created_at": now,
                "updated_at": now,
            })
        entity_map[("location", name)] = vid

        _insert_trace_generate(record_vid, vid, "location")

    # ── 3.5. organizations ──
    for org in entities.get("organizations", []):
        name = org.get("name", "")
        if not name:
            continue

        vid = _disambig_entity("organization", name, extra={
            "org_type": org.get("org_type", ""),
            "region": org.get("country", ""),
        }, article_title=article_title, article_excerpt=article_excerpt, db_conn=db_conn)
        if vid:
            _update_aliases("organization", vid, org.get("aliases", []))
        else:
            new_id = _next_id("organization")
            vid = f"org_{new_id}"
            _insert_entity("organization", vid, {
                "name": name,
                "aliases": "||".join(org.get("aliases", [])),
                "org_type": org.get("org_type", "other"),
                "industry": "",
                "region": org.get("country", ""),
                "description": org.get("description", ""),
                "confidence": 1.0,
                "created_at": now,
                "updated_at": now,
            })
        entity_map[("organization", name)] = vid

        _insert_trace_generate(record_vid, vid, "organization")

    # ── 4. activities → event ──
    act_type_map = {
        "operation": "military", "deployment": "military", "exercise": "military",
        "port_visit": "military", "transit": "military", "incident": "military",
        "surveillance": "military", "patrol": "military", "blockade": "military",
        "strike": "military", "escort": "military", "budgeting": "social",
        "other": "social",
    }

    for act in entities.get("activities", []):
        name = act.get("name", "")
        if not name:
            continue

        event_type = act_type_map.get(act.get("activity_type", ""), "social")
        start_date = str(act.get("start_date", "") or "")

        vid = _disambig_entity("activity", name, extra={
            "region": act.get("region", ""),
            "start_date": start_date,
        }, article_title=article_title, article_excerpt=article_excerpt, db_conn=db_conn)
        if not vid:
            new_id = _next_id("activity")
            vid = f"event_{new_id}"
            _insert_entity("activity", vid, {
                "name": name,
                "aliases": "",
                "event_type": event_type,
                "start_date": start_date,
                "end_date": "",
                "location_name": act.get("region", ""),
                "description": act.get("description", ""),
                "confidence": 1.0,
                "created_at": now,
                "updated_at": now,
            })
        entity_map[("activity", name)] = vid

        _insert_trace_generate(record_vid, vid, "activity")

    # ── 5. relations → related 边 ──
    type_map = {
        "equipment": "equipment", "ship": "equipment", "aircraft": "equipment",
        "person": "person", "personnel": "person",
        "location": "location", "base": "location",
        "activity": "activity", "operation": "activity",
        "organization": "organization", "org": "organization",
        "military_unit": "organization", "company": "organization",
        "agency": "organization", "government_agency": "organization",
    }

    for rel in entities.get("relations", []):
        subj_type = type_map.get(rel.get("subject_type", ""), rel.get("subject_type", ""))
        subj_name = rel.get("subject_name", "")
        rel_type = rel.get("relation", "")
        sub_type = rel.get("sub_type", "other")
        obj_type = type_map.get(rel.get("object_type", ""), rel.get("object_type", ""))
        obj_name = rel.get("object_name", "")

        if not subj_name or not obj_name or not rel_type:
            continue

        # 过滤低质量关系
        if rel_type in ("mentioned", "related_to", "involves", "associated_with", "trace"):
            continue
        if subj_type == obj_type and subj_name == obj_name:
            continue

        src_vid = entity_map.get((subj_type, subj_name))
        dst_vid = entity_map.get((obj_type, obj_name))

        # 如果 entity_map 里没有，尝试从 Nebula 查已有实体
        if not src_vid and subj_type:
            src_vid = _resolve_existing_vid(subj_type, subj_name)
            if src_vid:
                entity_map[(subj_type, subj_name)] = src_vid
        if not dst_vid and obj_type:
            dst_vid = _resolve_existing_vid(obj_type, obj_name)
            if dst_vid:
                entity_map[(obj_type, obj_name)] = dst_vid

        if src_vid and dst_vid:
            _insert_related(src_vid, dst_vid, rel_type, article_id, sub_type=sub_type)

    log.info(f"[Nebula] Saved article {article_id}: "
             f"{len(entities.get('equipment', []))} equip, "
             f"{len(entities.get('persons', []))} person, "
             f"{len(entities.get('locations', []))} loc, "
             f"{len(entities.get('activities', []))} event, "
             f"{len(entities.get('relations', []))} rel")

    # 收集 equipment→location 位置关系，供调用方写 equipment_positions
    location_rels = []
    # 新 prompt 输出 locate 大类，兼容旧格式（LLM 可能仍输出自由 relation）
    _locate_sub_types = {"located_at", "visited", "other"}
    _legacy_location_types = {
        "located_at", "deployed_to", "deployed_from", "operated_in",
        "homeported_at", "based_at", "visited", "transited", "ported_at",
    }

    for rel in entities.get("relations", []):
        subj_type = type_map.get(rel.get("subject_type", ""), rel.get("subject_type", ""))
        subj_name = rel.get("subject_name", "")
        rel_type = rel.get("relation", "")
        sub_type = rel.get("sub_type", "")
        obj_type = type_map.get(rel.get("object_type", ""), rel.get("object_type", ""))
        obj_name = rel.get("object_name", "")

        if not subj_name or not obj_name or not rel_type:
            continue
        if rel_type in ("mentioned", "related_to", "involves", "associated_with", "trace"):
            continue
        if subj_type == obj_type and subj_name == obj_name:
            continue

        # 新格式：relation=="locate"，旧格式：relation 是自由值
        is_locate = (rel_type == "locate") or (rel_type in _legacy_location_types)
        if subj_type == "equipment" and obj_type == "location" and is_locate:
            src_vid = entity_map.get((subj_type, subj_name))
            dst_vid = entity_map.get((obj_type, obj_name))
            if src_vid and dst_vid:
                coords = _fetch_location_coords(dst_vid)
                eq_id = _vid_to_numeric(src_vid)
                if coords and eq_id is not None:
                    location_rels.append({
                        "equipment_vid": src_vid,
                        "lat": coords[0],
                        "lng": coords[1],
                        "location_name": obj_name,
                        "relation": rel_type if rel_type != "locate" else sub_type,
                    })
    if location_rels:
        log.info(f"[Nebula] Collected {len(location_rels)} location relations for positions")

    # 配对 L2 不确定的实体：新 vid (from entity_map) × 候选 vid
    uncertain_pairs = []
    for entry in _l2_uncertain:
        new_vid = entity_map.get((entry["entity_type"], entry["name"]))
        if not new_vid:
            continue
        for cand in entry["candidates"]:
            uncertain_pairs.append({
                "entity_type": entry["entity_type"],
                "vid_a": new_vid,
                "vid_b": cand["vid"],
                "name_a": entry["name"],
                "name_b": cand.get("name", ""),
                "source": "extract_l2",
            })
    if uncertain_pairs:
        log.info(f"[Nebula] {len(uncertain_pairs)} uncertain pairs from article {article_id}")

    return {"uncertain_pairs": uncertain_pairs, "location_rels": location_rels}
