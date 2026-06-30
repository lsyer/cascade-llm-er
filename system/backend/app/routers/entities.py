"""实体查询 API — 图优先架构，全部查 Nebula Graph"""
from fastapi import APIRouter, Query
from typing import Optional

from app.nebula_service import get_nebula, vid_to_type, vid_to_id


def parse_aliases(val):
    """解析 Nebula 存储的 aliases 格式 '"{a,b,c}"' → ['a','b','c']"""
    if isinstance(val, list):
        return val
    if not val or not isinstance(val, str):
        return []
    s = val.strip()
    # Nebula set format: "{a, b, c}" or "{a}"
    if s.startswith('{') and s.endswith('}'):
        inner = s[1:-1]
        if not inner:
            return []
        # Split by comma, strip quotes
        items = []
        for item in inner.split(','):
            item = item.strip().strip('"').strip("'")
            if item:
                items.append(item)
        return items
    # Try JSON parse
    try:
        import json
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return parsed
        return [str(parsed)]
    except (json.JSONDecodeError, ValueError):
        return [s] if s else []

router = APIRouter()


# ==================== 器装 (Equipment) ====================

@router.get("/equipment")
async def list_equipment(
    equipment_type: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(100, le=500),
    offset: int = 0,
):
    """获取器装列表 — 从 Nebula equipment Tag 查询"""
    nb = get_nebula()
    conditions = []
    if equipment_type:
        conditions.append(f'equip_type == "{equipment_type}"')
    if category:
        conditions.append(f'category == "{category}"')
    if status:
        conditions.append(f'state == "{status}"')

    if conditions:
        where = " AND ".join(conditions)
        nql = f'LOOKUP ON equipment WHERE {where} YIELD id(vertex) AS vid, equipment.name AS name, equipment.equip_type AS equip_type, equipment.category AS category, equipment.state AS state, equipment.home_location AS home_location, equipment.latest_reported_at AS latest_reported_at, equipment.aliases AS aliases, equipment.description AS description, equipment.confidence AS confidence, equipment.created_at AS created_at, equipment.updated_at AS updated_at'
    else:
        nql = 'LOOKUP ON equipment YIELD id(vertex) AS vid, equipment.name AS name, equipment.equip_type AS equip_type, equipment.category AS category, equipment.state AS state, equipment.home_location AS home_location, equipment.latest_reported_at AS latest_reported_at, equipment.aliases AS aliases, equipment.description AS description, equipment.confidence AS confidence, equipment.created_at AS created_at, equipment.updated_at AS updated_at'

    rows = nb.query_rows(nql)

    # 搜索过滤（Nebula LOOKUP 不支持 ILIKE，在应用层过滤）
    if search:
        q = search.lower()
        filtered = []
        for r in rows:
            name = (r.get("name") or "").lower()
            aliases = (r.get("aliases") or "").lower()
            if q in name or q in aliases:
                filtered.append(r)
        rows = filtered

    # 分页
    total = len(rows)
    rows = rows[offset:offset + limit]

    # 解析 VID 提取原始 ID
    results = []
    for r in rows:
        r["id"] = vid_to_id(r.get("vid", ""))
        r["equipment_type"] = r.pop("equip_type", "")
        r["status"] = r.pop("state", "")
        results.append(r)

    return results


@router.get("/equipment/{eq_id}")
async def get_equipment(eq_id: int):
    """获取器装详情"""
    nb = get_nebula()
    vid = f"equip_{eq_id}"

    # 主数据
    row = nb.query_one(f'FETCH PROP ON equipment "{vid}" YIELD equipment.name AS name, equipment.equip_type AS equip_type, equipment.category AS category, equipment.state AS state, equipment.home_location AS home_location, equipment.latest_reported_at AS latest_reported_at, equipment.parent_unit AS parent_unit, equipment.aliases AS aliases, equipment.description AS description, equipment.confidence AS confidence, equipment.created_at AS created_at, equipment.updated_at AS updated_at')
    if not row:
        return {"error": "not found"}

    row["id"] = eq_id
    row["equipment_type"] = row.pop("equip_type", "")
    row["status"] = row.pop("state", "")
    row["aliases"] = parse_aliases(row.get("aliases"))

    # 解析坐标
    coords = row.get("home_location", "")
    if coords and "," in str(coords):
        parts = str(coords).split(",")
        try:
            row["home_lat"] = float(parts[0])
            row["home_lng"] = float(parts[1])
        except ValueError:
            pass

    # 关系（GO FROM，双向遍历8种独立边）
    _EDGE_TYPES = ["social", "employ", "locate", "participate", "org_relation", "use_edge", "spatial", "trace"]
    out_edges = []
    in_edges = []
    for et in _EDGE_TYPES:
        out_edges += nb.query_rows(
            f'GO FROM "{vid}" OVER {et} YIELD {et}._dst AS dst, {et}.sub_type AS relation, {et}.confidence AS confidence'
        )
        in_edges += nb.query_rows(
            f'GO FROM "{vid}" OVER {et} REVERSELY YIELD {et}._src AS src, {et}.sub_type AS relation, {et}.confidence AS confidence'
        )

    relations = []
    for e in out_edges:
        relations.append({
            "direction": "outgoing",
            "relation": e.get("relation", ""),
            "related_type": vid_to_type(e.get("dst", "")),
            "related_id": vid_to_id(e.get("dst", "")),
            "related_name": _fetch_name(nb, e.get("dst", "")),
        })
    for e in in_edges:
        relations.append({
            "direction": "incoming",
            "relation": e.get("relation", ""),
            "related_type": vid_to_type(e.get("src", "")),
            "related_id": vid_to_id(e.get("src", "")),
            "related_name": _fetch_name(nb, e.get("src", "")),
        })

    # 相关文章（通过 related 边反查）
    articles = await _fetch_source_articles(nb, vid)

    # 位置历史：从 PG equipment_positions 取
    positions = []
    try:
        from app.db import async_session
        from sqlalchemy import text as sql_text
        async with async_session() as session:
            pos_rows = await session.execute(sql_text("""
                SELECT id, ST_Y(location) AS lat, ST_X(location) AS lng,
                       reported_at, source_type, notes
                FROM equipment_positions
                WHERE equipment_vid = :vid
                ORDER BY reported_at DESC LIMIT 50
            """), {"vid": f"equip_{eq_id}"})
            for pr in pos_rows.mappings().all():
                pos = dict(pr)
                if pos.get("reported_at"):
                    pos["reported_at"] = pos["reported_at"].isoformat()
                positions.append(pos)
    except Exception:
        pass

    return {
        "equipment": row,
        "source": "news_extract" if articles else "seed",
        "articles": articles,
        "relations": relations,
        "positions": positions,
    }


# ==================== 人员 (Persons) ====================

@router.get("/persons")
async def list_persons(
    search: Optional[str] = None,
    limit: int = Query(100, le=500),
):
    nb = get_nebula()
    rows = nb.query_rows(
        'LOOKUP ON person YIELD id(vertex) AS vid, person.name AS name, person.occupation AS occupation, person.org_name AS org_name, person.aliases AS aliases, person.confidence AS confidence, person.created_at AS created_at, person.updated_at AS updated_at'
    )

    if search:
        q = search.lower()
        rows = [r for r in rows if q in (r.get("name") or "").lower()
                or q in (r.get("occupation") or "").lower()]

    rows = rows[:limit]
    for r in rows:
        r["id"] = vid_to_id(r.get("vid", ""))
        r["position"] = r.get("occupation", "")
        r["branch"] = r.get("org_name", "")

    return rows


@router.get("/persons/{person_id}")
async def get_person(person_id: int):
    nb = get_nebula()
    vid = f"person_{person_id}"

    row = nb.query_one(f'FETCH PROP ON person "{vid}" YIELD person.name AS name, person.occupation AS occupation, person.org_name AS org_name, person.aliases AS aliases, person.description AS description, person.confidence AS confidence, person.created_at AS created_at, person.updated_at AS updated_at')
    if not row:
        return {"error": "not found"}

    row["id"] = person_id
    row["aliases"] = parse_aliases(row.get("aliases"))
    row["position"] = row.get("occupation", "")
    row["branch"] = row.get("org_name", "")

    relations = _fetch_relations(nb, vid)
    articles = await _fetch_source_articles(nb, vid)

    return {
        "person": row,
        "career": [],
        "articles": articles,
        "relations": relations,
    }


# ==================== 地理位置 (Locations) ====================

@router.get("/locations")
async def list_locations(
    location_type: Optional[str] = None,
    region: Optional[str] = None,
    search: Optional[str] = None,
):
    nb = get_nebula()
    conditions = []
    if location_type:
        conditions.append(f'loc_type == "{location_type}"')
    if region:
        conditions.append(f'region == "{region}"')

    if conditions:
        where = " AND ".join(conditions)
        nql = f'LOOKUP ON location WHERE {where} YIELD id(vertex) AS vid, location.name AS name, location.loc_type AS loc_type, location.region AS region, location.coordinates AS coordinates, location.aliases AS aliases'
    else:
        nql = 'LOOKUP ON location YIELD id(vertex) AS vid, location.name AS name, location.loc_type AS loc_type, location.region AS region, location.coordinates AS coordinates, location.aliases AS aliases'

    rows = nb.query_rows(nql)

    if search:
        q = search.lower()
        rows = [r for r in rows if q in (r.get("name") or "").lower()
                or q in (r.get("aliases") or "").lower()]

    for r in rows:
        r["id"] = vid_to_id(r.get("vid", ""))
        r["location_type"] = r.pop("loc_type", "")
        coords = r.get("coordinates", "")
        if coords and "," in str(coords):
            parts = str(coords).split(",")
            try:
                r["lat"] = float(parts[0])
                r["lng"] = float(parts[1])
            except ValueError:
                pass

    return rows


@router.get("/locations/{loc_id}")
async def get_location(loc_id: int):
    nb = get_nebula()
    vid = f"loc_{loc_id}"

    row = nb.query_one(f'FETCH PROP ON location "{vid}" YIELD location.name AS name, location.loc_type AS loc_type, location.region AS region, location.coordinates AS coordinates, location.aliases AS aliases, location.description AS description, location.confidence AS confidence, location.created_at AS created_at, location.updated_at AS updated_at')
    if not row:
        return {"error": "not found"}

    row["id"] = loc_id
    row["aliases"] = parse_aliases(row.get("aliases"))
    row["location_type"] = row.pop("loc_type", "")
    coords = row.get("coordinates", "")
    if coords and "," in str(coords):
        parts = str(coords).split(",")
        try:
            row["lat"] = float(parts[0])
            row["lng"] = float(parts[1])
        except ValueError:
            pass

    relations = _fetch_relations(nb, vid)
    articles = await _fetch_source_articles(nb, vid)

    return {
        "location": row,
        "relations": relations,
        "articles": articles,
    }


# ==================== 活动/事件 (Events) ====================

@router.get("/activities")
async def list_activities(
    activity_type: Optional[str] = None,
    limit: int = Query(50, le=200),
):
    nb = get_nebula()

    # activity_type → event_type 映射
    event_type = activity_type
    if activity_type:
        type_map = {
            "offensive": "military", "defense": "military", "patrol": "military",
            "exercise": "military", "deployment": "military", "port_visit": "military",
            "operation": "military", "incident": "military", "other": "social",
        }
        event_type = type_map.get(activity_type, activity_type)

    if event_type:
        nql = f'LOOKUP ON event WHERE event.event_type == "{event_type}" YIELD id(vertex) AS vid, event.name AS name, event.event_type AS event_type, event.start_date AS start_date, event.end_date AS end_date, event.location_name AS location_name, event.description AS description'
    else:
        nql = 'LOOKUP ON event YIELD id(vertex) AS vid, event.name AS name, event.event_type AS event_type, event.start_date AS start_date, event.end_date AS end_date, event.location_name AS location_name, event.description AS description'

    rows = nb.query_rows(nql)[:limit]
    for r in rows:
        r["id"] = vid_to_id(r.get("vid", ""))
        r["activity_type"] = r.pop("event_type", "")
        r["region"] = r.pop("location_name", "")

    return rows


@router.get("/activities/{act_id}")
async def get_activity(act_id: int):
    nb = get_nebula()
    vid = f"event_{act_id}"

    row = nb.query_one(f'FETCH PROP ON event "{vid}" YIELD event.name AS name, event.event_type AS event_type, event.start_date AS start_date, event.end_date AS end_date, event.location_name AS location_name, event.description AS description, event.aliases AS aliases, event.confidence AS confidence, event.created_at AS created_at, event.updated_at AS updated_at')
    if not row:
        return {"error": "not found"}

    row["aliases"] = parse_aliases(row.get("aliases"))
    row["id"] = act_id
    row["activity_type"] = row.pop("event_type", "")
    row["region"] = row.pop("location_name", "")

    relations = _fetch_relations(nb, vid)
    articles = await _fetch_source_articles(nb, vid)

    return {
        "activity": row,
        "relations": relations,
        "articles": articles,
    }


# ==================== 文章 (Articles) ====================

@router.get("/articles")
async def list_articles(
    status: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
):
    """文章列表 — 从 PG 查询，含关联实体数"""
    from app.db import async_session
    from sqlalchemy import text as sql_text

    conditions = []
    params = {"lim": limit, "off": offset}

    if status:
        conditions.append("a.processing_status = :status")
        params["status"] = status
    if search:
        conditions.append("(a.title ILIKE :q OR a.content ILIKE :q)")
        params["q"] = f"%{search}%"

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    async with async_session() as db:
        # Count
        count_r = await db.execute(sql_text(f"SELECT COUNT(*) FROM articles a {where}"), params)
        total = count_r.scalar()

        # List
        r = await db.execute(sql_text(f"""
            SELECT a.id, a.title, a.url, a.published_at, a.processing_status,
                   a.entities_extracted, LENGTH(a.content) AS content_len
            FROM articles a {where}
            ORDER BY a.processing_status != 'failed' DESC, a.published_at DESC NULLS LAST, a.id DESC
            LIMIT :lim OFFSET :off
        """), params)
        rows = r.fetchall()

    articles = []
    for row in rows:
        articles.append({
            "id": row[0],
            "title": row[1],
            "url": row[2],
            "published_at": row[3].isoformat() if row[3] else None,
            "status": row[4],
            "entities_extracted": row[5],
            "content_len": row[6] or 0,
        })

    # Batch get entity counts from trace graph
    nb = get_nebula()
    for a in articles:
        rec_vid = f"record_article_{a['id']}"
        try:
            gen_rows = nb.query_rows(
                f'GO FROM "{rec_vid}" OVER trace YIELD trace._dst AS dst',
                space="usn_main"
            )
            a["entity_count"] = len(gen_rows)
        except Exception:
            a["entity_count"] = 0

    return {"total": total, "items": articles}


@router.get("/articles/{article_id}")
async def get_article(article_id: int):
    """文章详情 — PG 内容 + 溯源图关联实体"""
    from app.db import async_session
    from sqlalchemy import text as sql_text

    # PG 查文章内容
    async with async_session() as db:
        r = await db.execute(sql_text("""
            SELECT id, title, url, content, published_at, processing_status,
                   entities_extracted, raw_metadata
            FROM articles WHERE id = :id
        """), {"id": article_id})
        row = r.fetchone()

    if not row:
        return {"error": "not found"}

    article = {
        "id": row[0],
        "title": row[1],
        "url": row[2],
        "content": row[3] or "",
        "published_at": row[4].isoformat() if row[4] else None,
        "status": row[5],
        "entities_extracted": row[6],
    }

    # 溯源图查关联实体
    nb = get_nebula()
    entities = []
    rec_vid = f"record_article_{article_id}"

    gen_rows = nb.query_rows(
        f'GO FROM "{rec_vid}" OVER trace YIELD trace._dst AS target_vid, trace.sub_type AS target_type, trace.confidence AS confidence',
        space="usn_main"
    )

    for g in gen_rows:
        target = g.get("target_vid", "")
        entity_name = _fetch_name(nb, target)
        entities.append({
            "entity_type": vid_to_type(target),
            "entity_id": vid_to_id(target),
            "entity_name": entity_name,
            "confidence": g.get("confidence", 0.9),
        })

    return {
        "article": article,
        "entities": entities,
    }


# ==================== 搜索 ====================

@router.get("/search")
async def search_entities(q: str = Query(..., min_length=2)):
    """全局搜索 — 跨所有 Tag 类型"""
    nb = get_nebula()
    results = []
    qlower = q.lower()

    # Equipment
    rows = nb.query_rows('LOOKUP ON equipment YIELD id(vertex) AS vid, equipment.name AS name, equipment.category AS category')
    for r in rows:
        if qlower in (r.get("name") or "").lower():
            results.append({
                "type": "equipment",
                "id": vid_to_id(r.get("vid", "")),
                "name": r.get("name", ""),
                "subtitle": "",
                "category": r.get("category", ""),
            })

    # Person
    rows = nb.query_rows('LOOKUP ON person YIELD id(vertex) AS vid, person.name AS name, person.occupation AS occupation')
    for r in rows:
        if qlower in (r.get("name") or "").lower():
            results.append({
                "type": "person",
                "id": vid_to_id(r.get("vid", "")),
                "name": r.get("name", ""),
                "subtitle": r.get("occupation", ""),
                "category": "",
            })

    # Location
    rows = nb.query_rows('LOOKUP ON location YIELD id(vertex) AS vid, location.name AS name, location.loc_type AS loc_type')
    for r in rows:
        if qlower in (r.get("name") or "").lower():
            results.append({
                "type": "location",
                "id": vid_to_id(r.get("vid", "")),
                "name": r.get("name", ""),
                "subtitle": "",
                "category": r.get("loc_type", ""),
            })

    # Event
    rows = nb.query_rows('LOOKUP ON event YIELD id(vertex) AS vid, event.name AS name, event.event_type AS event_type')
    for r in rows:
        if qlower in (r.get("name") or "").lower():
            results.append({
                "type": "activity",
                "id": vid_to_id(r.get("vid", "")),
                "name": r.get("name", ""),
                "subtitle": "",
                "category": r.get("event_type", ""),
            })

    return results[:30]


# ==================== 统计 ====================

@router.get("/stats")
async def entity_stats():
    """实体统计 — 从 Nebula STATS"""
    nb = get_nebula()
    nb.query("SUBMIT JOB STATS")
    import time; time.sleep(3)
    r = nb.query("SHOW STATS")
    if not r.is_succeeded():
        return {"error": r.error_msg()}

    stats = {}
    for i in range(r.row_size()):
        vals = r.row_values(i)
        try:
            t = vals[0].as_string()
            name = vals[1].as_string()
            count = vals[2].as_string()
        except Exception:
            continue
        if t == "Tag":
            stats[name] = int(count)
        elif t == "Edge":
            stats[f"edge_{name}"] = int(count)
        elif t == "Space":
            stats["total_vertices"] = int(count)

    # 按装备类型细分
    equip_types = nb.query_rows(
        'LOOKUP ON equipment YIELD equipment.equip_type AS t'
    )
    type_counts = {}
    for r in equip_types:
        t = r.get("t", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    stats["equipment_by_type"] = [{"equipment_type": k, "count": v} for k, v in sorted(type_counts.items(), key=lambda x: -x[1])]

    return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 内部辅助函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fetch_name(nb, vid: str) -> str:
    """根据 VID 查实体名称"""
    etype = vid_to_type(vid)
    tag_map = {
        "equipment": ("equipment", "name"),
        "person": ("person", "name"),
        "location": ("location", "name"),
        "event": ("event", "name"),
        "organization": ("organization", "name"),
    }
    info = tag_map.get(etype)
    if not info:
        return vid
    tag, field = info
    r = nb.query_one(f'FETCH PROP ON {tag} "{vid}" YIELD {tag}.{field} AS name')
    return r.get("name", vid) if r else vid


def _fetch_relations(nb, vid: str) -> list[dict]:
    """获取实体的所有关系"""
    relations = []

    # 正向（遍历8种独立边）
    out = []
    inp = []
    _EDGE_TYPES = ["social", "employ", "locate", "participate", "org_relation", "use_edge", "spatial", "trace"]
    for et in _EDGE_TYPES:
        out += nb.query_rows(
            f'GO FROM "{vid}" OVER {et} YIELD {et}._dst AS dst, {et}.sub_type AS relation, {et}.confidence AS confidence, {et}.source AS source'
        )
        inp += nb.query_rows(
            f'GO FROM "{vid}" OVER {et} REVERSELY YIELD {et}._src AS src, {et}.sub_type AS relation, {et}.confidence AS confidence'
        )
    for e in out:
        relations.append({
            "direction": "outgoing",
            "relation": e.get("relation", ""),
            "related_type": vid_to_type(e.get("dst", "")),
            "related_id": vid_to_id(e.get("dst", "")),
            "related_name": _fetch_name(nb, e.get("dst", "")),
        })

    for e in inp:
        relations.append({
            "direction": "incoming",
            "relation": e.get("relation", ""),
            "related_type": vid_to_type(e.get("src", "")),
            "related_id": vid_to_id(e.get("src", "")),
            "related_name": _fetch_name(nb, e.get("src", "")),
        })

    return relations


async def _fetch_source_articles(nb, target_vid: str) -> list[dict]:
    """查找与该实体相关的报道文章

    路径：trace 边反查 → datarecord 节点(source_db/source_table/source_pk) → 回查 PG
    符合探数 v1.6 设计：图是知识层，原始数据在 PG，datarecord 只存溯源定位三要素。
    """
    # Step 1: trace 边反查，拿到所有 record VID
    rows = nb.query_rows(
        f'MATCH (r)-[t:trace]->(e) WHERE id(e) == "{target_vid}" '
        f'RETURN id(r) AS rid'
    )
    if not rows:
        return []

    # Step 2: FETCH datarecord 属性，收集 article_ids
    article_ids = []
    for r in rows:
        rec_vid = r.get("rid", "")
        rec_vid = rec_vid.strip('"') if rec_vid else ""
        if not rec_vid:
            continue
        row = nb.query_one(
            f'FETCH PROP ON datarecord "{rec_vid}" '
            f'YIELD datarecord.source_db AS sdb, datarecord.source_table AS stb, datarecord.source_pk AS spk'
        )
        if row and row.get("sdb") == "usn_monitor" and row.get("stb") == "articles":
            pk = row.get("spk", "")
            try:
                article_ids.append(int(pk))
            except (ValueError, TypeError):
                continue

    if not article_ids:
        return []

    # Step 3: 回查 PG 拿原始数据
    from app.db import async_session
    from sqlalchemy import text as sql_text

    async with async_session() as db:
        rows = await db.execute(sql_text("""
            SELECT id, title, url, published_at, content, processing_status
            FROM articles
            WHERE id = ANY(:ids)
            ORDER BY published_at DESC
            LIMIT 20
        """), {"ids": article_ids})
        return [{"id": str(r[0]), "title": r[1] or "", "url": r[2] or "",
                 "published_at": r[3].isoformat() if r[3] else "",
                 "content": r[4] or "", "status": r[5] or "pending"} for r in rows.fetchall()]
