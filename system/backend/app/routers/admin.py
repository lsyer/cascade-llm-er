"""管理 API：采集、抽取触发、任务状态"""
from fastapi import APIRouter, Depends, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import asyncio
import logging
import time

from app.db import get_db, async_session
from app.services.scraper import scrape_all_sources
from app.services.extractor import extract_pending_articles

log = logging.getLogger("usn")
router = APIRouter()

# 手动触发冷却期：5分钟
_last_manual_trigger = 0
MANUAL_COOLDOWN = 300  # seconds


@router.post("/scrape-extract")
async def trigger_scrape_extract():
    """手动触发采集+抽取（异步后台执行，5分钟冷却）"""
    global _last_manual_trigger
    from app.main import _task_state, run_scrape_and_extract
    
    if _task_state["running"]:
        return {"status": "already_running", "message": _task_state["message"]}
    
    elapsed = time.time() - _last_manual_trigger
    if elapsed < MANUAL_COOLDOWN:
        remaining = int(MANUAL_COOLDOWN - elapsed)
        return {"status": "cooldown", "message": f"请等待 {remaining} 秒后再次触发", "remaining": remaining}
    
    _last_manual_trigger = time.time()
    asyncio.ensure_future(run_scrape_and_extract())
    return {"status": "started"}


@router.get("/task-status")
async def task_status():
    """查询当前采集/抽取任务状态"""
    from app.main import _task_state
    return {
        "running": _task_state["running"],
        "phase": _task_state["phase"],
        "message": _task_state["message"],
        "scrape_new": _task_state["scrape_new"],
        "extract_done": _task_state["extract_done"],
        "log": _task_state["log"][-20:],
    }


@router.post("/scrape")
async def trigger_scrape(background_tasks: BackgroundTasks):
    """手动触发新闻采集"""
    background_tasks.add_task(run_scrape)
    return {"status": "scraping started"}


@router.post("/extract")
async def trigger_extract(background_tasks: BackgroundTasks):
    """手动触发实体抽取"""
    background_tasks.add_task(run_extract)
    return {"status": "extraction started"}


@router.post("/seed")
async def seed_basic_data(db: AsyncSession = Depends(get_db)):
    """导入基础舰船和基地数据"""
    from app.services.seed import seed_all
    count = await seed_all(db)
    return {"seeded": count}


@router.get("/stats")
async def system_stats(db: AsyncSession = Depends(get_db)):
    """系统统计 — 从 Nebula Graph 查询"""
    from app.nebula_service import get_nebula
    nb = get_nebula()
    nb.query("SUBMIT JOB STATS")
    import time; time.sleep(3)
    r = nb.query("SHOW STATS")

    stats = {}
    if r.is_succeeded():
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

    # PG 补充（positions 不在图里）
    pos_r = await db.execute(text("SELECT COUNT(*) FROM equipment_positions"))
    stats["positions"] = pos_r.scalar()

    return stats


async def run_scrape():
    async with async_session() as db:
        await scrape_all_sources(db)


async def run_extract():
    async with async_session() as db:
        await extract_pending_articles(db)


# ==================== 数据导入 API ====================

from pydantic import BaseModel
from typing import Optional


class EquipmentImport(BaseModel):
    name: str
    designation: Optional[str] = None
    equipment_type: Optional[str] = "ship"
    category: Optional[str] = None
    status: Optional[str] = "active"
    home_base: Optional[str] = None
    home_lat: Optional[float] = None
    home_lng: Optional[float] = None
    description: Optional[str] = None
    aliases: Optional[list[str]] = None
    source_url: Optional[str] = None


class PositionImport(BaseModel):
    equipment_name: Optional[str] = None
    designation: Optional[str] = None
    lat: float
    lng: float
    reported_at: str
    source_type: Optional[str] = "import"
    notes: Optional[str] = None


class LocationImport(BaseModel):
    name: str
    lat: float
    lng: float
    country: Optional[str] = None
    region: Optional[str] = None
    location_type: Optional[str] = None
    description: Optional[str] = None
    aliases: Optional[list[str]] = None


class ActivityImport(BaseModel):
    name: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    region: Optional[str] = None
    activity_type: Optional[str] = None
    description: Optional[str] = None


class ArticleImport(BaseModel):
    title: str
    url: str
    content: str
    published_at: Optional[str] = None
    source_name: Optional[str] = None


async def _resolve_equipment(designation=None, name=None):
    """按编号/名称查找已有器装（Nebula LOOKUP）"""
    from app.nebula_service import get_nebula
    nb = get_nebula()

    for q in [designation, name]:
        if q:
            rows = nb.query_rows(f'LOOKUP ON equipment WHERE equipment.name == "{_esc(q)}" YIELD id(vertex) AS vid, equipment.aliases AS aliases')
            if rows:
                vid = rows[0]["vid"]
                aliases = (rows[0].get("aliases") or "").split("||") if rows[0].get("aliases") else []
                return vid, aliases
    return None, []


def _esc(s):
    if s is None:
        return ""
    return str(s).replace('\\', '\\\\').replace('"', '\\"')


def _import_generate_trace(nb, vid: str, entity_type: str, source_label: str):
    """手动导入时创建 datarecord + generate 边，保证溯源链完整。"""
    from datetime import datetime
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    record_vid = f"record_import_{vid}"
    # 创建 datarecord 节点
    nb.query(
        f'INSERT VERTEX datarecord(source_db, source_table, source_pk, summary, imported_at) '
        f'VALUES "{record_vid}":("usn_monitor", "manual_import", "{vid}", '
        f'"手动导入: {_esc(source_label)}", "{now}")',
        space="usn_main",
    )
    # 创建 trace 边（v1.6: 替代旧 generate）
    nb.query(
        f'INSERT EDGE trace(sub_type, confidence, source, created_at) '
        f'VALUES "{record_vid}"->"{vid}":("generate", 1.0, "manual_import", "{now}")',
        space="usn_main",
    )



@router.post("/import/equipment")
async def import_equipment(items: list[EquipmentImport], db: AsyncSession = Depends(get_db)):
    """批量导入/更新器装数据（Nebula）"""
    from app.nebula_service import get_nebula
    from datetime import datetime
    nb = get_nebula()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    merged = 0
    created = 0
    for s in items:
        eq_vid, existing_aliases = await _resolve_equipment(s.designation, s.name)
        if eq_vid:
            # 合并别名
            new_aliases = existing_aliases[:]
            for a in (s.aliases or []):
                if a and a not in new_aliases:
                    new_aliases.append(a)
            # Nebula UPDATE
            updates = []
            if s.category:
                updates.append(f'category = "{_esc(s.category)}"')
            if s.status:
                updates.append(f'state = "{_esc(s.status)}"')
            updates.append(f'aliases = "{_esc("||".join(new_aliases))}"')
            updates.append(f'updated_at = "{now}"')
            if updates:
                nb.query(f'UPDATE VERTEX ON equipment "{eq_vid}" SET {", ".join(updates)}')

            merged += 1
        else:
            # 新建 — Nebula
            rows = nb.query_rows('LOOKUP ON equipment YIELD id(vertex) AS vid')
            max_id = 0
            for r in rows:
                vid = r.get("vid", "")
                if vid.startswith("equip_"):
                    try:
                        nid = int(vid.split("_", 1)[1])
                        if nid > max_id:
                            max_id = nid
                    except ValueError:
                        pass
            new_id = max_id + 1
            vid = f"equip_{new_id}"
            coords = f"{s.home_lat},{s.home_lng}" if s.home_lat and s.home_lng else ""
            aliases = "||".join(s.aliases or [])
            nb.query(f'''INSERT VERTEX equipment(name,aliases,equip_type,category,state,parent_unit,home_location,description,confidence,created_at,updated_at,latest_reported_at)
                         VALUES "{vid}":("{_esc(s.name)}","{_esc(aliases)}","{_esc(s.equipment_type or 'ship')}","{_esc(s.category or '')}","{_esc(s.status or 'active')}","","{_esc(coords)}","{_esc(s.description or '')}",1.0,"{now}","{now}","")''')
            # 溯源：datarecord + generate 边
            _import_generate_trace(nb, vid, "equipment", s.designation or s.name)


            created += 1
        await db.commit()
    log.info(f"[Import] Equipment: {merged} merged, {created} created from {len(items)} records")
    return {"merged": merged, "created": created, "total": len(items)}


@router.post("/import/positions")
async def import_positions(positions: list[PositionImport], db: AsyncSession = Depends(get_db)):
    """批量导入器装位置：PG 写时序历史 + Nebula 更新最新位置"""
    from app.nebula_service import get_nebula
    from datetime import datetime
    nb = get_nebula()

    imported = 0
    skipped = 0
    for p in positions:
        eq_vid, _ = await _resolve_equipment(p.designation, p.equipment_name)
        if not eq_vid:
            log.warning(f"[Import] Position: equipment not found ({p.designation or p.equipment_name})")
            skipped += 1
            continue
        try:
            reported_at = p.reported_at
            if isinstance(reported_at, str):
                from datetime import datetime as dt
                reported_at = dt.fromisoformat(reported_at.replace("Z", "+00:00"))
        except:
            skipped += 1
            continue

        # PG: 写时序历史
        await db.execute(text("""
            INSERT INTO equipment_positions (equipment_vid, location, reported_at, source_type, notes)
            VALUES (:vid, ST_SetSRID(ST_MakePoint(:lng, :lat), 4326), :at, :src, :notes)
        """), {
            "vid": eq_vid, "lat": p.lat, "lng": p.lng,
            "at": reported_at, "src": p.source_type, "notes": p.notes,
        })

        # Nebula: 更新最新位置（home_location + latest_reported_at）
        coords = f"{p.lat},{p.lng}"
        ts = reported_at.strftime("%Y-%m-%d %H:%M:%S") if reported_at else ""
        nb.query(f'UPDATE VERTEX ON equipment "{eq_vid}" SET home_location="{_esc(coords)}", latest_reported_at="{_esc(ts)}", updated_at="{_esc(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))}"')

        imported += 1

    await db.commit()
    log.info(f"[Import] Positions: {imported} imported, {skipped} skipped from {len(positions)} records")
    return {"imported": imported, "skipped": skipped, "total": len(positions)}


@router.post("/import/locations")
async def import_locations(locations: list[LocationImport], db: AsyncSession = Depends(get_db)):
    """批量导入位置（Nebula + PG）"""
    from app.nebula_service import get_nebula
    from datetime import datetime
    nb = get_nebula()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    merged = 0
    created = 0
    for b in locations:
        # Nebula: 检查是否已存在
        existing = nb.query_rows(f'LOOKUP ON location WHERE location.name == "{_esc(b.name)}" YIELD id(vertex) AS vid')
        if existing:
            merged += 1
            continue
        # 新建 Nebula location
        rows = nb.query_rows('LOOKUP ON location YIELD id(vertex) AS vid')
        max_id = 0
        for r in rows:
            vid = r.get("vid", "")
            if vid.startswith("loc_"):
                try:
                    nid = int(vid.split("_", 1)[1])
                    if nid > max_id:
                        max_id = nid
                except ValueError:
                    pass
        new_id = max_id + 1
        vid = f"loc_{new_id}"
        coords = f"{b.lat},{b.lng}" if b.lat and b.lng else ""
        aliases = "||".join(b.aliases or [])
        nb.query(f'''INSERT VERTEX location(name,aliases,loc_type,region,coordinates,description,confidence,created_at,updated_at)
                     VALUES "{vid}":("{_esc(b.name)}","{_esc(aliases)}","{_esc(b.location_type or 'other')}","{_esc(b.region or '')}","{_esc(coords)}","{_esc(b.description or '')}",1.0,"{now}","{now}")''')
        # 溯源：datarecord + generate 边
        _import_generate_trace(nb, vid, "location", b.name)


        created += 1
    await db.commit()
    log.info(f"[Import] Locations: {merged} existing, {created} created from {len(locations)} records")
    return {"merged": merged, "created": created, "total": len(locations)}


@router.post("/import/activities")
async def import_activities(activities: list[ActivityImport], db: AsyncSession = Depends(get_db)):
    """批量导入活动/事件（Nebula + PG）"""
    from app.nebula_service import get_nebula
    from datetime import datetime
    nb = get_nebula()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    merged = 0
    created = 0
    for o in activities:
        # Nebula: 检查是否已存在
        existing = nb.query_rows(f'LOOKUP ON event WHERE event.name == "{_esc(o.name)}" YIELD id(vertex) AS vid')
        if existing:
            merged += 1
            continue
        # 新建 Nebula event
        rows = nb.query_rows('LOOKUP ON event YIELD id(vertex) AS vid')
        max_id = 0
        for r in rows:
            vid = r.get("vid", "")
            if vid.startswith("event_"):
                try:
                    nid = int(vid.split("_", 1)[1])
                    if nid > max_id:
                        max_id = nid
                except ValueError:
                    pass
        new_id = max_id + 1
        vid = f"event_{new_id}"
        aliases = "||".join(o.aliases or []) if hasattr(o, 'aliases') else ""
        nb.query(f'''INSERT VERTEX event(name,aliases,event_type,start_date,end_date,occurred_at,location_name,description,confidence,created_at,updated_at)
                     VALUES "{vid}":("{_esc(o.name)}","{_esc(aliases)}","{_esc(o.activity_type or '')}","{_esc(str(o.start_date) if o.start_date else '')}","{_esc(str(o.end_date) if o.end_date else '')}","","{_esc(o.region or '')}","{_esc(o.description or '')}",1.0,"{now}","{now}")''')
        # 溯源：datarecord + generate 边
        _import_generate_trace(nb, vid, "activity", o.name)


        created += 1
    await db.commit()
    log.info(f"[Import] Activities: {merged} existing, {created} created from {len(activities)} records")
    return {"merged": merged, "created": created, "total": len(activities)}


# 文件导入：接受 JSON 文件上传
import json
from fastapi import UploadFile, File


@router.post("/import/articles")
async def import_articles(articles: list[ArticleImport], db: AsyncSession = Depends(get_db)):
    """导入原始文章，进入 LLM 抽取队列"""
    imported = 0
    skipped = 0
    for a in articles:
        # 去重
        r = await db.execute(text("SELECT id FROM articles WHERE url = :url"), {"url": a.url})
        if r.first():
            skipped += 1
            continue
        # 查找或创建 source
        source_id = None
        if a.source_name:
            r = await db.execute(text("SELECT id FROM news_sources WHERE name = :n"), {"n": a.source_name})
            row = r.first()
            if row:
                source_id = row[0]
            else:
                result = await db.execute(text("""
                    INSERT INTO news_sources (name, base_url, source_type, active)
                    VALUES (:name, :url, 'import', false)
                    RETURNING id
                """), {"name": a.source_name, "url": a.url})
                source_id = result.scalar()
        # 解析发布时间
        pub = None
        if a.published_at:
            from datetime import datetime as dt
            try:
                pub = dt.fromisoformat(a.published_at.replace("Z", "+00:00"))
            except: pass
        # 清理 HTML
        content = a.content
        if '<' in content:
            from bs4 import BeautifulSoup
            content = BeautifulSoup(content, "lxml").get_text(separator=" ", strip=True)
        await db.execute(text("""
            INSERT INTO articles (source_id, title, url, content, published_at, processing_status)
            VALUES (:sid, :title, :url, :content, :pub, 'pending')
            ON CONFLICT (url) DO NOTHING
        """), {
            "sid": source_id, "title": a.title, "url": a.url,
            "content": content[:10000], "pub": pub,
        })
        imported += 1
    await db.commit()
    log.info(f"[Import] Articles: {imported} new, {skipped} dup from {len(articles)} records")
    return {"imported": imported, "skipped": skipped, "total": len(articles)}


@router.post("/import/file")
async def import_file(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    上传 JSON 文件批量导入。文件格式：
    {"equipment": [...], "positions": [...], "locations": [...], "activities": [...], "articles": [...]}
    每个键可选，按需提供。
    """
    content = await file.read()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}"}

    results = {}

    if "articles" in data:
        articles = [ArticleImport(**a) for a in data["articles"]]
        results["articles"] = await import_articles(articles, db)

    if "equipment" in data:
        items = [EquipmentImport(**e) for e in data["equipment"]]
        results["equipment"] = await import_equipment(items, db)

    if "positions" in data:
        positions = [PositionImport(**p) for p in data["positions"]]
        results["positions"] = await import_positions(positions, db)

    if "locations" in data:
        locs = [LocationImport(**l) for l in data["locations"]]
        results["locations"] = await import_locations(locs, db)

    if "activities" in data:
        acts = [ActivityImport(**a) for a in data["activities"]]
        results["activities"] = await import_activities(acts, db)

    if not results:
        return {"error": "No valid data keys found. Expected: articles, equipment, positions, locations, activities"}

    return {"status": "ok", "results": results}


# ==================== 单篇文章重新处理 ====================

@router.post("/reprocess/{article_id}")
async def reprocess_article(article_id: int, background_tasks: BackgroundTasks):
    """撤回文章产生的所有数据（Nebula + PG），然后重新抽取"""
    async with async_session() as db:
        # 确认文章存在
        r = await db.execute(text("SELECT id, title FROM articles WHERE id = :id"), {"id": article_id})
        article = r.first()
        if not article:
            return {"status": "error", "message": f"Article {article_id} not found"}

        log.info(f"[Reprocess] Rolling back article {article_id}: {article[1][:60]}")

        # ── Nebula 撤回 ──
        from app.nebula_service import get_nebula, vid_to_type
        nb = get_nebula()

        deleted_entities = []
        kept_entities = []

        # 1. 溯源图：查 trace 边，找到该文章产生的所有实体
        record_vid = f"record_article_{article_id}"
        gen_rows = nb.query_rows(
            f'GO FROM "{record_vid}" OVER trace YIELD trace._dst AS target_vid, trace.sub_type AS target_type',
            space="usn_main"
        )

        affected_targets = []
        for g in gen_rows:
            target_vid = g.get("target_vid", "")
            target_type = g.get("target_type", "")
            if not target_vid:
                continue
            affected_targets.append((target_vid, target_type))

        # 2. 检查每个目标实体是否被其他 record 支撑
        for target_vid, target_type in affected_targets:
            other_supports = nb.query_rows(
                f'GO FROM "{target_vid}" OVER trace REVERSELY YIELD trace._src AS src_vid',
                space="usn_main"
            )
            other_records = [r for r in other_supports if r.get("src_vid", "") != record_vid]

            if len(other_records) == 0:
                # 孤立实体，删节点（WITH EDGE 自动清理所有独立边）
                nb.query(f'DELETE VERTEX "{target_vid}" WITH EDGE')
                deleted_entities.append(f"{target_type}:{target_vid}")
                log.info(f"[Reprocess] Deleted orphan: {target_vid}")
            else:
                # 非孤立实体：保留节点，但删除由此文章产生的独立边
                _EDGE_TYPES = ["social", "employ", "locate", "participate", "org_relation", "use_edge", "spatial", "trace"]
                for et in _EDGE_TYPES:
                    fwd_edges = nb.query_rows(
                        f'GO FROM "{target_vid}" OVER {et} '
                        f'YIELD {et}._dst AS dst, {et}.source AS src'
                    )
                    for fe in fwd_edges:
                        dst = fe.get("dst", "")
                        src = fe.get("src", "")
                        if dst and src == str(article_id):
                            nb.query(f'DELETE EDGE {et} "{target_vid}" -> "{dst}"')
                    rev_edges = nb.query_rows(
                        f'GO FROM "{target_vid}" OVER {et} REVERSELY '
                        f'YIELD {et}._src AS src_v, {et}.source AS src'
                    )
                    for re_ in rev_edges:
                        src_v = re_.get("src_v", "")
                        src = re_.get("src", "")
                        if src_v and src == str(article_id):
                            nb.query(f'DELETE EDGE {et} "{src_v}" -> "{target_vid}"')
                kept_entities.append(f"{target_type}:{target_vid} ({len(other_records)} other records)")
                log.info(f"[Reprocess] Kept: {target_vid} ({len(other_records)} other supports), cleaned edges")

        # 3. 删 datarecord + trace 边（必须在步骤2之后，否则 trace 反查失效）
        nb.query(f'DELETE VERTEX "{record_vid}" WITH EDGE', space="usn_main")

        # ── PG 撤回（保留兼容清理）──

        # 重置文章状态
        await db.execute(text("""
            UPDATE articles SET processing_status = 'pending', entities_extracted = false
            WHERE id = :id
        """), {"id": article_id})
        await db.commit()

        log.info(f"[Reprocess] Rolled back article {article_id}: "
                 f"deleted {len(deleted_entities)} entities, "
                 f"kept {len(kept_entities)} entities")

    # 后台重新抽取
    background_tasks.add_task(run_extract)

    return {
        "status": "reprocessing",
        "article_id": article_id,
        "deleted_entities": deleted_entities,
        "kept_entities": kept_entities,
    }


@router.post("/articles/{article_id}/reset")
async def reset_article_status(article_id: int):
    """重置文章状态为 pending（用于重试失败的抽取）"""
    async with async_session() as db:
        r = await db.execute(text("SELECT id, processing_status FROM articles WHERE id = :id"), {"id": article_id})
        row = r.first()
        if not row:
            return {"status": "error", "message": f"Article {article_id} not found"}
        old_status = row[1]
        await db.execute(text("""
            UPDATE articles SET processing_status = 'pending', entities_extracted = false
            WHERE id = :id
        """), {"id": article_id})
        await db.commit()
        log.info(f"[Reset] Article {article_id}: {old_status} → pending")
    return {"status": "reset", "article_id": article_id, "previous_status": old_status}


@router.post("/articles/{article_id}/refetch")
async def refetch_article_content(article_id: int):
    """重新采集文章正文（用于 GDELT 等来源只拿到标题未拿到正文的情况）"""
    import httpx
    from bs4 import BeautifulSoup

    async with async_session() as db:
        r = await db.execute(text("SELECT id, url, content FROM articles WHERE id = :id"), {"id": article_id})
        row = r.first()
        if not row:
            return {"status": "error", "message": f"Article {article_id} not found"}

        url = row[1]
        old_len = len(row[2] or "")

        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "lxml")

                content = ""
                for selector in ["article", "[role='main']", ".article-body", ".post-content",
                                  ".entry-content", ".story-body", ".article-content"]:
                    el = soup.select_one(selector)
                    if el:
                        content = el.get_text(separator=" ", strip=True)
                        if len(content) > 200:
                            break

                if not content or len(content) < 200:
                    ps = soup.find_all("p")
                    content = " ".join(p.get_text(strip=True) for p in ps)

                if len(content) < 50:
                    return {"status": "error", "message": f"抓取失败，仅获得 {len(content)} 字符"}

                # 更新正文 + 重置状态
                await db.execute(text("""
                    UPDATE articles SET content = :content, processing_status = 'pending', entities_extracted = false
                    WHERE id = :id
                """), {"content": content[:8000], "id": article_id})
                await db.commit()

                log.info(f"[Refetch] Article {article_id}: {old_len} → {len(content)} chars")
                return {"status": "ok", "old_len": old_len, "new_len": len(content)}

        except Exception as e:
            log.error(f"[Refetch] Article {article_id} failed: {e}")
            return {"status": "error", "message": str(e)}


@router.post("/extract-result/{article_id}")
async def save_extract_result(article_id: int, entities: dict):
    """接收外部抽取结果并写入 Nebula + PG。

    外部处理器（如 Hermes cron）完成 LLM 抽取后调用此接口写入。
    entities 格式同 extractor.extract_with_llm() 返回值：
    {
        "equipment": [...], "persons": [...], "locations": [...],
        "activities": [...], "relations": [...]
    }
    """
    from app.services.extractor import save_to_nebula

    async with async_session() as db:
        # 确认文章存在
        r = await db.execute(text("SELECT id, title, content FROM articles WHERE id = :id"), {"id": article_id})
        article = r.first()
        if not article:
            return {"status": "error", "message": f"Article {article_id} not found"}

        # 标记 processing
        await db.execute(text("""
            UPDATE articles SET processing_status = 'processing' WHERE id = :id
        """), {"id": article_id})
        await db.commit()

        try:
            # 写入 Nebula
            excerpt = (article[2] or "")[:500] if article[2] else ""
            save_to_nebula(article_id, article[1], entities, article_excerpt=excerpt)

            # 更新 PG 状态
            counts = {k: len(v) for k, v in entities.items() if isinstance(v, list)}
            await db.execute(text("""
                UPDATE articles SET processing_status = 'done', entities_extracted = true WHERE id = :id
            """), {"id": article_id})
            await db.commit()

            log.info(f"[External] Saved extract result for article {article_id}: {counts}")
            return {"status": "ok", "article_id": article_id, "counts": counts}

        except Exception as e:
            log.error(f"[External] Error saving result for article {article_id}: {e}", exc_info=True)
            await db.execute(text("""
                UPDATE articles SET processing_status = 'failed' WHERE id = :id
            """), {"id": article_id})
            await db.commit()
            return {"status": "error", "message": str(e)}


# ==================== 全量融合消歧 API ====================

from typing import Literal
import json as _json


class ResolveAction(BaseModel):
    action: Literal["merge", "discard", "keep"]
    target_vid: Optional[str] = None  # for merge action


# ─── 1. POST /api/admin/fusion/analyze ───

@router.post("/fusion/analyze")
async def fusion_analyze(db: AsyncSession = Depends(get_db)):
    """触发全量融合分析：从 Nebula 取所有实体，两两计算相似度，结果写入 PG pending_entities。"""
    from app.nebula_service import get_nebula
    from app.services.fusion import (
        fetch_all_entities, fetch_entity_props, compute_similarity,
        _quick_related, _tag_for_type, llm_judge_pairs, merge_entities,
        L1_MERGE, L1_REJECT,
    )

    nb = get_nebula()

    # Tags to analyze
    tags_config = [
        ("equipment", "equipment"),
        ("person", "person"),
        ("location", "location"),
        ("event", "activity"),
    ]

    # Clear previous pending analysis
    await db.execute(text("DELETE FROM pending_entities WHERE status = 'pending'"))
    await db.commit()

    total_pairs = 0
    total_merge = 0
    total_pending = 0
    total_different = 0
    summary_by_type = {}
    llm_batch = []  # collect pairs for LLM batch processing

    for tag, entity_type in tags_config:
        entities = fetch_all_entities(nb, tag)
        log.info(f"[Fusion/Analyze] {tag}: {len(entities)} entities found")

        if len(entities) < 2:
            summary_by_type[entity_type] = {"entities": len(entities), "pairs": 0, "merge": 0, "pending": 0}
            continue

        # Quick pre-filter: only compute pairs that might be related
        candidates = []
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                a = entities[i]
                b = entities[j]
                if _quick_related(a["name"], b["name"], a["aliases"], b["aliases"]):
                    candidates.append((a, b))

        log.info(f"[Fusion/Analyze] {tag}: {len(candidates)} candidate pairs after pre-filter")

        merge_count = 0
        pending_count = 0
        diff_count = 0

        # Batch: fetch props for all candidates
        props_cache = {}
        for a, b in candidates:
            if a["vid"] not in props_cache:
                props_cache[a["vid"]] = fetch_entity_props(nb, tag, a["vid"])
            if b["vid"] not in props_cache:
                props_cache[b["vid"]] = fetch_entity_props(nb, tag, b["vid"])

        for a, b in candidates:
            props_a = props_cache[a["vid"]]
            props_b = props_cache[b["vid"]]
            sim, decision = compute_similarity(entity_type, a, b, props_a, props_b)

            # L1 路由：merge/reject/escalate
            if decision == 'merge' or sim >= L1_MERGE:
                raw_data = {
                    "props_a": {k: v for k, v in (props_a or {}).items() if v},
                    "props_b": {k: v for k, v in (props_b or {}).items() if v},
                }
                await db.execute(text("""
                    INSERT INTO pending_entities (entity_type, vid_a, vid_b, name_a, name_b, similarity, raw_data, status, llm_verdict, resolved_at)
                    VALUES (:etype, :vid_a, :vid_b, :name_a, :name_b, :sim, :raw_data, 'merged', 'l1_auto_merge', NOW())
                """), {
                    "etype": entity_type, "vid_a": a["vid"], "vid_b": b["vid"],
                    "name_a": a["name"], "name_b": b["name"], "sim": round(sim, 4),
                    "raw_data": _json.dumps(raw_data, ensure_ascii=False),
                })
                merge_count += 1
                continue

            if decision == 'reject' or sim <= L1_REJECT:
                diff_count += 1
                continue

            # 灰色区间：送 LLM 批量判断
            total_pairs += 1
            llm_batch.append({
                "vid_a": a["vid"], "vid_b": b["vid"],
                "name_a": a["name"], "name_b": b["name"],
                "similarity": round(sim, 4),
                "props_a": {k: v for k, v in (props_a or {}).items() if v},
                "props_b": {k: v for k, v in (props_b or {}).items() if v},
            })

            # Flush LLM batch every 10 pairs
            if len(llm_batch) >= 10:
                counts = await _flush_llm_batch(nb, tag, entity_type, llm_batch, db)
                merge_count += counts["merge"]
                pending_count += counts["pending"]
                diff_count += counts["disambig"]
                llm_batch.clear()

        # Flush remaining LLM batch
        if llm_batch:
            counts = await _flush_llm_batch(nb, tag, entity_type, llm_batch, db)
            merge_count += counts["merge"]
            pending_count += counts["pending"]
            diff_count += counts["disambig"]
            llm_batch.clear()

        total_merge += merge_count
        total_pending += pending_count
        total_different += diff_count
        summary_by_type[entity_type] = {
            "entities": len(entities),
            "candidates": len(candidates),
            "pairs": merge_count + pending_count,
            "merge": merge_count,
            "pending": pending_count,
            "different": diff_count,
        }

    await db.commit()
    log.info(f"[Fusion/Analyze] Done: {total_merge} auto-merge, {total_pending} pending, {total_different} different")

    return {
        "status": "ok",
        "total_pairs_analyzed": total_pairs,
        "auto_merge": total_merge,
        "pending": total_pending,
        "different": total_different,
        "by_type": summary_by_type,
    }


async def _flush_llm_batch(nb, tag, entity_type, batch, db, **kwargs):
    """L2: Send a batch of pairs to LLM, execute results, return counts."""
    merge_c = 0
    pending_c = 0
    disambig_c = 0

    log.info(f"[Fusion/L2] Sending {len(batch)} pairs to LLM for {entity_type}")

    try:
        results = await llm_judge_pairs(entity_type, batch)
    except Exception as e:
        log.error(f"[Fusion/L2] LLM call failed: {e}")
        # LLM failed → all go to pending for human review
        for rec in batch:
            raw_data = {"props_a": rec["props_a"], "props_b": rec["props_b"]}
            await db.execute(text("""
                INSERT INTO pending_entities (entity_type, vid_a, vid_b, name_a, name_b, similarity, raw_data, status, notes)
                VALUES (:etype, :vid_a, :vid_b, :name_a, :name_b, :sim, :raw_data, 'pending', :notes)
            """), {
                "etype": entity_type, "vid_a": rec["vid_a"], "vid_b": rec["vid_b"],
                "name_a": rec["name_a"], "name_b": rec["name_b"], "sim": rec["similarity"],
                "raw_data": _json.dumps(raw_data, ensure_ascii=False),
                "notes": f"LLM failed: {e}",
            })
            pending_c += 1
        return {"merge": merge_c, "pending": pending_c, "disambig": disambig_c}

    for j, rec in enumerate(batch):
        key = f"pair_{j}"
        result = results.get(key, {})
        action = result.get("action", "").lower()
        reason = result.get("reason", "")
        raw_data = {"props_a": rec["props_a"], "props_b": rec["props_b"]}

        # Verify entities still exist
        try:
            pa = fetch_entity_props(nb, tag, rec["vid_a"])
            pb = fetch_entity_props(nb, tag, rec["vid_b"])
            if not pa or not pb:
                await db.execute(text("""
                    INSERT INTO pending_entities (entity_type, vid_a, vid_b, name_a, name_b, similarity, raw_data, status, notes, resolved_at)
                    VALUES (:etype, :vid_a, :vid_b, :name_a, :name_b, :sim, :raw_data, 'discarded', 'entity deleted', NOW())
                """), {
                    "etype": entity_type, "vid_a": rec["vid_a"], "vid_b": rec["vid_b"],
                    "name_a": rec["name_a"], "name_b": rec["name_b"], "sim": rec["similarity"],
                    "raw_data": _json.dumps(raw_data, ensure_ascii=False),
                })
                continue
        except Exception:
            continue

        if action == "merge":
            keep_vid = result.get("keep_vid", rec["vid_a"])
            if keep_vid not in (rec["vid_a"], rec["vid_b"]):
                keep_vid = rec["vid_a"]
            remove_vid = rec["vid_b"] if keep_vid == rec["vid_a"] else rec["vid_a"]

            try:
                merge_entities(nb, tag, keep_vid, remove_vid)
                await db.execute(text("""
                    INSERT INTO pending_entities (entity_type, vid_a, vid_b, name_a, name_b, similarity, raw_data, status, llm_verdict, resolved_vid, resolved_at, notes)
                    VALUES (:etype, :vid_a, :vid_b, :name_a, :name_b, :sim, :raw_data, 'merged', 'merge', :vid, NOW(), :notes)
                """), {
                    "etype": entity_type, "vid_a": rec["vid_a"], "vid_b": rec["vid_b"],
                    "name_a": rec["name_a"], "name_b": rec["name_b"], "sim": rec["similarity"],
                    "raw_data": _json.dumps(raw_data, ensure_ascii=False),
                    "vid": keep_vid, "notes": reason,
                })
                merge_c += 1
            except Exception as e:
                log.error(f"[Fusion/L2] Merge failed: {e}")
                await db.execute(text("""
                    INSERT INTO pending_entities (entity_type, vid_a, vid_b, name_a, name_b, similarity, raw_data, status, notes)
                    VALUES (:etype, :vid_a, :vid_b, :name_a, :name_b, :sim, :raw_data, 'pending', :notes)
                """), {
                    "etype": entity_type, "vid_a": rec["vid_a"], "vid_b": rec["vid_b"],
                    "name_a": rec["name_a"], "name_b": rec["name_b"], "sim": rec["similarity"],
                    "raw_data": _json.dumps(raw_data, ensure_ascii=False),
                    "notes": f"merge error: {e}",
                })
                pending_c += 1

        elif action in ("disambiguate", "different"):
            await db.execute(text("""
                INSERT INTO pending_entities (entity_type, vid_a, vid_b, name_a, name_b, similarity, raw_data, status, llm_verdict, resolved_at, notes)
                VALUES (:etype, :vid_a, :vid_b, :name_a, :name_b, :sim, :raw_data, 'disambig', 'disambiguate', NOW(), :notes)
            """), {
                "etype": entity_type, "vid_a": rec["vid_a"], "vid_b": rec["vid_b"],
                "name_a": rec["name_a"], "name_b": rec["name_b"], "sim": rec["similarity"],
                "raw_data": _json.dumps(raw_data, ensure_ascii=False),
                "notes": reason,
            })
            disambig_c += 1

        else:
            # LLM returned unclear/unknown → pending for human
            await db.execute(text("""
                INSERT INTO pending_entities (entity_type, vid_a, vid_b, name_a, name_b, similarity, raw_data, status, llm_verdict, notes)
                VALUES (:etype, :vid_a, :vid_b, :name_a, :name_b, :sim, :raw_data, 'pending', :verdict, :notes)
            """), {
                "etype": entity_type, "vid_a": rec["vid_a"], "vid_b": rec["vid_b"],
                "name_a": rec["name_a"], "name_b": rec["name_b"], "sim": rec["similarity"],
                "raw_data": _json.dumps(raw_data, ensure_ascii=False),
                "verdict": action or "unclear",
                "notes": reason or "LLM returned unclear result",
            })
            pending_c += 1

    return {"merge": merge_c, "pending": pending_c, "disambig": disambig_c}


# ─── 2. POST /api/admin/fusion/execute ───

@router.post("/fusion/execute")
async def fusion_execute(db: AsyncSession = Depends(get_db)):
    """执行融合：从 PG pending_entities 取 status='pending' 的记录，LLM 判断后执行合并。"""
    from app.nebula_service import get_nebula
    from app.services.fusion import (
        llm_judge_pairs, merge_entities, fetch_entity_props,
        _tag_for_type,
    )

    nb = get_nebula()

    # Fetch all pending records
    r = await db.execute(text("""
        SELECT id, entity_type, vid_a, vid_b, name_a, name_b, similarity, raw_data
        FROM pending_entities WHERE status = 'pending'
        ORDER BY similarity DESC
    """))
    rows = r.fetchall()

    if not rows:
        return {"status": "ok", "message": "No pending entities to process", "merged": 0, "disambiguated": 0}

    log.info(f"[Fusion/Execute] Processing {len(rows)} pending pairs")

    # Group by entity_type for batch LLM
    by_type: dict[str, list[dict]] = {}
    for row in rows:
        etype = row[1]
        raw_data = row[7] if isinstance(row[7], dict) else (_json.loads(row[7]) if row[7] else {})
        record = {
            "id": row[0],
            "entity_type": etype,
            "vid_a": row[2],
            "vid_b": row[3],
            "name_a": row[4],
            "name_b": row[5],
            "similarity": row[6],
            "props_a": raw_data.get("props_a", {}),
            "props_b": raw_data.get("props_b", {}),
        }
        by_type.setdefault(etype, []).append(record)

    total_merged = 0
    total_disambig = 0
    total_errors = 0

    for etype, records in by_type.items():
        tag = _tag_for_type(etype)

        # Batch LLM: send up to 10 pairs at a time
        batch_size = 10
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            results = await llm_judge_pairs(etype, batch)

            for j, rec in enumerate(batch):
                key = f"pair_{j}"
                result = results.get(key, {})

                # Verify both entities still exist
                try:
                    pa = fetch_entity_props(nb, tag, rec["vid_a"])
                    pb = fetch_entity_props(nb, tag, rec["vid_b"])
                    if not pa or not pb:
                        log.warning(f"[Fusion/Execute] Skipping pair {rec['vid_a']}/{rec['vid_b']}: entity deleted")
                        await db.execute(text(
                            "UPDATE pending_entities SET status = 'discarded', notes = 'entity deleted', resolved_at = NOW() WHERE id = :id"
                        ), {"id": rec["id"]})
                        continue
                except Exception:
                    await db.execute(text(
                        "UPDATE pending_entities SET status = 'discarded', notes = 'fetch error', resolved_at = NOW() WHERE id = :id"
                    ), {"id": rec["id"]})
                    continue

                action = result.get("action", "").lower()
                reason = result.get("reason", "")

                if action == "merge":
                    keep_vid = result.get("keep_vid", rec["vid_a"])
                    # Validate keep_vid
                    if keep_vid not in (rec["vid_a"], rec["vid_b"]):
                        log.warning(f"[Fusion/Execute] Invalid keep_vid={keep_vid}, using vid_a={rec['vid_a']}")
                        keep_vid = rec["vid_a"]
                    remove_vid = rec["vid_b"] if keep_vid == rec["vid_a"] else rec["vid_a"]

                    try:
                        merge_entities(nb, tag, keep_vid, remove_vid)
                        await db.execute(text("""
                            UPDATE pending_entities
                            SET status = 'merged', resolved_vid = :vid, llm_verdict = :verdict, resolved_at = NOW(), notes = :notes
                            WHERE id = :id
                        """), {
                            "vid": keep_vid,
                            "verdict": action,
                            "notes": reason,
                            "id": rec["id"],
                        })
                        total_merged += 1
                    except Exception as e:
                        log.error(f"[Fusion/Execute] Merge failed: {e}")
                        await db.execute(text(
                            "UPDATE pending_entities SET notes = :notes WHERE id = :id"
                        ), {"notes": f"merge error: {e}", "id": rec["id"]})
                        total_errors += 1

                else:
                    # disambiguate
                    await db.execute(text("""
                        UPDATE pending_entities
                        SET status = 'disambig', llm_verdict = :verdict, resolved_at = NOW(), notes = :notes
                        WHERE id = :id
                    """), {
                        "verdict": "disambiguate",
                        "notes": reason,
                        "id": rec["id"],
                    })
                    total_disambig += 1

    await db.commit()

    # NOTE: LLM batch results are NOT human-confirmed yet — they go through
    # human review before entering the feedback training loop.
    # The retrain trigger is in resolve_pending_entity (manual resolution),
    # not here. LLM batch is a pre-filtering step, not ground truth.
    log.info(f"[Fusion/Execute] Done: {total_merged} merged, {total_disambig} disambiguated, {total_errors} errors "
             f"(pending human confirmation for feedback)")

    return {
        "status": "ok",
        "total": len(rows),
        "merged": total_merged,
        "disambiguated": total_disambig,
        "errors": total_errors,
    }


# ─── 3. GET /api/admin/pending-entities ───

@router.get("/pending-entities")
async def list_pending_entities(
    status: Optional[str] = None,
    entity_type: Optional[str] = None,
    limit: int = Query(50, le=500),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """列出待决实体，支持 status/entity_type 筛选和分页。"""
    conditions = []
    params: dict = {"lim": limit, "off": offset}

    if status:
        conditions.append("status = :status")
        params["status"] = status
    if entity_type:
        conditions.append("entity_type = :etype")
        params["etype"] = entity_type

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    # Count
    count_r = await db.execute(text(f"SELECT COUNT(*) FROM pending_entities {where}"), params)
    total = count_r.scalar()

    # List
    r = await db.execute(text(f"""
        SELECT id, entity_type, vid_a, vid_b, name_a, name_b, similarity,
               status, llm_verdict, resolved_vid, notes, created_at, resolved_at
        FROM pending_entities {where}
        ORDER BY similarity DESC NULLS LAST, created_at DESC
        LIMIT :lim OFFSET :off
    """), params)
    rows = r.fetchall()

    items = []
    for row in rows:
        items.append({
            "id": row[0],
            "entity_type": row[1],
            "vid_a": row[2],
            "vid_b": row[3],
            "name_a": row[4],
            "name_b": row[5],
            "similarity": row[6],
            "status": row[7],
            "llm_verdict": row[8],
            "resolved_vid": row[9],
            "notes": row[10],
            "created_at": row[11].isoformat() if row[11] else None,
            "resolved_at": row[12].isoformat() if row[12] else None,
        })

    return {"total": total, "items": items}


# ─── 3.5 GET /api/admin/pending-entities/{id}/detail ───

@router.get("/pending-entities/{pending_id}/detail")
async def get_pending_detail(
    pending_id: int,
    db: AsyncSession = Depends(get_db),
):
    """返回待决实体对 A/B 的完整属性对比 + 相关报道，供人工判断。"""
    from app.nebula_service import get_nebula
    from app.services.fusion import _tag_for_type
    nb = get_nebula()

    # 取 pending 记录
    r = await db.execute(text("""
        SELECT id, entity_type, vid_a, vid_b, name_a, name_b, similarity,
               status, llm_verdict, notes, raw_data, created_at
        FROM pending_entities WHERE id = :id
    """), {"id": pending_id})
    row = r.fetchone()
    if not row:
        return {"error": "not found"}

    entity_type = row[1]
    vid_a, vid_b = row[2], row[3]
    tag = _tag_for_type(entity_type)

    # Entity tag → fields mapping
    _TAG_FIELDS = {
        'equipment': ['name', 'equip_type', 'category', 'state', 'home_location', 'latest_reported_at', 'parent_unit', 'aliases', 'description', 'confidence', 'created_at', 'updated_at'],
        'person': ['name', 'occupation', 'org_name', 'aliases', 'description', 'confidence', 'created_at', 'updated_at'],
        'location': ['name', 'loc_type', 'region', 'coordinates', 'aliases', 'description', 'confidence', 'created_at', 'updated_at'],
        'event': ['name', 'event_type', 'start_date', 'end_date', 'location_name', 'aliases', 'description', 'confidence', 'created_at', 'updated_at'],
        'organization': ['name', 'org_type', 'aliases', 'description', 'confidence', 'created_at', 'updated_at'],
    }

    def _fetch_props(vid):
        fields = _TAG_FIELDS.get(tag, ['name', 'aliases', 'description', 'confidence'])
        field_str = ', '.join(f'{tag}.{f} AS {f}' for f in fields)
        try:
            row = nb.query_one(f'FETCH PROP ON {tag} "{vid}" YIELD {field_str}')
            return row or {}
        except Exception:
            return {}

    async def _fetch_related(vid):
        """查 trace 反查 → datarecord → PG articles"""
        try:
            rows = nb.query_rows(
                f'MATCH (r)-[t:trace]->(e) WHERE id(e) == "{vid}" RETURN id(r) AS rid'
            )
        except Exception:
            return []
        if not rows:
            return []
        article_ids = []
        for rr in rows:
            rec_vid = rr.get("rid", "").strip('"')
            if not rec_vid:
                continue
            rec = nb.query_one(
                f'FETCH PROP ON datarecord "{rec_vid}" '
                f'YIELD datarecord.source_db AS sdb, datarecord.source_table AS stb, datarecord.source_pk AS spk'
            )
            if rec and rec.get("sdb") == "usn_monitor" and rec.get("stb") == "articles":
                try:
                    article_ids.append(int(rec.get("spk", "")))
                except (ValueError, TypeError):
                    continue
        if not article_ids:
            return []
        ar = await db.execute(text("""
            SELECT id, title, url, published_at, content
            FROM articles WHERE id = ANY(:ids)
            ORDER BY published_at DESC LIMIT 10
        """), {"ids": article_ids})
        return [{"id": str(a[0]), "title": a[1] or "", "url": a[2] or "",
                 "published_at": a[3].isoformat() if a[3] else "",
                 "content": (a[4] or "")[:500]} for a in ar.fetchall()]

    def _extract_sentences(vid, articles):
        """从相关文章内容中提取包含实体名最长的一句话"""
        props = _fetch_props(vid)
        name = props.get("name", "")
        if not name or not articles:
            return ""
        import re
        best = ""
        for art in articles:
            content = art.get("content", "")
            sentences = re.split(r'[。！？；\n]', content)
            for s in sentences:
                if name in s and len(s) > len(best):
                    best = s.strip()
        return best[:300] if best else ""

    props_a = _fetch_props(vid_a)
    props_b = _fetch_props(vid_b)
    articles_a = await _fetch_related(vid_a)
    articles_b = await _fetch_related(vid_b)
    sent_a = _extract_sentences(vid_a, articles_a)
    sent_b = _extract_sentences(vid_b, articles_b)

    # Parse raw_data if exists
    raw = row[10]  # raw_data
    raw_data = {}
    if raw:
        try:
            raw_data = _json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            pass

    return {
        "id": row[0],
        "entity_type": entity_type,
        "vid_a": vid_a,
        "vid_b": vid_b,
        "name_a": row[4],
        "name_b": row[5],
        "similarity": float(row[6]) if row[6] else 0,
        "status": row[7],
        "llm_verdict": row[8],
        "notes": row[9],
        "created_at": row[11].isoformat() if row[11] else None,
        "props_a": props_a,
        "props_b": props_b,
        "articles_a": articles_a,
        "articles_b": articles_b,
        "longest_sentence_a": sent_a,
        "longest_sentence_b": sent_b,
        "llm_analysis": raw_data,
    }


# ─── 4. POST /api/admin/pending-entities/{id}/resolve ───

@router.post("/pending-entities/{pending_id}/resolve")
async def resolve_pending_entity(
    pending_id: int,
    body: ResolveAction,
    db: AsyncSession = Depends(get_db),
):
    """手动处理一条待决实体：merge（合并到指定 vid）/ discard（丢弃）/ keep（标记为独立实体）。"""
    # Fetch the record
    r = await db.execute(text("""
        SELECT id, entity_type, vid_a, vid_b, name_a, name_b, similarity, status
        FROM pending_entities WHERE id = :id
    """), {"id": pending_id})
    row = r.first()
    if not row:
        return {"status": "error", "message": f"Pending entity {pending_id} not found"}

    if row[7] not in ("pending",):
        return {"status": "error", "message": f"Record already resolved with status '{row[7]}'"}

    etype = row[1]
    vid_a = row[2]
    vid_b = row[3]

    if body.action == "merge":
        if not body.target_vid:
            return {"status": "error", "message": "target_vid required for merge action"}

        target_vid = body.target_vid
        if target_vid not in (vid_a, vid_b):
            return {"status": "error", "message": f"target_vid must be one of: {vid_a}, {vid_b}"}

        remove_vid = vid_b if target_vid == vid_a else vid_a

        # Execute merge in Nebula
        from app.nebula_service import get_nebula
        from app.services.fusion import merge_entities, _tag_for_type

        nb = get_nebula()
        tag = _tag_for_type(etype)

        try:
            merge_entities(nb, tag, target_vid, remove_vid)
        except Exception as e:
            return {"status": "error", "message": f"Merge failed: {e}"}

        await db.execute(text("""
            UPDATE pending_entities
            SET status = 'merged', resolved_vid = :vid, resolved_at = NOW(),
                notes = 'manual merge', resolved_by = 'admin'
            WHERE id = :id
        """), {"vid": target_vid, "id": pending_id})

        await db.commit()

        # Trigger feedback loop retraining check
        try:
            from app.services.feedback import maybe_retrain_async
            from app.services.feedback import count_new_resolved_since_last_train
            n_new = await db.run_sync(count_new_resolved_since_last_train)
            if n_new >= 50:
                retrain_result = await db.run_sync(
                    lambda sync_db: maybe_retrain_async(sync_db)
                )
                if retrain_result.get('trained'):
                    log.info(f"[Feedback] Retrained after resolve: {retrain_result.get('message')}")
        except Exception as e:
            log.warning(f"[Feedback] Retrain check failed (non-fatal): {e}")

        return {"status": "ok", "action": "merge", "kept": target_vid, "removed": remove_vid}

    elif body.action == "discard":
        await db.execute(text("""
            UPDATE pending_entities
            SET status = 'discarded', resolved_at = NOW(), notes = 'manual discard', resolved_by = 'admin'
            WHERE id = :id
        """), {"id": pending_id})
        await db.commit()
        return {"status": "ok", "action": "discard"}

    elif body.action == "keep":
        # Both entities are independent — mark as disambiguated
        await db.execute(text("""
            UPDATE pending_entities
            SET status = 'disambig', resolved_at = NOW(), notes = 'manual disambiguate: both entities are independent', resolved_by = 'admin'
            WHERE id = :id
        """), {"id": pending_id})
        await db.commit()

        # Trigger feedback loop retraining check
        try:
            from app.services.feedback import maybe_retrain_async
            from app.services.feedback import count_new_resolved_since_last_train
            n_new = await db.run_sync(count_new_resolved_since_last_train)
            if n_new >= 50:
                retrain_result = await db.run_sync(
                    lambda sync_db: maybe_retrain_async(sync_db)
                )
                if retrain_result.get('trained'):
                    log.info(f"[Feedback] Retrained after resolve: {retrain_result.get('message')}")
        except Exception as e:
            log.warning(f"[Feedback] Retrain check failed (non-fatal): {e}")

        return {"status": "ok", "action": "keep", "message": "Both entities marked as independent"}

    return {"status": "error", "message": f"Unknown action: {body.action}"}


# ─── 5. GET /api/admin/graph/stats ───

@router.get("/graph/stats")
async def graph_stats():
    """图统计：返回节点和边数量（从 Nebula STATS）。"""
    from app.nebula_service import get_nebula

    nb = get_nebula()
    nb.query("SUBMIT JOB STATS")
    import time as _t
    _t.sleep(3)
    r = nb.query("SHOW STATS")

    stats = {"tags": {}, "edges": {}, "total_vertices": 0, "total_edges": 0}
    if r.is_succeeded():
        for i in range(r.row_size()):
            vals = r.row_values(i)
            try:
                t = vals[0].as_string()
                name = vals[1].as_string()
                count = _safe_int(vals[2])
            except Exception:
                continue
            if t == "Tag":
                stats["tags"][name] = count
                stats["total_vertices"] += count
            elif t == "Edge":
                stats["edges"][name] = count
                stats["total_edges"] += count

    return stats


# ─── 6. GET /api/admin/graph/nodes ───

@router.get("/graph/nodes")
async def graph_nodes(
    tag: str = Query("equipment"),
    search: str = Query(""),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """图节点浏览：按 Tag 类型列出节点，支持搜索。"""
    from app.nebula_service import get_nebula

    nb = get_nebula()

    valid_tags = ["equipment", "location", "event", "person", "organization", "datarecord", "dataset"]
    if tag not in valid_tags:
        return {"total": 0, "items": []}

    search_safe = search.replace("'", "").replace('"', '')

    # 查询节点 — 用 MATCH + properties(n)
    # Nebula MATCH WHERE 属性过滤不可靠，搜索在 Python 层做
    nql = f"MATCH (n:{tag}) RETURN id(n), properties(n) AS props SKIP {offset} LIMIT {limit}"
    count_nql = f"MATCH (n:{tag}) RETURN count(n) AS cnt"

    if search:
        # 搜索模式：取全量 vid+name，Python 过滤
        all_nql = f"MATCH (n:{tag}) RETURN id(n), properties(n) AS props"
        r_all = nb.query(all_nql)
        all_items = []
        if r_all.is_succeeded():
            for i in range(r_all.row_size()):
                vals = r_all.row_values(i)
                vid = _safe_val(vals[0])
                row = {"vid": vid}
                if not vals[1].is_null():
                    try:
                        pm = vals[1].as_map()
                        for k, v in pm.items():
                            if not v.is_null():
                                row[k] = _safe_val(v)
                    except Exception:
                        pass
                # 搜索匹配
                searchable = " ".join(str(v) for v in row.values() if v).lower()
                if search_safe.lower() in searchable:
                    all_items.append(row)
        return {"total": len(all_items), "items": all_items[offset:offset + limit]}

    # 非搜索模式：直接分页
    cr = nb.query(count_nql)
    total = _safe_int(cr.row_values(0)[0]) if cr.is_succeeded() and cr.row_size() > 0 else 0

    r = nb.query(nql)
    items = []
    if r.is_succeeded():
        for i in range(r.row_size()):
            vals = r.row_values(i)
            vid = _safe_val(vals[0])
            row = {"vid": vid}
            if not vals[1].is_null():
                try:
                    pm = vals[1].as_map()
                    for k, v in pm.items():
                        if not v.is_null():
                            row[k] = _safe_val(v)
                except Exception:
                    pass
            items.append(row)

    return {"total": total, "items": items}


def _safe_val(v):
    """安全取 Nebula 值。"""
    if v.is_null():
        return None
    try:
        return v.as_string()
    except Exception:
        pass
    try:
        return v.as_int()
    except Exception:
        pass
    try:
        return v.as_double()
    except Exception:
        pass
    return str(v)


def _safe_int(v):
    """安全取整数值。"""
    if v.is_null():
        return 0
    try:
        return int(v.as_string())
    except Exception:
        pass
    try:
        return v.as_int()
    except Exception:
        pass
    return 0


# ─── 7. GET /api/admin/graph/node/<vid> ───

@router.get("/graph/node/{vid:path}")
async def graph_node_detail(vid: str):
    """节点详情：属性 + 关联边。"""
    from app.nebula_service import get_nebula

    nb = get_nebula()
    escaped = vid.replace('"', '\\"')

    # 用 MATCH 查节点
    node = None
    mr = nb.query(f'MATCH (n) WHERE id(n) == "{escaped}" RETURN id(n) AS vid, labels(n) AS tags, properties(n) AS props LIMIT 1')
    if mr.is_succeeded() and mr.row_size() > 0:
        vals = mr.row_values(0)
        node = {"vid": _safe_val(vals[0])}
        try:
            node["tags_raw"] = _safe_val(vals[1])
        except:
            pass
        try:
            node["props_raw"] = _safe_val(vals[2])
        except:
            pass

    # 关联边
    edges = []
    er = nb.query(f'GO FROM "{escaped}" OVER * BIDIRECT YIELD src(edge) AS src, dst(edge) AS dst, type(edge) AS etype')
    if er.is_succeeded():
        for i in range(er.row_size()):
            ev = er.row_values(i)
            edges.append({
                "src": _safe_val(ev[0]),
                "dst": _safe_val(ev[1]),
                "type": _safe_val(ev[2]),
            })

    return {"node": node, "edges": edges}
