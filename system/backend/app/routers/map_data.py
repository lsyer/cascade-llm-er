"""地图数据 API — 图优先架构"""
from fastapi import APIRouter, Query
from typing import Optional

from app.nebula_service import get_nebula, vid_to_id, vid_to_type
from app.routers.entities import parse_aliases

router = APIRouter()


@router.get("/overview")
async def map_overview(
    ship_type: Optional[str] = None,
    days: int = Query(90),
):
    """态势总览 — 从 Nebula 查所有实体及位置"""
    nb = get_nebula()

    # Equipment（活跃舰船）
    equip_rows = nb.query_rows(
        'LOOKUP ON equipment YIELD id(vertex) AS vid, equipment.name AS name, equipment.category AS category, equipment.state AS state, equipment.home_location AS home_location, equipment.latest_reported_at AS latest_reported_at, equipment.aliases AS aliases, equipment.equip_type AS equip_type'
    )

    # 过滤活跃 ship
    equipment = []
    for r in equip_rows:
        if r.get("equip_type") != "ship":
            continue
        if r.get("state") and r["state"] not in ("active", ""):
            continue
        if ship_type and r.get("category") != ship_type:
            continue

        r["id"] = vid_to_id(r.get("vid", ""))
        r["designation"] = ""
        r["status"] = r.get("state", "")
        r["aliases"] = parse_aliases(r.get("aliases"))

        # 解析坐标
        coords = r.get("home_location", "")
        if coords and "," in str(coords):
            parts = str(coords).split(",")
            try:
                r["home_lat"] = float(parts[0])
                r["home_lng"] = float(parts[1])
                r["lat"] = r["home_lat"]
                r["lng"] = r["home_lng"]
            except ValueError:
                pass

        equipment.append(r)

    # Locations
    loc_rows = nb.query_rows(
        'LOOKUP ON location YIELD id(vertex) AS vid, location.name AS name, location.region AS region, location.loc_type AS loc_type, location.coordinates AS coordinates'
    )
    locations = []
    for r in loc_rows:
        r["id"] = vid_to_id(r.get("vid", ""))
        r["country"] = ""
        r["location_type"] = r.pop("loc_type", "")
        coords = r.get("coordinates", "")
        if coords and "," in str(coords):
            parts = str(coords).split(",")
            try:
                r["lat"] = float(parts[0])
                r["lng"] = float(parts[1])
            except ValueError:
                pass
        locations.append(r)

    # Activities (events)
    event_rows = nb.query_rows(
        'LOOKUP ON event YIELD id(vertex) AS vid, event.name AS name, event.event_type AS event_type, event.start_date AS start_date, event.end_date AS end_date, event.location_name AS location_name'
    )
    activities = []
    for r in event_rows:
        r["id"] = vid_to_id(r.get("vid", ""))
        r["region"] = r.get("location_name", "")
        r["activity_type"] = r.get("event_type", "")
        activities.append(r)

    # Stats — 补全前端期望的字段
    total_ships = sum(1 for e in equipment if e.get("equip_type") == "ship")
    total_aircraft = sum(1 for e in equip_rows if e.get("equip_type") == "aircraft")
    total_weapons = sum(1 for e in equip_rows if e.get("equip_type") == "weapon")
    equipment_with_pos = sum(1 for e in equipment if e.get("lat"))

    # 近7日报道数 — 从 datarecord 查
    from datetime import datetime, timedelta, timezone
    seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    articles_7d = 0
    try:
        rec_rows = nb.query_rows(
            'LOOKUP ON datarecord YIELD datarecord.imported_at AS imported_at'
        )
        articles_7d = sum(1 for r in rec_rows if r.get("imported_at", "") >= seven_days_ago)
    except Exception:
        pass

    stats = {
        "total_equipment": len(equipment),
        "total_ships": total_ships,
        "total_aircraft": total_aircraft,
        "total_weapons": total_weapons,
        "total_locations": len(locations),
        "equipment_with_pos": equipment_with_pos,
        "articles_7d": articles_7d,
        "recent_activities": len(activities),
    }

    return {
        "equipment": equipment,
        "locations": locations,
        "activities": activities,
        "stats": stats,
    }


@router.get("/equipment/{eq_id}/track")
async def equipment_track(eq_id: int, days: int = Query(90)):
    """装备轨迹 — 查 PG 时序历史（图存状态，时序存历史）"""
    from app.db import async_session
    from sqlalchemy import text
    from datetime import datetime, timedelta, timezone

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    async with async_session() as db:
        r = await db.execute(text("""
            SELECT
                ST_Y(location) AS lat,
                ST_X(location) AS lng,
                reported_at,
                source_type,
                notes
            FROM equipment_positions
            WHERE equipment_vid = :vid AND reported_at >= :cutoff
            ORDER BY reported_at ASC
        """), {"vid": f"equip_{eq_id}", "cutoff": cutoff})
        rows = r.fetchall()

    return [
        {
            "lat": float(row[0]),
            "lng": float(row[1]),
            "reported_at": row[2].isoformat() if row[2] else None,
            "source_type": row[3],
            "notes": row[4],
        }
        for row in rows
    ]


@router.get("/heatmap")
async def heatmap():
    """热力图 — 基于地点坐标密度"""
    nb = get_nebula()
    loc_rows = nb.query_rows(
        'LOOKUP ON location YIELD location.name AS name, location.coordinates AS coordinates'
    )
    points = []
    for r in loc_rows:
        coords = r.get("coordinates", "")
        if coords and "," in str(coords):
            parts = str(coords).split(",")
            try:
                points.append({
                    "lat": float(parts[0]),
                    "lng": float(parts[1]),
                    "weight": 1,
                })
            except ValueError:
                pass
    return points
