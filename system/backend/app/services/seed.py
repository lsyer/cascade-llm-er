"""基础数据种子 — 已知美海军舰船和基地，直接写入 Nebula Graph"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging

log = logging.getLogger("usn.seed")

# 主要活跃舰船
SHIPS = [
    {"name": "USS Gerald R. Ford", "designation": "CVN-78", "category": "aircraft_carrier", "home_base": "Naval Station Norfolk", "home_lat": 36.9466, "home_lng": -76.3267, "aliases": ["福特号", "Ford", "CVN 78"]},
    {"name": "USS Dwight D. Eisenhower", "designation": "CVN-69", "category": "aircraft_carrier", "home_base": "Naval Station Norfolk", "home_lat": 36.9466, "home_lng": -76.3267, "aliases": ["艾森豪威尔号", "Ike", "Eisenhower"]},
    {"name": "USS Carl Vinson", "designation": "CVN-70", "category": "aircraft_carrier", "home_base": "Naval Air Station North Island", "home_lat": 32.6967, "home_lng": -117.2122, "aliases": ["卡尔·文森号", "Vinson"]},
    {"name": "USS Theodore Roosevelt", "designation": "CVN-71", "category": "aircraft_carrier", "home_base": "Naval Air Station North Island", "home_lat": 32.6967, "home_lng": -117.2122, "aliases": ["罗斯福号", "TR", "Roosevelt"]},
    {"name": "USS Abraham Lincoln", "designation": "CVN-72", "category": "aircraft_carrier", "home_base": "Naval Station Norfolk", "home_lat": 36.9466, "home_lng": -76.3267, "aliases": ["林肯号", "Lincoln", "Abe"]},
    {"name": "USS George Washington", "designation": "CVN-73", "category": "aircraft_carrier", "home_base": "Naval Station Norfolk", "home_lat": 36.9466, "home_lng": -76.3267, "aliases": ["华盛顿号", "GW", "Washington"]},
    {"name": "USS John C. Stennis", "designation": "CVN-74", "category": "aircraft_carrier", "home_base": "Naval Station Norfolk", "home_lat": 36.9466, "home_lng": -76.3267, "aliases": ["斯坦尼斯号", "Stennis", "JCS"]},
    {"name": "USS Harry S. Truman", "designation": "CVN-75", "category": "aircraft_carrier", "home_base": "Naval Station Norfolk", "home_lat": 36.9466, "home_lng": -76.3267, "aliases": ["杜鲁门号", "Truman", "HST"]},
    {"name": "USS Ronald Reagan", "designation": "CVN-76", "category": "aircraft_carrier", "home_base": "Naval Station Norfolk", "home_lat": 36.9466, "home_lng": -76.3267, "aliases": ["里根号", "Reagan", "RR"]},
    {"name": "USS George H.W. Bush", "designation": "CVN-77", "category": "aircraft_carrier", "home_base": "Naval Station Norfolk", "home_lat": 36.9466, "home_lng": -76.3267, "aliases": ["布什号", "Bush", "GHWB"]},
    {"name": "USS Arleigh Burke", "designation": "DDG-51", "category": "destroyer", "home_base": "Naval Station Norfolk", "home_lat": 36.9466, "home_lng": -76.3267, "aliases": ["伯克号"]},
    {"name": "USS John S. McCain", "designation": "DDG-56", "category": "destroyer", "home_base": "Fleet Activities Yokosuka", "home_lat": 35.2836, "home_lng": 139.6731, "aliases": ["麦凯恩号", "McCain"]},
    {"name": "USS Fitzgerald", "designation": "DDG-62", "category": "destroyer", "home_base": "Naval Station San Diego", "home_lat": 32.6736, "home_lng": -117.1206, "aliases": ["菲茨杰拉德号"]},
    {"name": "USS Mustin", "designation": "DDG-89", "category": "destroyer", "home_base": "Fleet Activities Yokosuka", "home_lat": 35.2836, "home_lng": 139.6731, "aliases": ["马斯汀号"]},
    {"name": "USS Milius", "designation": "DDG-69", "category": "destroyer", "home_base": "Fleet Activities Yokosuka", "home_lat": 35.2836, "home_lng": 139.6731, "aliases": ["米利厄斯号"]},
    {"name": "USS Benfold", "designation": "DDG-65", "category": "destroyer", "home_base": "Fleet Activities Yokosuka", "home_lat": 35.2836, "home_lng": 139.6731, "aliases": ["本福德号"]},
    {"name": "USS Higgins", "designation": "DDG-76", "category": "destroyer", "home_base": "Naval Station San Diego", "home_lat": 32.6736, "home_lng": -117.1206, "aliases": ["希金斯号"]},
    {"name": "USS Ralph Johnson", "designation": "DDG-114", "category": "destroyer", "home_base": "Naval Station Mayport", "home_lat": 30.3494, "home_lng": -81.4244, "aliases": ["拉尔夫·约翰逊号"]},
    {"name": "USS Daniel Inouye", "designation": "DDG-118", "category": "destroyer", "home_base": "Naval Station Pearl Harbor", "home_lat": 21.3697, "home_lng": -157.9553, "aliases": ["丹尼尔·井上号"]},
    {"name": "USS Bunker Hill", "designation": "CG-52", "category": "cruiser", "home_base": "Naval Station San Diego", "home_lat": 32.6736, "home_lng": -117.1206, "aliases": ["邦克山号"]},
    {"name": "USS Antietam", "designation": "CG-54", "category": "cruiser", "home_base": "Fleet Activities Yokosuka", "home_lat": 35.2836, "home_lng": 139.6731, "aliases": ["安提坦号"]},
    {"name": "USS Shiloh", "designation": "CG-67", "category": "cruiser", "home_base": "Fleet Activities Yokosuka", "home_lat": 35.2836, "home_lng": 139.6731, "aliases": ["夏伊洛号"]},
    {"name": "USS Chancellorsville", "designation": "CG-62", "category": "cruiser", "home_base": "Fleet Activities Yokosuka", "home_lat": 35.2836, "home_lng": 139.6731, "aliases": ["钱瑟勒斯维尔号"]},
    {"name": "USS America", "designation": "LHA-6", "category": "amphibious_assault", "home_base": "Sasebo Naval Base", "home_lat": 33.1583, "home_lng": 129.7219, "aliases": ["美国号", "America"]},
    {"name": "USS Tripoli", "designation": "LHA-7", "category": "amphibious_assault", "home_base": "Naval Station San Diego", "home_lat": 32.6736, "home_lng": -117.1206, "aliases": ["的黎波里号"]},
    {"name": "USS Wasp", "designation": "LHD-1", "category": "amphibious_assault", "home_base": "Naval Station Norfolk", "home_lat": 36.9466, "home_lng": -76.3267, "aliases": ["黄蜂号"]},
    {"name": "USS Boxer", "designation": "LHD-4", "category": "amphibious_assault", "home_base": "Naval Station San Diego", "home_lat": 32.6736, "home_lng": -117.1206, "aliases": ["拳师号"]},
    {"name": "USS Makin Island", "designation": "LHD-8", "category": "amphibious_assault", "home_base": "Naval Station San Diego", "home_lat": 32.6736, "home_lng": -117.1206, "aliases": ["马金岛号"]},
    {"name": "USS Essex", "designation": "LHD-2", "category": "amphibious_assault", "home_base": "Naval Station San Diego", "home_lat": 32.6736, "home_lng": -117.1206, "aliases": ["埃塞克斯号"]},
    {"name": "USS Connecticut", "designation": "SSN-22", "category": "submarine", "home_base": "Naval Base Kitsap", "home_lat": 47.5558, "home_lng": -122.6406, "aliases": ["康涅狄格号"]},
    {"name": "USS Asheville", "designation": "SSN-758", "category": "submarine", "home_base": "Naval Base Guam", "home_lat": 13.4443, "home_lng": 144.7937, "aliases": ["阿什维尔号"]},
    {"name": "USS Springfield", "designation": "SSN-761", "category": "submarine", "home_base": "Naval Base Guam", "home_lat": 13.4443, "home_lng": 144.7937, "aliases": ["斯普林菲尔德号"]},
    {"name": "USS Illinois", "designation": "SSN-786", "category": "submarine", "home_base": "Naval Base Pearl Harbor", "home_lat": 21.3697, "home_lng": -157.9553, "aliases": ["伊利诺伊号"]},
    {"name": "USS Delaware", "designation": "SSN-791", "category": "submarine", "home_base": "Naval Station Norfolk", "home_lat": 36.9466, "home_lng": -76.3267, "aliases": ["特拉华号"]},
    {"name": "USS Ohio", "designation": "SSBN-726", "category": "ssbn", "home_base": "Naval Base Kitsap", "home_lat": 47.5558, "home_lng": -122.6406, "aliases": ["俄亥俄号"]},
    {"name": "USS Nebraska", "designation": "SSBN-739", "category": "ssbn", "home_base": "Naval Base Kitsap", "home_lat": 47.5558, "home_lng": -122.6406, "aliases": ["内布拉斯加号"]},
    {"name": "USS Charleston", "designation": "LCS-18", "category": "littoral_combat", "home_base": "Naval Base San Diego", "home_lat": 32.6736, "home_lng": -117.1206, "aliases": ["查尔斯顿号"]},
    {"name": "USS Tulsa", "designation": "LCS-16", "category": "littoral_combat", "home_base": "Naval Base San Diego", "home_lat": 32.6736, "home_lng": -117.1206, "aliases": ["塔尔萨号"]},
]

BASES = [
    {"name": "Naval Station Norfolk", "country": "United States", "region": "EUCOM", "location_type": "naval_station", "lat": 36.9466, "lng": -76.3267, "aliases": ["诺福克海军基地", "Norfolk"]},
    {"name": "Naval Station San Diego", "country": "United States", "region": "INDOPACOM", "location_type": "naval_station", "lat": 32.6736, "lng": -117.1206, "aliases": ["圣迭戈海军基地", "San Diego NS"]},
    {"name": "Naval Air Station North Island", "country": "United States", "region": "INDOPACOM", "location_type": "naval_air_station", "lat": 32.6967, "lng": -117.2122, "aliases": ["北岛海军航空站"]},
    {"name": "Naval Station Mayport", "country": "United States", "region": "SOUTHCOM", "location_type": "naval_station", "lat": 30.3494, "lng": -81.4244, "aliases": ["梅波特海军基地"]},
    {"name": "Naval Base Kitsap", "country": "United States", "region": "INDOPACOM", "location_type": "naval_base", "lat": 47.5558, "lng": -122.6406, "aliases": ["基萨普海军基地", "Bremerton"]},
    {"name": "Naval Station Pearl Harbor", "country": "United States", "region": "INDOPACOM", "location_type": "naval_station", "lat": 21.3697, "lng": -157.9553, "aliases": ["珍珠港海军基地", "Pearl Harbor"]},
    {"name": "Naval Submarine Base Kings Bay", "country": "United States", "region": "EUCOM", "location_type": "submarine_base", "lat": 30.7967, "lng": -81.5256, "aliases": ["金斯湾潜艇基地"]},
    {"name": "Naval Air Station Jacksonville", "country": "United States", "region": "SOUTHCOM", "location_type": "naval_air_station", "lat": 30.2342, "lng": -81.6814, "aliases": ["杰克逊维尔海军航空站"]},
    {"name": "Naval Station Everett", "country": "United States", "region": "INDOPACOM", "location_type": "naval_station", "lat": 47.9864, "lng": -122.2244, "aliases": ["埃弗里特海军基地"]},
    {"name": "Fleet Activities Yokosuka", "country": "Japan", "region": "INDOPACOM", "location_type": "naval_base", "lat": 35.2836, "lng": 139.6731, "aliases": ["横须贺海军基地", "CFAY", "Yokosuka"]},
    {"name": "Sasebo Naval Base", "country": "Japan", "region": "INDOPACOM", "location_type": "naval_base", "lat": 33.1583, "lng": 129.7219, "aliases": ["佐世保海军基地", "CFAS", "Sasebo"]},
    {"name": "Naval Air Facility Atsugi", "country": "Japan", "region": "INDOPACOM", "location_type": "naval_air_station", "lat": 35.4553, "lng": 139.4497, "aliases": ["厚木海军航空设施", "Atsugi"]},
    {"name": "Marine Corps Air Station Iwakuni", "country": "Japan", "region": "INDOPACOM", "location_type": "marine_air_station", "lat": 34.1381, "lng": 132.2403, "aliases": ["岩国陆战队航空站", "Iwakuni"]},
    {"name": "Naval Base Guam", "country": "Guam (US)", "region": "INDOPACOM", "location_type": "naval_base", "lat": 13.4443, "lng": 144.7937, "aliases": ["关岛海军基地", "Guam"]},
    {"name": "Andersen Air Force Base", "country": "Guam (US)", "region": "INDOPACOM", "location_type": "air_base", "lat": 13.5833, "lng": 144.9167, "aliases": ["安德森空军基地", "Andersen AFB"]},
    {"name": "Fleet Activities Chinhae", "country": "South Korea", "region": "INDOPACOM", "location_type": "naval_base", "lat": 35.1028, "lng": 128.5697, "aliases": ["镇海海军基地", "Chinhae"]},
    {"name": "Changi Naval Base", "country": "Singapore", "region": "INDOPACOM", "location_type": "naval_base", "lat": 1.3067, "lng": 103.9894, "aliases": ["樟宜海军基地", "CNB"]},
    {"name": "Naval Support Activity Bahrain", "country": "Bahrain", "region": "CENTCOM", "location_type": "naval_base", "lat": 26.2361, "lng": 50.6111, "aliases": ["巴林海军支援设施", "NSA Bahrain"]},
    {"name": "Camp Lemonnier", "country": "Djibouti", "region": "AFRICOM", "location_type": "naval_base", "lat": 11.5417, "lng": 43.1583, "aliases": ["莱蒙尼尔营", "Djibouti"]},
    {"name": "Naval Station Rota", "country": "Spain", "region": "EUCOM", "location_type": "naval_station", "lat": 36.6453, "lng": -6.3497, "aliases": ["罗塔海军基地", "Rota"]},
    {"name": "Naval Support Activity Naples", "country": "Italy", "region": "EUCOM", "location_type": "naval_base", "lat": 40.8844, "lng": 14.2428, "aliases": ["那不勒斯海军支援设施", "Naples"]},
    {"name": "Naval Air Station Sigonella", "country": "Italy", "region": "EUCOM", "location_type": "naval_air_station", "lat": 37.4011, "lng": 14.9219, "aliases": ["西戈内拉海军航空站", "Sigonella"]},
    {"name": "Souda Bay Naval Base", "country": "Greece", "region": "EUCOM", "location_type": "naval_base", "lat": 35.4897, "lng": 24.1656, "aliases": ["苏达湾海军基地", "Souda Bay"]},
    {"name": "HMAS Stirling", "country": "Australia", "region": "INDOPACOM", "location_type": "naval_base", "lat": -32.2128, "lng": 115.3114, "aliases": ["斯特林海军基地", "Fleet Base West"]},
    {"name": "RAF Mildenhall", "country": "United Kingdom", "region": "EUCOM", "location_type": "air_base", "lat": 52.3614, "lng": 0.4864, "aliases": ["米尔登霍尔皇家空军基地"]},
]


def _esc(s):
    if s is None:
        return ""
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def _nb_exec(nql):
    """执行 Nebula nGQL"""
    from nebula3.gclient.net import ConnectionPool
    from nebula3.Config import Config
    pool = ConnectionPool()
    pool.init([("usn-nebula-graphd", 9669)], Config())
    sess = pool.get_session("root", "nebula")
    sess.execute("USE usn_main")
    r = sess.execute(nql)
    sess.release()
    pool.release()
    return r


def _nb_query(nql):
    """查 Nebula 返回 rows"""
    r = _nb_exec(nql)
    if not r.is_succeeded():
        return []
    keys = r.keys()
    rows = []
    for i in range(r.row_size()):
        row = {}
        for j, key in enumerate(keys):
            val = r.row_values(i)[j]
            try:
                row[key] = val.as_string() if not val.is_empty() else None
            except Exception:
                row[key] = str(val)
        rows.append(row)
    return rows


def _next_equip_id():
    rows = _nb_query('LOOKUP ON equipment YIELD id(vertex) AS vid')
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
    return max_id + 1


def _next_loc_id():
    rows = _nb_query('LOOKUP ON location YIELD id(vertex) AS vid')
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
    return max_id + 1


def _find_loc_by_name(name):
    rows = _nb_query(f'LOOKUP ON location WHERE location.name == "{_esc(name)}" YIELD id(vertex) AS vid')
    return rows[0]["vid"] if rows else None


def _insert_equipment(vid, ship, now):
    coords = f'{ship["home_lat"]},{ship["home_lng"]}' if ship.get("home_lat") else ""
    aliases = "||".join(ship.get("aliases", []))
    nql = f'''INSERT VERTEX equipment(name,aliases,equip_type,category,state,parent_unit,home_location,description,confidence,created_at,updated_at,latest_reported_at) VALUES "{vid}":("{_esc(ship['name'])}","{_esc(aliases)}","ship","{_esc(ship.get('category',''))}","active","","{_esc(coords)}","",1.0,"{now}","{now}","")'''
    r = _nb_exec(nql)
    return r.is_succeeded() or "Existed" in r.error_msg()


def _insert_location(vid, base, now):
    coords = f'{base["lat"]},{base["lng"]}'
    aliases = "||".join(base.get("aliases", []))
    nql = f'''INSERT VERTEX location(name,aliases,loc_type,region,coordinates,description,confidence,created_at,updated_at) VALUES "{vid}":("{_esc(base['name'])}","{_esc(aliases)}","{_esc(base.get('location_type','naval_base'))}","{_esc(base.get('region',''))}","{_esc(coords)}","",1.0,"{now}","{now}")'''
    r = _nb_exec(nql)
    return r.is_succeeded() or "Existed" in r.error_msg()


def _link_homebase(equip_vid, base_name, now):
    """建 ship → location 的 home_base locate 边（v1.6: 独立边）"""
    loc_vid = _find_loc_by_name(base_name)
    if not loc_vid:
        return False
    nql = f'INSERT EDGE locate(sub_type, confidence, source, created_at) VALUES "{equip_vid}"->"{loc_vid}":("home_base", 1.0, "seed", "{now}")'
    r = _nb_exec(nql)
    return r.is_succeeded() or "Existed" in r.error_msg()


async def seed_all(db: AsyncSession) -> dict:
    """导入基础数据到 Nebula Graph + PG（PG 保留用于 scraper 去重）"""
    from datetime import datetime
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    ship_count = 0
    base_count = 0
    link_count = 0

    # ── 1. 基地 → Nebula location + PG locations ──
    for base in BASES:
        # Nebula
        existing_vid = _find_loc_by_name(base["name"])
        if not existing_vid:
            loc_id = _next_loc_id()
            vid = f"loc_{loc_id}"
            if _insert_location(vid, base, now):
                base_count += 1
                log.info(f"[Seed] Location: {base['name']} → {vid}")



    await db.commit()

    # ── 2. 舰船 → Nebula equipment ──
    for ship in SHIPS:
        # Nebula
        eid = _next_equip_id()
        vid = f"equip_{eid}"
        if _insert_equipment(vid, ship, now):
            ship_count += 1
            log.info(f"[Seed] Equipment: {ship['name']} → {vid}")

            # 连 home_base
            if ship.get("home_base"):
                if _link_homebase(vid, ship["home_base"], now):
                    link_count += 1



    await db.commit()

    log.info(f"[Seed] Done: {ship_count} equipment, {base_count} locations, {link_count} home_base links (Nebula)")
    return {"equipment": ship_count, "locations": base_count, "links": link_count}
