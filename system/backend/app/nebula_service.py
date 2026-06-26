"""Nebula Graph 查询服务 — USN Monitor 图优先架构的核心"""
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from functools import lru_cache
import logging

log = logging.getLogger("usn.nebula")

NEBULA_HOST = "usn-nebula-graphd"
NEBULA_PORT = 9669
NEBULA_USER = "root"
NEBULA_PASS = "nebula"
MAIN_SPACE = "usn_main"


class NebulaService:
    """Nebula 连接池 + 查询封装"""

    def __init__(self):
        self._pool = None

    def init(self):
        if self._pool:
            return
        config = Config()
        self._pool = ConnectionPool()
        self._pool.init([(NEBULA_HOST, NEBULA_PORT)], config)
        log.info("Nebula connection pool initialized")

    def get_session(self, space: str = MAIN_SPACE):
        if not self._pool:
            self.init()
        sess = self._pool.get_session(NEBULA_USER, NEBULA_PASS)
        sess.execute(f"USE {space}")
        return sess

    def release(self, sess):
        sess.release()

    def query(self, nql: str, space: str = MAIN_SPACE):
        """执行 nGQL，返回 ResultSet"""
        sess = self.get_session(space)
        try:
            r = sess.execute(nql)
            return r
        finally:
            self.release(sess)

    def query_rows(self, nql: str, space: str = MAIN_SPACE) -> list[dict]:
        """执行 nGQL，返回 [{key: val, ...}, ...]"""
        r = self.query(nql, space)
        if not r.is_succeeded():
            log.warning(f"Nebula query failed: {r.error_msg()[:100]} | NQL: {nql[:100]}")
            return []
        keys = r.keys()
        rows = []
        for i in range(r.row_size()):
            row = {}
            for j, key in enumerate(keys):
                val = r.row_values(i)[j]
                row[key] = _unwrap(val)
            rows.append(row)
        return rows

    def query_one(self, nql: str, space: str = MAIN_SPACE) -> dict | None:
        """返回第一行或 None"""
        rows = self.query_rows(nql, space)
        return rows[0] if rows else None


# 全局单例
_nebula: NebulaService | None = None


def get_nebula() -> NebulaService:
    global _nebula
    if _nebula is None:
        _nebula = NebulaService()
        _nebula.init()
    return _nebula


def _unwrap(val):
    """Nebula ValueWrapper → Python 值"""
    try:
        if val.is_empty():
            return None
        return val.as_string()
    except Exception:
        try:
            return val.as_int()
        except Exception:
            try:
                return val.as_double()
            except Exception:
                try:
                    return val.as_bool()
                except Exception:
                    return str(val)


# ── 实体类型映射 ──
TAG_TO_TYPE = {
    "person": "person",
    "organization": "organization",
    "identifier": "identifier",
    "location": "location",
    "event": "event",
    "equipment": "equipment",
    "dataset": "dataset",
    "datarecord": "datarecord",
}

# VID prefix → entity_type (for reverse mapping)
PREFIX_TO_TYPE = {
    "person": "person",
    "org": "organization",
    "id": "identifier",
    "loc": "location",
    "event": "event",
    "equip": "equipment",
    "ds": "dataset",
    "dr": "datarecord",
}


def vid_to_type(vid: str) -> str:
    """从 VID 推断实体类型"""
    prefix = vid.split("_")[0]
    return PREFIX_TO_TYPE.get(prefix, "unknown")


def vid_to_id(vid: str) -> int | str:
    """从 VID 提取 PG 原始 ID"""
    parts = vid.split("_", 1)
    return parts[1] if len(parts) > 1 else vid
