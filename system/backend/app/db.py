"""USN Monitor — 数据库连接与模型"""
from __future__ import annotations

import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from geoalchemy2 import WKTElement
import json

_raw_url = os.getenv("DATABASE_URL", "postgresql://usn:usn_monitor_2026@localhost:15432/usn_monitor")
# 将 postgresql:// 替换为 asyncpg 驱动
DATABASE_URL = _raw_url.replace("postgresql://", "postgresql+asyncpg://").replace("postgresql+psycopg2://", "postgresql+asyncpg://")

engine = create_async_engine(DATABASE_URL, pool_size=10, max_overflow=5)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


# ---- Dependency ----
async def get_db():
    async with async_session() as session:
        yield session
