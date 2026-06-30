"""USN Monitor — FastAPI 主入口"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os, logging, asyncio

from app.db import engine, Base, async_session
from app.routers import entities, map_data, admin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("usn")

# 全局任务状态
_task_state = {
    "running": False,
    "phase": "",        # "scraping" | "extracting" | "done" | "failed"
    "message": "",
    "started_at": None,
    "scrape_new": 0,
    "extract_done": 0,
    "log": [],          # 最近日志条目
}


async def run_scrape_and_extract():
    """采集 + 抽取（统一入口）"""
    if _task_state["running"]:
        log.warning("[Task] Already running, skipping")
        return
    _task_state.update(running=True, phase="scraping", message="开始采集...",
                       started_at=asyncio.get_event_loop().time(),
                       scrape_new=0, extract_done=0, log=[])
    _add_log("🚀 任务开始")

    try:
        async with async_session() as db:
            from app.services.scraper import scrape_all_sources
            from app.services.extractor import extract_pending_articles

            # Phase 1: 采集
            _task_state["phase"] = "scraping"
            _task_state["message"] = "正在从新闻源采集..."
            _add_log("📡 开始采集新闻源")
            count = await scrape_all_sources(db)
            _task_state["scrape_new"] = count
            _add_log(f"📰 采集完成: {count} 篇新文章")

            # Phase 2: 抽取
            _task_state["phase"] = "extracting"
            _task_state["message"] = "正在用 AI 提取实体..."
            _add_log("🤖 开始 AI 实体抽取")
            processed = await extract_pending_articles(db)
            _task_state["extract_done"] = processed
            _add_log(f"✅ 抽取完成: {processed} 篇文章已处理")

            _task_state["phase"] = "done"
            _task_state["message"] = f"完成: 采集 {count} 篇, 处理 {processed} 篇"
            _add_log(f"🏁 任务完成: 采集{count}篇, 处理{processed}篇")

    except Exception as e:
        log.error(f"[Task] Error: {e}", exc_info=True)
        _task_state["phase"] = "failed"
        _task_state["message"] = f"失败: {e}"
        _add_log(f"❌ 失败: {e}")
    finally:
        _task_state["running"] = False


def _add_log(msg: str):
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    _task_state["log"].append(f"[{ts}] {msg}")
    # 只保留最近 50 条
    if len(_task_state["log"]) > 50:
        _task_state["log"] = _task_state["log"][-50:]
    log.info(msg)


async def scheduled_scrape_and_extract():
    """定时采集 + 抽取"""
    log.info("[Scheduler] 定时任务触发")
    await run_scrape_and_extract()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    async with engine.begin() as conn:
        pass  # 表由 init.sql 创建

    # APScheduler 定时任务
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        scheduled_scrape_and_extract,
        "interval", hours=4, id="scrape_extract",
        replace_existing=True
    )
    scheduler.start()
    log.info("[Scheduler] Started — scrape every 4 hours")
    yield
    # Shutdown
    scheduler.shutdown(wait=False)
    await engine.dispose()


app = FastAPI(title="USN Monitor", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(entities.router, prefix="/api", tags=["entities"])
app.include_router(map_data.router, prefix="/api/map", tags=["map"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])

# 静态前端
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend_dist")

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import FileResponse as _FR

class FrontendMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if response.status_code == 404 and not request.url.path.startswith("/api") and not request.url.path.startswith("/docs") and not request.url.path.startswith("/openapi"):
            fp = os.path.join(frontend_path, request.url.path.lstrip("/"))
            if request.url.path != "/" and os.path.isfile(fp):
                resp = _FR(fp)
                # JS/CSS/HTML 不缓存，确保更新即时生效
                if fp.endswith(('.js', '.css', '.html')):
                    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                    resp.headers["Pragma"] = "no-cache"
                return resp
            idx = os.path.join(frontend_path, "index.html")
            if os.path.isfile(idx):
                return _FR(idx)
        return response

if os.path.isdir(frontend_path):
    app.add_middleware(FrontendMiddleware)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
