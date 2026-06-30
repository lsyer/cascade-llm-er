"""新闻采集器 — 从公开源抓取美海军相关新闻"""
from __future__ import annotations

import httpx
import feedparser
import json
import re
import asyncio
from bs4 import BeautifulSoup
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from datetime import datetime, timezone
import logging

log = logging.getLogger("usn.scraper")

# US Navy 相关关键词
NAVY_KEYWORDS = re.compile(
    r"\b(USS|USNS|CVN|DDG|SSN|SSBN|CG-|LHD|LHA|LPD|LSD|LCAC|"
    r"Navy|Naval|fleet|carrier|destroyer|submarine|cruiser|"
    r"amphibious|sealift|strike group|battle group|"
    r"INDOPACOM|EUCOM|CENTCOM|SOUTHCOM|AFRICOM|"
    r"7th Fleet|5th Fleet|6th Fleet|2nd Fleet|3rd Fleet|4th Fleet|10th Fleet|"
    r"Naval Station|Naval Base|Naval Air Station)\b",
    re.IGNORECASE
)

# 排除纯社区/商业/健康类文章（仅用于 GDELT 二次过滤）
GDELT_EXCLUDE = re.compile(
    r"\b(community event|mental health|therapy|partnership with|"
    r"charity|fundrais|volunteer|wellness|yoga|diversity|inclusion|"
    r"talkspace|insurance|hiring|recruitment fair)\b",
    re.IGNORECASE
)

DEFAULT_SOURCES = [
    # --- RSS 新闻源（原始） ---
    {
        "name": "USNI News",
        "base_url": "https://news.usni.org",
        "source_type": "rss",
        "scrape_config": {"feed_url": "https://news.usni.org/feed"},
    },
    {
        "name": "USNI Fleet Tracker",
        "base_url": "https://news.usni.org/category/fleet-tracker",
        "source_type": "rss",
        "scrape_config": {"feed_url": "https://news.usni.org/category/fleet-tracker/feed"},
    },
    {
        "name": "USNI USN",
        "base_url": "https://news.usni.org/category/usn",
        "source_type": "rss",
        "scrape_config": {"feed_url": "https://news.usni.org/category/usn/feed"},
    },
    {
        "name": "Naval News",
        "base_url": "https://www.navalnews.com",
        "source_type": "rss",
        "scrape_config": {"feed_url": "https://www.navalnews.com/feed/"},
    },
    {
        "name": "The War Zone",
        "base_url": "https://www.twz.com",
        "source_type": "rss",
        "scrape_config": {"feed_url": "https://www.twz.com/feed"},
    },
    {
        "name": "Defense News - Navy",
        "base_url": "https://www.defensenews.com/naval",
        "source_type": "rss",
        "scrape_config": {"feed_url": "https://www.defensenews.com/naval/atom.xml"},
    },
    {
        "name": "Navy Times",
        "base_url": "https://www.navytimes.com",
        "source_type": "rss",
        "scrape_config": {"feed_url": "https://www.navytimes.com/arc/outboundfeeds/rss/"},
    },
    # --- RSS 新闻源（新增：位置信息密集） ---
    {
        "name": "Breaking Defense",
        "base_url": "https://breakingdefense.com",
        "source_type": "rss",
        "scrape_config": {"feed_url": "https://breakingdefense.com/feed/"},
    },
    {
        "name": "Seapower Magazine",
        "base_url": "https://seapowermagazine.org",
        "source_type": "rss",
        "scrape_config": {"feed_url": "https://seapowermagazine.org/feed/"},
    },
    {
        "name": "UK Defence Journal",
        "base_url": "https://ukdefencejournal.org.uk",
        "source_type": "rss",
        "scrape_config": {"feed_url": "https://ukdefencejournal.org.uk/feed/"},
    },
    # --- API 数据源: GDELT 多主题查询 ---
    {
        "name": "GDELT-ShipDeploy",
        "base_url": "https://api.gdeltproject.org",
        "source_type": "gdelt",
        "scrape_config": {
            "query": '("USS Ford" OR "USS Reagan" OR "USS Eisenhower" OR "USS Truman" OR "USS Carl Vinson" OR "carrier strike group" OR "amphibious ready group")',
            "mode": "ArtList",
            "maxrecords": 15,
            "timespan": "7d",
        },
    },
    {
        "name": "GDELT-SeaAreas",
        "base_url": "https://api.gdeltproject.org",
        "source_type": "gdelt",
        "scrape_config": {
            "query": '("South China Sea" OR "Taiwan Strait" OR "Strait of Hormuz" OR "East China Sea" OR "Philippine Sea") (navy OR warship OR carrier)',
            "mode": "ArtList",
            "maxrecords": 15,
            "timespan": "7d",
        },
    },
    {
        "name": "GDELT-Exercises",
        "base_url": "https://api.gdeltproject.org",
        "source_type": "gdelt",
        "scrape_config": {
            "query": '("RIMPAC" OR "naval exercise" OR "Freedom of Navigation" OR "Malabar exercise" OR "war game")',
            "mode": "ArtList",
            "maxrecords": 15,
            "timespan": "14d",
        },
    },
    {
        "name": "GDELT-FleetNews",
        "base_url": "https://api.gdeltproject.org",
        "source_type": "gdelt",
        "scrape_config": {
            "query": '("US Navy" OR "US 7th Fleet" OR "US 5th Fleet" OR "US 6th Fleet") (deploy OR patrol OR operation)',
            "mode": "ArtList",
            "maxrecords": 15,
            "timespan": "7d",
        },
    },
    # --- GDELT 新增：部署位置密集型查询 ---
    {
        "name": "GDELT-DeployOps",
        "base_url": "https://api.gdeltproject.org",
        "source_type": "gdelt",
        "scrape_config": {
            "query": '(USS OR USNS) ("deployed to" OR "returned to" OR "arrived in" OR "underway from" OR "operating in")',
            "mode": "ArtList",
            "maxrecords": 15,
            "timespan": "14d",
        },
    },
    {
        "name": "GDELT-PortCalls",
        "base_url": "https://api.gdeltproject.org",
        "source_type": "gdelt",
        "scrape_config": {
            "query": '(USS OR warship OR destroyer OR carrier) ("port visit" OR "port call" OR "docked at" OR homeport OR "naval station")',
            "mode": "ArtList",
            "maxrecords": 15,
            "timespan": "14d",
        },
    },
    {
        "name": "GDELT-ShipTransit",
        "base_url": "https://api.gdeltproject.org",
        "source_type": "gdelt",
        "scrape_config": {
            "query": '(warship OR USS OR "naval vessel") (transited OR "sailed through" OR "passed through" OR "conducted operations") (Sea OR Strait OR Gulf OR Ocean)',
            "mode": "ArtList",
            "maxrecords": 15,
            "timespan": "14d",
        },
    },
]


async def ensure_sources(db: AsyncSession):
    """确保数据源已注册"""
    for src in DEFAULT_SOURCES:
        existing = await db.execute(
            text("SELECT id FROM news_sources WHERE name = :name"),
            {"name": src["name"]}
        )
        if not existing.first():
            await db.execute(text("""
                INSERT INTO news_sources (name, base_url, source_type, scrape_config)
                VALUES (:name, :base_url, :source_type, CAST(:config AS jsonb))
            """), {
                "name": src["name"],
                "base_url": src["base_url"],
                "source_type": src["source_type"],
                "config": json.dumps(src["scrape_config"]),
            })
            await db.commit()
    await db.commit()


async def scrape_all_sources(db: AsyncSession) -> int:
    """采集所有活跃数据源"""
    await ensure_sources(db)

    rows = await db.execute(text(
        "SELECT * FROM news_sources WHERE active = true"
    ))
    sources = rows.mappings().all()
    log.info(f"[Scraper] Found {len(sources)} active sources")

    total_new = 0
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=30.0,
        headers={"User-Agent": "USNMonitor/1.0 (research)"}
    ) as client:
        for source in sources:
            try:
                log.info(f"[Scraper] Fetching {source['name']} ({source['source_type']})...")
                st = source["source_type"]
                if st == "rss":
                    new = await scrape_rss(client, db, dict(source))
                elif st == "gdelt":
                    new = await scrape_gdelt(client, db, dict(source))
                elif st == "web_scrape":
                    new = await scrape_web(client, db, dict(source))
                else:
                    new = 0
                    log.warning(f"[Scraper] Unknown source type: {st}")
                total_new += new
                log.info(f"[Scraper] {source['name']}: {new} new articles")
            except Exception as e:
                log.error(f"[Scraper] Error scraping {source['name']}: {e}")
                # GDELT 限速，等一下再继续
                if "429" in str(e):
                    log.info("[Scraper] Rate limited, waiting 30s...")
                    await asyncio.sleep(30)

    await db.execute(text("""
        UPDATE news_sources SET last_fetched_at = NOW()
    """))
    await db.commit()
    log.info(f"[Scraper] Total new articles: {total_new}")
    return total_new


async def scrape_rss(client: httpx.AsyncClient, db: AsyncSession, source: dict) -> int:
    """采集 RSS 源"""
    config = source.get("scrape_config", {}) or {}
    feed_url = config.get("feed_url", source["base_url"])

    resp = await client.get(feed_url)
    feed = feedparser.parse(resp.text)

    new_count = 0
    for entry in feed.entries[:30]:
        url = entry.get("link", "")
        if not url:
            continue

        # 去重
        existing = await db.execute(
            text("SELECT id FROM articles WHERE url = :url"),
            {"url": url}
        )
        if existing.first():
            continue

        title = entry.get("title", "")
        content = ""
        if hasattr(entry, "summary"):
            content = entry.summary
        if hasattr(entry, "content"):
            for c in entry.content:
                content = c.get("value", content)

        if content:
            content = BeautifulSoup(content, "lxml").get_text(separator=" ", strip=True)

        # 关键词过滤（Fleet Tracker 类文章全收）
        is_tracker = "fleet" in source["name"].lower() or "tracker" in source["name"].lower()
        full_text = f"{title} {content}"
        if not is_tracker and not NAVY_KEYWORDS.search(full_text):
            continue

        published = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

        await db.execute(text("""
            INSERT INTO articles (source_id, title, url, content, published_at, processing_status)
            VALUES (:sid, :title, :url, :content, :pub, 'pending')
            ON CONFLICT (url) DO NOTHING
        """), {
            "sid": source["id"],
            "title": title,
            "url": url,
            "content": content[:10000],
            "pub": published,
        })
        new_count += 1

    await db.commit()
    return new_count


async def scrape_gdelt(client: httpx.AsyncClient, db: AsyncSession, source: dict) -> int:
    """采集 GDELT 全球事件数据"""
    config = source.get("scrape_config", {}) or {}
    params = {
        "query": config.get("query", '"USS" OR "US Navy"'),
        "mode": config.get("mode", "ArtList"),
        "maxrecords": config.get("maxrecords", 30),
        "format": "json",
        "timespan": config.get("timespan", "7d"),
    }

    # GDELT 限速：每次请求间隔 10 秒
    await asyncio.sleep(10)
    resp = await client.get(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        params=params
    )
    if resp.status_code == 429:
        log.warning("[GDELT] Rate limited, waiting 60s...")
        await asyncio.sleep(60)
        resp = await client.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params=params
        )
    if resp.status_code == 429:
        log.warning("[GDELT] Still rate limited, skipping")
        return 0
    if resp.status_code != 200 or not resp.text.strip().startswith("{"):
        log.warning(f"[GDELT] Unexpected response: status={resp.status_code} body={resp.text[:100]}")
        return 0

    data = resp.json()
    articles = data.get("articles", [])
    log.info(f"[GDELT] Got {len(articles)} articles")

    new_count = 0
    for art in articles:
        url = art.get("url", "")
        title = art.get("title", "")
        if not url or not title:
            continue

        # 去重
        existing = await db.execute(
            text("SELECT id FROM articles WHERE url = :url"),
            {"url": url}
        )
        if existing.first():
            continue

        # 关键词过滤
        if not NAVY_KEYWORDS.search(f"{title}"):
            continue

        # GDELT 排除：过滤社区/商业/健康等无关文章
        if GDELT_EXCLUDE.search(title):
            log.debug(f"[GDELT] Excluded: {title[:80]}")
            continue

        published = None
        pub_date = art.get("seendate", "")
        if pub_date:
            try:
                # GDELT format: 20260505T123000Z
                published = datetime.strptime(pub_date, "%Y%m%dT%H%M%Sz").replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                pass

        # 尝试抓取正文
        content = ""
        try:
            art_resp = await client.get(url, timeout=15)
            if art_resp.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(art_resp.text, "lxml")
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
        except Exception as e:
            log.debug(f"[GDELT] Failed to fetch content for {url}: {e}")

        await db.execute(text("""
            INSERT INTO articles (source_id, title, url, content, published_at, processing_status)
            VALUES (:sid, :title, :url, :content, :pub, 'pending')
            ON CONFLICT (url) DO NOTHING
        """), {
            "sid": source["id"],
            "title": title,
            "url": url,
            "content": content[:8000],
            "pub": published,
        })
        new_count += 1

    await db.commit()
    return new_count


async def scrape_web(client: httpx.AsyncClient, db: AsyncSession, source: dict) -> int:
    """采集网页源（基础版）"""
    return 0
