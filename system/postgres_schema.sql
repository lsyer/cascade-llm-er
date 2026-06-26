-- USN Monitor 数据库初始化
-- 由 DB 实际结构导出，2026-05-06

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- 模糊匹配，用于消歧

-- ============================================================
-- 核心实体：器装（舰船、飞机、车辆、武器等）
-- ============================================================
CREATE TABLE equipment (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    equipment_type VARCHAR NOT NULL DEFAULT 'ship',  -- ship, aircraft, vehicle, weapon, system
    designation VARCHAR,                              -- CVN-78, DDG-51 等
    category VARCHAR,                                 -- aircraft_carrier, destroyer, submarine 等
    status VARCHAR DEFAULT 'active',
    parent_unit VARCHAR,
    home_base VARCHAR,                                -- 母港名称
    home_location GEOMETRY(Point, 4326),              -- 母港坐标
    commission_date DATE,
    description TEXT,
    aliases TEXT[] DEFAULT '{}',
    source_url TEXT,
    properties JSONB DEFAULT '{}',                    -- 扩展属性
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_equipment_type ON equipment(equipment_type);
CREATE INDEX idx_equipment_category ON equipment(category);
CREATE INDEX idx_equipment_home ON equipment USING GIST(home_location);

-- ============================================================
-- 核心实体：人员
-- ============================================================
CREATE TABLE persons (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    rank VARCHAR,
    position VARCHAR,
    service_branch VARCHAR,                           -- US Navy, US Army 等
    description TEXT,
    aliases TEXT[] DEFAULT '{}',
    source_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- 核心实体：地理位置
-- ============================================================
CREATE TABLE locations (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    location GEOMETRY(Point, 4326),
    location_type VARCHAR DEFAULT 'unknown',          -- naval_station, city, strait, sea_area 等
    country VARCHAR,
    region VARCHAR,                                   -- INDOPACOM, EUCOM, CENTCOM 等
    description TEXT,
    aliases TEXT[] DEFAULT '{}',
    source_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_locations_type ON locations(location_type);
CREATE INDEX idx_locations_region ON locations(region);
CREATE INDEX idx_locations_geo ON locations USING GIST(location);

-- ============================================================
-- 核心实体：活动（演习、部署、巡逻、作战等）
-- ============================================================
CREATE TABLE activities (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    activity_type VARCHAR DEFAULT 'operation',        -- operation, exercise, deployment, patrol, strike
    start_date DATE,
    end_date DATE,
    region VARCHAR,
    area GEOMETRY(Polygon, 4326),                     -- 活动区域
    description TEXT,
    aliases TEXT[] DEFAULT '{}',
    source_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- 实体关系
-- ============================================================
CREATE TABLE entity_relations (
    id SERIAL PRIMARY KEY,
    subject_type VARCHAR NOT NULL,                    -- equipment, person, location, activity
    subject_id INTEGER NOT NULL,
    relation VARCHAR NOT NULL,                        -- commanded, deployed_to, participated_in 等
    object_type VARCHAR NOT NULL,
    object_id INTEGER NOT NULL,
    source_article_id INTEGER,
    confidence REAL DEFAULT 0.9,
    reported_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_relations_subject ON entity_relations(subject_type, subject_id);
CREATE INDEX idx_relations_object ON entity_relations(object_type, object_id);

-- ============================================================
-- 器装位置追踪（时序）
-- ============================================================
CREATE TABLE equipment_positions (
    id SERIAL PRIMARY KEY,
    equipment_id INTEGER REFERENCES equipment(id) ON DELETE CASCADE,
    location GEOMETRY(Point, 4326) NOT NULL,
    reported_at TIMESTAMPTZ NOT NULL,
    source_article_id INTEGER,
    source_type VARCHAR DEFAULT 'news',               -- news, ais, official
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_eqpos_equipment_time ON equipment_positions(equipment_id, reported_at DESC);
CREATE INDEX idx_eqpos_location ON equipment_positions USING GIST(location);

-- ============================================================
-- 新闻源
-- ============================================================
CREATE TABLE news_sources (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    base_url TEXT NOT NULL,
    source_type VARCHAR,                              -- rss, gdelt, web_scrape
    scrape_config JSONB,
    last_fetched_at TIMESTAMPTZ,
    fetch_interval_minutes INTEGER DEFAULT 240,       -- 默认4小时
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- 新闻文章
-- ============================================================
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES news_sources(id),
    title VARCHAR NOT NULL,
    url TEXT UNIQUE NOT NULL,
    content TEXT,
    published_at TIMESTAMPTZ,
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    entities_extracted BOOLEAN DEFAULT false,
    processing_status VARCHAR DEFAULT 'pending',      -- pending, processing, done, failed
    raw_metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_articles_status ON articles(processing_status);
CREATE INDEX idx_articles_published ON articles(published_at DESC);

-- ============================================================
-- 实体提及（文章↔实体关联）
-- ============================================================
CREATE TABLE entity_mentions (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    entity_type VARCHAR NOT NULL,                     -- equipment, person, location, activity
    entity_id INTEGER NOT NULL,
    mention_text VARCHAR,                             -- 原文中的提及文本
    context_snippet TEXT,                             -- 提及上下文句子
    confidence REAL DEFAULT 1.0,
    extracted_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_mentions_entity ON entity_mentions(entity_type, entity_id);
CREATE INDEX idx_mentions_article ON entity_mentions(article_id);

-- ============================================================
-- 融合消歧暂存队列
-- ============================================================
CREATE TABLE IF NOT EXISTS pending_entities (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(32),       -- equipment/person/location/activity
    vid_a VARCHAR(128),            -- 实体A的 VID
    vid_b VARCHAR(128),            -- 实体B的 VID
    name_a VARCHAR,                -- 实体A名称
    name_b VARCHAR,                -- 实体B名称
    similarity FLOAT DEFAULT 0,    -- 相似度分数
    raw_data JSONB,                -- 两个实体的属性快照
    llm_verdict TEXT,              -- LLM 判断结果
    status VARCHAR(16) DEFAULT 'pending',  -- pending/merged/discarded/disambig
    resolved_at TIMESTAMPTZ,
    resolved_vid VARCHAR(128),     -- 最终合并到哪个节点
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pending_status ON pending_entities(status);
CREATE INDEX IF NOT EXISTS idx_pending_type ON pending_entities(entity_type);


