--
-- PostgreSQL database dump
--

-- Dumped from database version 16.14 (Debian 16.14-1.pgdg11+1)
-- Dumped by pg_dump version 16.4 (Debian 16.4-1.pgdg110+2)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: tiger; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA tiger;


--
-- Name: tiger_data; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA tiger_data;


--
-- Name: topology; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA topology;


--
-- Name: SCHEMA topology; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON SCHEMA topology IS 'PostGIS Topology schema';


--
-- Name: btree_gin; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS btree_gin WITH SCHEMA public;


--
-- Name: EXTENSION btree_gin; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION btree_gin IS 'support for indexing common datatypes in GIN';


--
-- Name: fuzzystrmatch; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS fuzzystrmatch WITH SCHEMA public;


--
-- Name: EXTENSION fuzzystrmatch; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION fuzzystrmatch IS 'determine similarities and distance between strings';


--
-- Name: postgis; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public;


--
-- Name: EXTENSION postgis; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION postgis IS 'PostGIS geometry and geography spatial types and functions';


--
-- Name: postgis_tiger_geocoder; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS postgis_tiger_geocoder WITH SCHEMA tiger;


--
-- Name: EXTENSION postgis_tiger_geocoder; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION postgis_tiger_geocoder IS 'PostGIS tiger geocoder and reverse geocoder';


--
-- Name: postgis_topology; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS postgis_topology WITH SCHEMA topology;


--
-- Name: EXTENSION postgis_topology; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION postgis_topology IS 'PostGIS topology spatial types and functions';


--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: articles; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.articles (
    id integer NOT NULL,
    source_id integer,
    title character varying(500) NOT NULL,
    url text NOT NULL,
    content text,
    published_at timestamp with time zone,
    fetched_at timestamp with time zone DEFAULT now(),
    entities_extracted boolean DEFAULT false,
    processing_status character varying(20) DEFAULT 'pending'::character varying,
    raw_metadata jsonb,
    created_at timestamp with time zone DEFAULT now(),
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, COALESCE(content, ''::text))) STORED,
    embedding public.vector(1024)
);


--
-- Name: articles_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.articles_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: articles_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.articles_id_seq OWNED BY public.articles.id;


--
-- Name: equipment_positions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.equipment_positions (
    id integer NOT NULL,
    location public.geometry(Point,4326) NOT NULL,
    reported_at timestamp with time zone NOT NULL,
    source_article_id integer,
    source_type character varying,
    notes text,
    created_at timestamp with time zone DEFAULT now(),
    equipment_vid character varying(64)
);


--
-- Name: equipment_positions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.equipment_positions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: equipment_positions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.equipment_positions_id_seq OWNED BY public.equipment_positions.id;


--
-- Name: extracted_information_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.extracted_information_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: l1_models; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.l1_models (
    id integer NOT NULL,
    entity_type character varying(32) NOT NULL,
    model_path character varying(256) NOT NULL,
    field_names jsonb NOT NULL,
    merge_threshold double precision NOT NULL,
    reject_threshold double precision NOT NULL,
    n_samples integer NOT NULL,
    cv_accuracy double precision,
    trained_at timestamp without time zone DEFAULT now()
);


--
-- Name: l1_models_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.l1_models_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: l1_models_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.l1_models_id_seq OWNED BY public.l1_models.id;


--
-- Name: news_sources; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.news_sources (
    id integer NOT NULL,
    name character varying(200) NOT NULL,
    base_url text NOT NULL,
    source_type character varying(30),
    scrape_config jsonb,
    last_fetched_at timestamp with time zone,
    fetch_interval_minutes integer DEFAULT 360,
    active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: news_sources_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.news_sources_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: news_sources_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.news_sources_id_seq OWNED BY public.news_sources.id;


--
-- Name: pending_entities; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.pending_entities (
    id integer NOT NULL,
    article_id integer,
    entity_type character varying(32),
    raw_data jsonb,
    context text,
    confidence double precision DEFAULT 0,
    candidates jsonb,
    llm_verdict text,
    status character varying(16) DEFAULT 'pending'::character varying,
    created_at timestamp without time zone DEFAULT now(),
    resolved_at timestamp without time zone,
    resolved_vid character varying(64),
    notes text,
    vid_a character varying(128),
    vid_b character varying(128),
    name_a character varying,
    name_b character varying,
    similarity double precision DEFAULT 0,
    resolved_by character varying
);


--
-- Name: pending_entities_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.pending_entities_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: pending_entities_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.pending_entities_id_seq OWNED BY public.pending_entities.id;


--
-- Name: person_profile; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.person_profile (
    id integer NOT NULL,
    target_name text NOT NULL,
    aliases jsonb DEFAULT '[]'::jsonb,
    occupation text,
    org_name text,
    region text,
    education text,
    description text,
    custom_fields jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: person_profile_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.person_profile_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: person_profile_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.person_profile_id_seq OWNED BY public.person_profile.id;


--
-- Name: person_profiles; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.person_profiles (
    id integer NOT NULL,
    name text NOT NULL,
    aliases jsonb DEFAULT '[]'::jsonb,
    nationality text,
    date_of_birth text,
    place_of_birth text,
    organization text,
    "position" text,
    education text,
    region text,
    description text,
    custom_fields jsonb DEFAULT '{}'::jsonb,
    search_count integer DEFAULT 0,
    last_search_at timestamp with time zone,
    status text DEFAULT 'active'::text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: person_profiles_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.person_profiles_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: person_profiles_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.person_profiles_id_seq OWNED BY public.person_profiles.id;


--
-- Name: person_research_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.person_research_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: research_results_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.research_results_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: research_sessions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.research_sessions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: session_result_map_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.session_result_map_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: articles id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.articles ALTER COLUMN id SET DEFAULT nextval('public.articles_id_seq'::regclass);


--
-- Name: equipment_positions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.equipment_positions ALTER COLUMN id SET DEFAULT nextval('public.equipment_positions_id_seq'::regclass);


--
-- Name: l1_models id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.l1_models ALTER COLUMN id SET DEFAULT nextval('public.l1_models_id_seq'::regclass);


--
-- Name: news_sources id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.news_sources ALTER COLUMN id SET DEFAULT nextval('public.news_sources_id_seq'::regclass);


--
-- Name: pending_entities id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pending_entities ALTER COLUMN id SET DEFAULT nextval('public.pending_entities_id_seq'::regclass);


--
-- Name: person_profile id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.person_profile ALTER COLUMN id SET DEFAULT nextval('public.person_profile_id_seq'::regclass);


--
-- Name: person_profiles id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.person_profiles ALTER COLUMN id SET DEFAULT nextval('public.person_profiles_id_seq'::regclass);


--
-- Name: articles articles_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.articles
    ADD CONSTRAINT articles_pkey PRIMARY KEY (id);


--
-- Name: articles articles_url_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.articles
    ADD CONSTRAINT articles_url_key UNIQUE (url);


--
-- Name: equipment_positions equipment_positions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.equipment_positions
    ADD CONSTRAINT equipment_positions_pkey PRIMARY KEY (id);


--
-- Name: l1_models l1_models_entity_type_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.l1_models
    ADD CONSTRAINT l1_models_entity_type_key UNIQUE (entity_type);


--
-- Name: l1_models l1_models_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.l1_models
    ADD CONSTRAINT l1_models_pkey PRIMARY KEY (id);


--
-- Name: news_sources news_sources_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.news_sources
    ADD CONSTRAINT news_sources_pkey PRIMARY KEY (id);


--
-- Name: pending_entities pending_entities_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pending_entities
    ADD CONSTRAINT pending_entities_pkey PRIMARY KEY (id);


--
-- Name: person_profile person_profile_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.person_profile
    ADD CONSTRAINT person_profile_pkey PRIMARY KEY (id);


--
-- Name: person_profiles person_profiles_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.person_profiles
    ADD CONSTRAINT person_profiles_pkey PRIMARY KEY (id);


--
-- Name: idx_articles_content_fts; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_articles_content_fts ON public.articles USING gin (to_tsvector('english'::regconfig, COALESCE(content, ''::text)));


--
-- Name: idx_articles_content_tsv; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_articles_content_tsv ON public.articles USING gin (content_tsv);


--
-- Name: idx_articles_published; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_articles_published ON public.articles USING btree (published_at DESC);


--
-- Name: idx_articles_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_articles_status ON public.articles USING btree (processing_status);


--
-- Name: idx_ep_equipment_vid; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_ep_equipment_vid ON public.equipment_positions USING btree (equipment_vid);


--
-- Name: idx_ep_reported_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_ep_reported_at ON public.equipment_positions USING btree (reported_at DESC);


--
-- Name: idx_equip_positions_geom; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_equip_positions_geom ON public.equipment_positions USING gist (location);


--
-- Name: idx_pending_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pending_status ON public.pending_entities USING btree (status);


--
-- Name: idx_pending_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pending_type ON public.pending_entities USING btree (entity_type);


--
-- Name: idx_person_profile_name; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_person_profile_name ON public.person_profile USING btree (target_name);


--
-- Name: idx_pp_name; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_pp_name ON public.person_profiles USING btree (name);


--
-- Name: idx_pp_org; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pp_org ON public.person_profiles USING btree (organization);


--
-- Name: idx_pp_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pp_status ON public.person_profiles USING btree (status);


--
-- Name: articles articles_source_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.articles
    ADD CONSTRAINT articles_source_id_fkey FOREIGN KEY (source_id) REFERENCES public.news_sources(id);


--
-- Name: pending_entities pending_entities_article_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pending_entities
    ADD CONSTRAINT pending_entities_article_id_fkey FOREIGN KEY (article_id) REFERENCES public.articles(id);


--
-- PostgreSQL database dump complete
--

