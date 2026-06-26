# CascadeRule-LLM: Code, Dataset & System

Code, data, and system implementation to reproduce all experiments in:

> **CascadeRule-LLM: Hierarchical Adaptive Weight Learning for Entity Resolution in Knowledge Graphs**

## Dataset: MINEC (Military Intelligence News Entity Corpus)

- **1,007** military intelligence news articles (Sep 2024 – Jun 2026)
- **6,601** graph entities across 5 types: Equipment, Event, Location, Person, Organization
- **2,639** candidate entity pairs with 20-dimensional signal features
- Expert labels by GLM-5.2 (99.8% validity: 2,634 valid)
- Cross-validated: GPT-5.4 Cohen's κ = 0.788; human 3-way accuracy 88.2% (zero false positives)

## System Implementation

The `system/` directory contains the pipeline that produces the MINEC corpus: news scraping, LLM-based entity extraction, and the cascade fusion system (Layer 1 rule scoring + Layer 2 LLM judgment).

**Includes:**
- `docker-compose.yml` — PostgreSQL (PostGIS) + Nebula Graph + FastAPI API
- `postgres_schema.sql` — 10 tables: equipment, persons, locations, activities, entity_relations, equipment_positions, news_sources, articles, entity_mentions, pending_entities
- `postgres_data.sql` — Full data dump (5.4MB): all entities, relations, and article metadata
- `nebula_verify.py` — Nebula Graph schema creation (5 entity tags + 5 edge types + trace space) and PG→Nebula migration
- `nebula_schema.ngql` — Current Nebula schema DDL (8 tags + 8 edge types + indexes)
- `nebula_data.json` — Full graph data export (8,250 nodes + 15,873 edges, 4.9MB)
- `backend/app/` — Core backend (FastAPI):
  - `services/extractor.py` — LLM-based entity extraction pipeline (schema, prompts, Nebula write)
  - `services/fusion.py` — Layer 1 rule scoring + Layer 2 LLM judgment + graph writeback
  - `services/dedup.py` — Candidate pair generation (fuzzy matching)
  - `services/scraper.py` — RSS/news source scraping
  - `services/seed.py` — Graph seed data
  - `routers/admin.py` — Admin API (pending entities, fusion triggers)
  - `routers/entities.py` — Entity CRUD and graph queries
  - `routers/map_data.py` — Geospatial API for map visualization
  - `nebula_service.py` — Nebula Graph connection pool and query helpers
- `frontend_dist/` — Web UI (dashboard map, entity browser, article viewer, pending queue)
  - Navigation: dashboard, articles, knowledge graph, pending queue
  - JS modules: app, map, data, detail, filters, render, state, knowledge, pending, interactions

**Excluded** (proprietary, not part of the entity resolution pipeline):
- `chat_service.py` / `chat_router.py` — intelligent Q&A module
- `research_v2.py` / `research_router.py` — networked person investigation module

### Quick Start

```bash
cd system/
cp .env.example .env  # set POSTGRES_PASSWORD, LLM_API_KEY, etc.
docker compose up -d  # starts PostgreSQL + Nebula + API

# Initialize Nebula schema + migrate data from PG
docker exec usn-api python3 /app/nebula_verify.py  # or run scripts/nebula_verify.py

# Import data
docker exec usn-db psql -U usn -d usn_monitor < postgres_data.sql

# Trigger scrape + extract cycle
curl -X POST http://localhost:8100/api/admin/scrape-extract
```

## Structure

```
├── code/
│   ├── l1_scorer.py                    # L1 bidirectional rule scorer (20 signals, λ=1.5)
│   ├── config.py                       # API endpoints, model names
│   ├── l2_multimodel_experiment.py     # L2 comparison: GLM-5/4.5/4.5-Air, Qwen3.6
│   ├── feedback_loop_experiment.py     # 5-round feedback convergence
│   ├── classifier_ablation_lr_rf.py   # LR vs RF 5-fold CV (Table X)
│   ├── per_type_accuracy.py            # Per-type accuracy: 5 types × 4 models
│   ├── per_type_threshold_analysis.py  # Per-type threshold sweep
│   ├── per_type_lr_cv.py              # Per-type LR with strict 5-fold CV
│   ├── hybrid_strategy.py             # Hybrid per-type + fallback (Table XIII)
│   ├── per_type_feature_importance.py # Per-type top features (Table XIV)
│   └── cross_model_validation.py      # GPT-5.4 κ annotation
├── data/
│   ├── dataset_v3_cleaned.json         # 2,639 pairs: signals + L1 scores + L1 decisions
│   └── entity_fragments.json           # Source text fragments for L2 grounding
└── results/
    ├── l1_results_all.json             # L1 scores + decisions for all pairs
    ├── l2_v3_comparison.json           # L2 multi-model summary
    ├── compiled_results.json           # Cross-table compiled metrics
    ├── feedback_loop_experiment.json   # 5-round convergence data (Table XI, Fig 4)
    ├── classifier_ablation.json        # LR vs RF results (Table X)
    ├── hierarchy_comparison.json       # 3-level hierarchy + LR weights (Table XIII)
    ├── per_type_feature_importance.json# Per-type top-3 weights (Table XIV)
    ├── per_type_accuracy_by_method.json# 5 types × 4 methods (Figure 5)
    ├── gpt54_annotation_300.json       # GPT-5.4 cross-model annotation (κ = 0.788)
    └── checkpoints_v3/                 # Per-model L2 outputs
        ├── glm-5_checkpoint.json       # 2,639 verdicts
        ├── glm-4.5_checkpoint.json
        ├── glm-4.5-air_checkpoint.json
        ├── glm-5.2_checkpoint.json     # Expert annotator labels
        └── qwen3.6_checkpoint.json
```

## Paper Table → Data Mapping

| Paper | Description | Data File |
|-------|-------------|-----------|
| Table I | Signal weights & dimensions | `code/l1_scorer.py` |
| Table II | Entity distribution | `data/dataset_v3_cleaned.json` |
| Table III–V | L1 distribution & accuracy | `results/l1_results_all.json` |
| Table VI | L2 multi-model comparison | `results/l2_v3_comparison.json` + `checkpoints_v3/` |
| Table VII | Fragment ablation | `checkpoints_v3/` + `data/entity_fragments.json` |
| Table VIII | Per-type L2 accuracy | `checkpoints_v3/` |
| Table IX | Pipeline accuracy | `results/compiled_results.json` |
| Table X | LR vs RF ablation | `results/classifier_ablation.json` |
| Table XI | Feedback convergence | `results/feedback_loop_experiment.json` |
| Table XII | Top-5 feature weights | `results/feedback_loop_experiment.json` |
| Table XIII | Hierarchy comparison | `results/hierarchy_comparison.json` |
| Table XIV | Per-type feature weights | `results/per_type_feature_importance.json` |
| Figure 3 | Type difficulty (bar chart) | `checkpoints_v3/` |
| Figure 4 | Feedback convergence (line) | `results/feedback_loop_experiment.json` |
| Figure 5 | Per-type × method (line) | `results/per_type_accuracy_by_method.json` |
| κ validation | GPT-5.4 cross-model | `results/gpt54_annotation_300.json` |

## Reproduction

```bash
# All scripts use paths relative to this directory (release/ as CWD).

# 1. L1 scoring (produces l1_results_all.json)
python code/l1_scorer.py

# 2. L2 multi-model experiment (requires API access to GLM models)
python code/l2_multimodel_experiment.py

# 3. Feedback loop convergence (Table XI, Figure 4)
python code/feedback_loop_experiment.py

# 4. Classifier ablation: LR vs RF (Table X)
python code/classifier_ablation_lr_rf.py

# 5. Per-type analysis & hybrid strategy (Tables XIII–XIV, Figure 5)
python code/per_type_accuracy.py
python code/hybrid_strategy.py
python code/per_type_feature_importance.py

# 6. Cross-model validation (requires GPT-5.4 API)
python code/cross_model_validation.py
```

## Requirements

- Python 3.11+
- scikit-learn, numpy, pandas
- GLM API access (Zhipu AI) or OpenAI-compatible endpoint
- Optional: Qwen3.6 local vLLM server for L2 evaluation

## License

- **Code**: MIT License
- **Dataset**: CC BY 4.0
