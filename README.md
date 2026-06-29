# CascadeRule-LLM: Cascade Entity Resolution with LLM Fallback

Experiment code and data for the paper "CascadeRule-LLM: A Two-Layer Cascade Architecture for Entity Resolution in Knowledge Graphs".

## Repository Structure

```
release/
├── code/                    # Experiment scripts
│   ├── l1_scorer.py         # Layer 1 bidirectional rule scorer (τ=0.65, merge=0.6, reject=-0.4)
│   ├── run_experiments.py   # Full experiment pipeline (L1→L2→hierarchy→feedback→per-type)
│   ├── sweep_dual_threshold.py   # 99-config merge/reject threshold sweep
│   ├── sweep_jaccard_fine.py     # 19-config Jaccard τ fine sweep (0.05 intervals)
│   ├── sweep_weak_field.py       # 20-config weak-field parameter sweep
│   ├── gen_threshold_figure.py   # Generate threshold_sensitivity.pdf
│   ├── gen_jaccard_figure.py     # Generate jaccard_sensitivity.pdf
│   ├── 26_gpt54_annotation.py    # GPT-5.4 cross-model validation on 300 sampled pairs
│   ├── 28_human_annotation_analysis.py # Human adjudication analysis for disagreement cases
│   ├── 29_three_way_analysis.py  # Three-way GLM/GPT/human comparison utilities
│   └── config.py            # Shared paths and constants
│
├── data/                    # Datasets
│   ├── dataset_v3_cleaned.json   # 2,639 labeled entity pairs (MINEC)
│   └── entity_fragments.json     # Source text fragments for L2 grounding
│
├── results/                 # Experiment outputs (τ=0.65)
│   ├── l1_distribution.json      # Table III: merge/escalate/reject distribution
│   ├── l1_accuracy.json          # Table IV: L1 accuracy + precision
│   ├── expert_bucket.json        # Table V: expert labels per L1 bucket
│   ├── pipeline_accuracy.json    # Table VIII: L1+L2 pipeline accuracy
│   ├── classifier_ablation.json  # Table IX: LR vs RF 5-fold CV
│   ├── feedback_convergence.json # Table X: feedback rounds
│   ├── hierarchy_comparison.json # Table XI: Fixed/Unified/Hybrid comparison
│   ├── per_type_accuracy.json    # Table XII: per-type accuracy
│   ├── feature_importance.json   # Learned LR feature weights
│   ├── human_annotation_analysis.json  # 59 disagreement cases, 51 comparable, GLM-5.2=88.2%
│   ├── gpt54_annotation_300.json       # Cross-model validation sample annotations
│   ├── gpt54_annotation_300_checkpoint.json # GPT-5.4 raw checkpoint for reproducibility
│   ├── scored_pairs.json         # All 2,639 pairs with L1 scores
│   ├── dual_threshold_sweep.json # 99-config heatmap data
│   ├── jaccard_fine_sweep.json   # 19-config τ sweep data
│   ├── jaccard_extended_sweep.json
│   ├── sweep_results.json        # 20-config weak-field sweep
│   └── checkpoints_v3/           # L2 model verdict checkpoints
│       ├── glm-5_checkpoint.json
│       ├── glm-4.5-air_checkpoint.json
│       ├── glm-4.5_checkpoint.json
│       ├── glm-5.2_checkpoint.json
│       └── qwen3.6_checkpoint.json
│
├── dbp15k/                  # DBP15K cross-domain validation
│   ├── code/                # 4 scripts used for the compact release pipeline
│   ├── run_dbp15k_hierarchy.py   # Legacy/full hierarchy experiment script
│   ├── run_dbp15k_l2_resilient.py # Earlier resilient L2 runner
│   ├── run_dbp15k_l2_expanded.py  # Expanded hard-case L2 evaluation runner
│   ├── threshold_sensitivity.py   # Earlier threshold sweep script
│   ├── data/                # ZH-EN labels + gold alignment
│   └── results/             # L1 fixed/adaptive, L2 GLM-5.2, pipeline
│
└── system/                  # MINEC system (open-source subset)
    ├── docker-compose.yml
    ├── Dockerfile
    ├── nebula_schema.ngql
    ├── postgres_schema.sql
    ├── backend/             # FastAPI + Nebula + PostgreSQL
    └── frontend_dist/       # Static frontend
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| τ (Jaccard field match) | 0.65 | Weak-field overlap threshold for match/conflict |
| θ_merge (routing) | 0.6 | L1 score ≥ this → auto-merge |
| θ_reject (routing) | -0.4 | L1 score ≤ this → auto-reject |
| λ (conflict penalty) | 1.5 | Conflict fields weighted 1.5× |
| Scoring range | [-1, +1] | Bidirectional: match adds, conflict subtracts |

## Key Results

| Metric | Value |
|--------|-------|
| L1 accuracy (fixed weights) | 78.3% |
| L1 interception rate | 38.7% |
| Merge precision | 75.6% |
| Reject precision | 80.4% |
| Pipeline accuracy (L1+GLM-5) | 89.3% (empirical) / 89.8% (formula) |
| L2 accuracy (GLM-5) | 97.1% |
| Unified LR accuracy (5-fold CV) | 81.1% |
| Hybrid LR accuracy (5-fold CV) | 81.3% |
| LLM cost reduction (adaptive) | 82% |
| Human adjudication set | 59 disagreements / 51 comparable |
| GLM-5.2 vs human (disagreements only) | 88.2% |
| DBP15K fixed L1 (paper-reported) | 77.9% acc / 79.5% intercept / 81.1% pipeline / 20.5% LLM cost |
| DBP15K adaptive L1 (paper-reported) | 87.7% acc / 91.3% intercept / 88.2% pipeline / 8.7% LLM cost |
| DBP15K GLM-5.2 hard-case accuracy | 92.8% on 499 valid / 500 sampled |

## Data Integrity

The two primary files under `data/` are byte-identical to the originals in `experiments/data/`:

- `dataset_v3_cleaned.json`
- `entity_fragments.json`

See `PROVENANCE.md` for SHA256 checksums and for the distinction between raw DBP15K outputs and final paper-reported summaries.

## Reproduction

```bash
cd code/

# 1. Run full experiment pipeline
python3 run_experiments.py

# 2. Generate threshold sensitivity figures
python3 gen_threshold_figure.py
python3 gen_jaccard_figure.py

# 3. Parameter sweeps (optional)
python3 sweep_dual_threshold.py    # 99 configs
python3 sweep_jaccard_fine.py      # 19 configs
python3 sweep_weak_field.py        # 20 configs
```

## DBP15K Validation

```bash
cd dbp15k/code/
python3 01_l1_fixed_weight.py
python3 02_l2_glm52.py
python3 03_hierarchy_feedback.py
python3 04_threshold_sensitivity.py
```

## Citation

```bibtex
@article{cascaderule2026,
  title={CascadeRule-LLM: A Two-Layer Cascade Architecture for Entity Resolution in Knowledge Graphs},
  author={Li, Shaoyong and Xin, Jiang and Feng, Yao},
  journal={Journal of Intelligent Information Systems (under review)},
  year={2026}
}
```
