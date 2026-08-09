# CascadeRuleLLM: Learning to Route Rule–LLM Cascades

Experiment code and data for the paper "Learning to Route Rule–LLM Cascades for Knowledge Graph Entity Resolution". This `release/` directory is the authoritative paper artifact under `experiments/`.

## Repository Structure

```
release/
├── code/                    # Canonical experiment scripts
│   ├── l1_scorer.py                # Layer 1 signal scorer (bidirectional rule score)
│   ├── run_experiments.py          # Core pipeline: threshold selection + routing ablation
│   ├── run_glm5turbo_l2.py         # MINEC L2 verdict runner (GLM-5-Turbo, primary model)
│   ├── fragment_ablation_v2.py     # Fragment ablation: 4 models × 290 paired comparisons
│   ├── compute_l1only_turbo.py     # L1-only utility baseline (Turbo checkpoint)
│   ├── compute_pure_l2.py          # Pure L2 ceiling (all pairs through L2)
│   ├── compute_oof_probabilities.py # Out-of-fold probabilities for proper per-fold CV
│   ├── select_l1_acc_max_points.py # Table 4: max-L1-accuracy operating points
│   ├── final_operating_points.py   # Final threshold operating points
│   ├── verify_pipeline_vs_pure_l2.py # Pipeline vs pure L2 verification
│   ├── extract_per_type_weights.py # Per-type LR coefficient extraction
│   ├── generate_threshold_sensitivity.py  # Dual-threshold sensitivity figure
│   ├── generate_jaccard_sensitivity.py    # Jaccard threshold sensitivity figure
│   ├── generate_pareto_frontier.py        # Pareto frontier figure
│   └── config.py                   # Shared API keys and endpoints
│
├── data/                    # Datasets
│   ├── dataset_v3_cleaned.json     # 2,639 candidate pairs with props (MINEC)
│   ├── entity_fragments.json       # Source text fragments for L2 grounding
│   ├── minec_ground_truth_v2.json  # Corrected ground truth (manual review merged)
│   └── minec_human_annotation_300.json  # Independent human annotation set (300 pairs)
│
├── results/                 # Canonical experiment outputs
│   ├── checkpoints_v3/             # L2 model verdict checkpoints
│   │   ├── glm-5-turbo_checkpoint.json   # Primary L2 (90.7% strict)
│   │   ├── glm-4.5-air_checkpoint.json
│   │   ├── glm-4.5_checkpoint.json
│   │   ├── glm-5.2_checkpoint.json        # Annotator model (not evaluated)
│   │   └── qwen3.6_checkpoint.json
│   ├── scored_pairs.json           # 2,591 pairs with L1 scores + features
│   ├── pipeline_accuracy_empirical.json  # Fixed cascade pipeline accuracy
│   ├── threshold_selection_experiments.json  # Main routing ablation results
│   ├── dual_threshold_sweep.json   # 300+ threshold configurations sweep
│   ├── l1_only_empirical_turbo.json        # L1-only utility baseline
│   ├── classifier_ablation.json    # LR/RF standalone accuracy
│   ├── final_operating_points.json # Pipeline-aware operating points
│   ├── l1_acc_max_operating_points.json    # Max-L1-accuracy operating points
│   ├── per_type_accuracy.json      # Per-entity-type L2 accuracy
│   ├── per_type_lr_coefficients.json # d=42 LR coefficients per type
│   ├── fragment_ablation_v2_results.json   # Fragment ablation summary
│   ├── fragment_ablation_v2_checkpoint.json # Fragment ablation raw verdicts
│   ├── pipeline_vs_pure_l2.json    # Pipeline vs pure L2 detailed comparison
│   ├── pareto_frontier_data.json   # Pareto frontier data
│   ├── oof_probabilities.npz       # Out-of-fold prediction matrix
│   ├── feedback_convergence.json   # Feedback convergence simulation
│   ├── glm52_canonical_annotation.json     # GLM-5.2 labels
│   └── claude_opus_canonical_annotation.json # Claude Opus labels
│
├── dbp15k/                  # DBP15K cross-domain validation
│   ├── run_dbp15k_glm5turbo.py     # DBP15K L2 runner (GLM-5-Turbo)
│   ├── data/                       # ZH-EN labels + gold alignment
│   └── results/                    # L1/L2/pipeline results
│
├── models/                  # Trained LR classifiers (5-fold CV)
│
├── verification/            # Regression tests
│   └── test_cascaderule_llm.py
│
└── system/                  # MINEC system (open-source subset)
    ├── docker-compose.yml
    ├── nebula_schema.ngql
    ├── postgres_schema.sql
    └── backend/             # FastAPI + Nebula + PostgreSQL
```

## Key Results (GLM-5-Turbo as primary L2)

| Metric | Value |
|--------|-------|
| L1 accuracy (fixed weights) | 79.3% |
| L1 interception rate | 61.4% |
| L2 accuracy (GLM-5-Turbo, strict) | 90.7% (1,443/1,591) |
| L2 accuracy range (4 models) | 83.5%–90.7% |
| Fixed cascade pipeline (strict) | 86.3% |
| Unified pipeline-aware | 91.5% |
| Hybrid pipeline-aware | 90.6% |
| Unified L1-only utility | 81.5% |
| Hybrid L1-only utility | 82.0% |
| Fragment ablation avg delta | +4.6 pp (paired, n=290) |
| DBP15K L2 accuracy (GLM-5-Turbo) | 95.4% (4,843 valid pairs) |
| Human annotation subset (300 pairs) | 95.7% (GLM-5-Turbo) / 85.3% (pipeline) |

## Fixed L1 Parameters

| Parameter | Value |
|-----------|-------|
| τ (Jaccard field match) | 0.65 |
| θ_merge (routing) | 0.6 |
| θ_reject (routing) | -0.4 |
| λ (conflict penalty) | 1.5 |

## Reproduction

```bash
cd code/

# MINEC: L2 verdicts (requires GLM API key in config.py or .env)
python3 run_glm5turbo_l2.py

# MINEC: threshold selection + routing ablation (uses L2 checkpoint)
python3 run_experiments.py

# MINEC: L1-only utility baseline
python3 compute_l1only_turbo.py

# MINEC: pure L2 ceiling
python3 compute_pure_l2.py

# MINEC: OOF probabilities for proper per-fold CV
python3 compute_oof_probabilities.py

# MINEC: fragment ablation (4 models × 290 pairs)
python3 fragment_ablation_v2.py

# Paper figures
python3 generate_threshold_sensitivity.py
python3 generate_jaccard_sensitivity.py
python3 generate_pareto_frontier.py
```

## DBP15K Validation

```bash
cd dbp15k/
python3 run_dbp15k_glm5turbo.py
```

## Annotation Pipeline

MINEC labels are produced by a multi-stage annotation pipeline:
1. **GLM-5.2** provides provisional same/different labels (outside the evaluated cascade)
2. **Claude Opus 4.8** surfaces unresolved cases and disagreements for manual review
3. **Human adjudication** resolves the review queue (336 cases)

An independent human annotation set of 300 stratified-random pairs (`data/minec_human_annotation_300.json`) validates label quality: GLM-5-Turbo achieves 95.7% agreement with human labels, consistent with full-dataset results.

## Citation

```bibtex
@article{cascaderule2026,
  title={Learning to Route Rule--LLM Cascades for Knowledge Graph Entity Resolution},
  author={Li, Shaoyong and Xin, Jiang and Feng, Yao},
  journal={Journal of Intelligent Information Systems (under review)},
  year={2026}
}
```
