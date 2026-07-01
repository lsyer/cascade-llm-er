# CascadeRule-LLM: Cascade Entity Resolution with LLM Fallback

Experiment code and data for the paper "CascadeRule-LLM: A Two-Layer Cascade Architecture for Entity Resolution in Knowledge Graphs". This `release/` directory is the authoritative paper artifact under `experiments/`: figures, tables, code, and the open-source system snapshot should be interpreted from here rather than from historical exploratory scripts in sibling folders.

## Repository Structure

## Authority Boundary

- `release/` is the paper-facing, reproducible artifact and should be treated as the authoritative snapshot for experiments, results, and the open-source system subset.
- `legacy_workspace/` contains historical exploratory scripts and results accumulated during iterative development. It is retained for auditability, not as the normative release interface.
- `release/system/` is synchronized from the current DEV source for the components that matter to the paper-facing feedback and adaptive-L1 pipeline.
- `release/verification/` contains regression/self-check scripts for maintainers. These are validation helpers, not part of the paper's primary result-generation pipeline.

```
release/
├── code/                    # Current canonical experiment scripts
│   ├── l1_scorer.py         # Layer 1 scorer used by the current release experiments
│   ├── run_experiments.py   # Core rerun pipeline on unified/corrected MINEC ground truth
│   ├── generate_threshold_frontier.py # Threshold-objective ablation + Pareto frontier generation
│   ├── generate_threshold_sensitivity.py # MINEC dual-threshold sensitivity figure generator
│   ├── generate_jaccard_sensitivity.py   # MINEC Jaccard-threshold sensitivity figure generator
│   ├── 26_gpt54_annotation.py    # GPT-5.4 cross-model validation on 300 sampled pairs
│   ├── 28_human_annotation_analysis.py # Human adjudication analysis for disagreement cases
│   ├── 29_three_way_analysis.py  # Three-way GLM/GPT/human comparison utilities
│   ├── 30_claude_opus_probe.py   # Early Claude probe on sampled disagreements
│   ├── 31_reannotate_glm52_canonical.py # GLM-5.2 canonical re-annotation
│   ├── 32_annotate_claude_opus_canonical.py # Claude canonical re-annotation
│   └── config.py            # Shared paths and constants
│
├── data/                    # Datasets
│   ├── dataset_v3_cleaned.json   # 2,639 candidate pairs (MINEC)
│   ├── entity_fragments.json     # Source text fragments for L2 grounding
│   ├── minec_ground_truth_v2.json # Current corrected ground truth with manual review merged in
│   ├── minec_ground_truth_v1.json # Earlier consolidated version retained for auditability
│   ├── minec_disagreements_v1.json
│   └── minec_data_gap_candidates_v1.json # Suspected extraction/source sparsity audit list
│
├── results/                 # Current canonical experiment outputs
│   ├── l1_distribution.json
│   ├── l1_accuracy.json
│   ├── expert_bucket.json
│   ├── pipeline_accuracy_empirical.json
│   ├── classifier_ablation.json
│   ├── feedback_convergence.json
│   ├── per_type_accuracy.json
│   ├── feature_importance.json
│   ├── scored_pairs.json
│   ├── threshold_objective_ablation.json   # Main current paper evidence
│   ├── threshold_frontier.json             # Cost–accuracy frontier / Pareto view
│   ├── human_annotation_analysis.json
│   ├── gpt54_annotation_300.json
│   ├── gpt54_annotation_300_checkpoint.json
│   ├── glm52_canonical_annotation.json
│   ├── glm52_canonical_checkpoint.json
│   ├── claude_opus_canonical_annotation.json
│   ├── claude_opus_canonical_checkpoint.json
│   ├── claude_opus_probe_50.json
│   └── checkpoints_v3/           # L2 model verdict checkpoints still used by current reruns
│       ├── glm-5_checkpoint.json
│       ├── glm-4.5-air_checkpoint.json
│       ├── glm-4.5_checkpoint.json
│       ├── glm-5.2_checkpoint.json
│       └── qwen3.6_checkpoint.json
│
├── legacy/                  # Superseded scripts/results kept temporarily for auditability
│   ├── code/
│   │   ├── gen_jaccard_figure.py   # Historical paper-asset generator (superseded)
│   │   └── sweep_jaccard_fine.py   # Historical τ sweep source retained for auditability
│   └── results/
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
├── verification/            # Non-paper regression/self-check scripts for the release snapshot
│   └── test_cascaderule_llm.py
│
└── system/                  # MINEC system (open-source subset)
    ├── docker-compose.yml
    ├── Dockerfile
    ├── nebula_schema.ngql
    ├── postgres_schema.sql
    ├── backend/             # FastAPI + Nebula + PostgreSQL
    └── frontend_dist/       # Static frontend
```

## Feature-Space Note

`extract_features()` does not emit a pre-hardcoded static 20-dimensional vector. It dynamically expands field-level and aggregate signals from the effective properties of each entity pair, then aligns them through a shared `feature_names` list during training. In the current MINEC unified-LR experiment, the resulting aligned feature space contains 20 dimensions. Hybrid LR uses per-type training subsets and per-type thresholds over this same aligned global feature space rather than a separately hand-defined dimensionality for each type.

## Threshold Semantics

- **Fixed L1** uses the bidirectional rule score scale `[-1,+1]` with fixed routing thresholds `0.6 / -0.4`.
- **Unified LR / Hybrid LR** use probability space `[0,1]`. Their `theta_merge` and `theta_reject` are not copies of `0.6 / -0.4`; they are automatically tuned on training data via `select_thresholds` after logistic-regression training.

## Fixed L1 Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| τ (Jaccard field match) | 0.65 | Weak-field overlap threshold for match/conflict |
| θ_merge (routing, Fixed L1) | 0.6 | Rule score ≥ this → auto-merge |
| θ_reject (routing, Fixed L1) | -0.4 | Rule score ≤ this → auto-reject |
| λ (conflict penalty) | 1.5 | Conflict fields weighted 1.5× |
| Scoring range | [-1, +1] | Bidirectional: match adds, conflict subtracts |

## Key Results

| Metric | Value |
|--------|-------|
| L1 accuracy (fixed weights) | 79.3% |
| L1 interception rate | 38.6% |
| Merge precision | 82.9% |
| Reject precision | 76.4% |
| Fixed pipeline accuracy (strict / resolved) | 85.8% / 86.9% |
| Unified legacy strict pipeline | 81.6% |
| Hybrid legacy strict pipeline | 81.6% |
| Unified normalized strict pipeline | 86.7% |
| Hybrid normalized strict pipeline | 86.8% |
| Unified pipeline-aware strict pipeline | 91.0% |
| Hybrid pipeline-aware strict pipeline | 90.5% |
| Unified matched-fixed | 86.2% strict @ 74.5% interception |
| Hybrid matched-fixed | 86.1% strict @ 67.4% interception |
| GLM-5 strict L2 accuracy within fixed pipeline escalations | 91.8% |
| Human adjudication set | 59 disagreements / 51 comparable |
| GLM-5.2 vs human (disagreements only) | 88.2% |
| DBP15K fixed L1 (paper-reported) | 77.9% acc / 79.5% intercept / 81.1% pipeline / 20.5% LLM cost |
| DBP15K adaptive L1 (paper-reported) | 87.7% acc / 91.3% intercept / 88.2% pipeline / 8.7% LLM cost |
| DBP15K GLM-5.2 hard-case accuracy | 92.8% on 499 valid / 500 sampled |

## Data Integrity

The two primary files under `data/` are byte-identical to the originals in `experiments/data/` (the shared top-level working copy):

- `dataset_v3_cleaned.json`
- `entity_fragments.json`

See `PROVENANCE.md` for SHA256 checksums and for the distinction between raw DBP15K outputs and final paper-reported summaries.

## Reproduction

```bash
cd code/

# Canonical MINEC rerun on the corrected ground truth
python3 run_experiments.py

# Threshold-objective ablation and frontier generation
python3 generate_threshold_frontier.py

# Paper figures generated from the authoritative release results
python3 generate_threshold_sensitivity.py
python3 generate_jaccard_sensitivity.py
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
