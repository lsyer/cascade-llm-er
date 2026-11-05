# CascadeRuleLLM: Learning to Route Rule–LLM Cascades

Experiment code and data for the paper "Learning to Route Rule–LLM Cascades for Knowledge Graph Entity Resolution". This `release/` directory is the authoritative paper artifact under `experiments/`.

All MINEC results use the corrected ground truth `data/minec_ground_truth_v2.json` (`final_label`; same 1,398 / different 1,193 / unknown 48; labeled n = 2,591).

## Repository Structure

```
release/
├── code/                    # Canonical experiment scripts
│   ├── l1_scorer.py                  # Layer 1 signal scorer (bidirectional rule score)
│   ├── run_experiments.py            # Core pipeline: threshold selection + routing ablation
│   ├── run_glm5turbo_l2.py           # MINEC L2 verdict runner (GLM-5-Turbo, primary model)
│   ├── compute_oof_probabilities.py  # Out-of-fold probabilities (oof_probabilities.npz)
│   ├── classifier_operating_points.py # Table 4: L1-acc-constrained operating points
│   ├── mode3 evaluation (run_experiments.py)  # Nested pipeline-aware 5-fold selection
│   ├── feedback_replay_gt2.py        # Feedback-convergence simulation (GT v2 protocol)
│   ├── entity_split_cv.py            # Entity-disjoint cross-validation
│   ├── qwen_full_l2.py / fragment_rerun_qwen.py  # Qwen judge: full L2 run / fragment rerun
│   ├── ditto_baseline.py             # Ditto router baseline (OOF probabilities)
│   ├── baseline_routing.py           # Equal-budget external baselines
│   ├── recompute_jaccard_gt2.py / generate_*    # Figures (τ-sweep, threshold, pareto)
│   └── config.py                     # Shared API keys and endpoints
│
├── data/                    # Datasets
│   ├── dataset_v3_cleaned.json       # 2,639 candidate pairs with props (MINEC)
│   ├── entity_fragments.json         # Source text fragments for L2 grounding
│   ├── minec_ground_truth_v2.json    # Corrected ground truth (manual review merged)
│   ├── minec_human_annotation_300.json # Independent blind annotation (300 pairs)
│   ├── manual_review_336_decisions.txt  # 336 manual adjudications
│   └── human_verified_pairs_context.json # Derived view for the 599 human-verified pairs
│
├── results/                 # Canonical experiment outputs (GT v2)
│   ├── checkpoints_v3/               # L2 model verdict checkpoints
│   ├── scored_pairs.json             # Pairs with L1 scores + features
│   ├── threshold_selection_experiments.json # Routing-objective ablation (Table 3)
│   ├── classifier_operating_points_gt2.json # Classifier ablation + pure-L2 + OOF matrix (Table 4)
│   ├── l2_comparison_gt2.json        # L2 model comparison on escalated pairs (Table 5)
│   ├── fragment_ablation_v2_results.json    # Fragment ablation, GLM-family rows (Table 6)
│   ├── fragment_rerun_qwen.json      # Fragment ablation, Qwen row (Table 6)
│   ├── per_type_l2_gt2.json          # Accuracy by entity type (Table 7)
│   ├── mode3_replay_gt2.json         # Nested pipeline-aware 5-fold replay
│   ├── entity_split_cv.json          # Entity-disjoint CV (90.7% @ 73.9% cost)
│   ├── qwen_l2_full.json             # Qwen judge on all 2,591 pairs (95.2%)
│   ├── qwen_cascade_replay.json      # Judge-swap frontier + equal-budget replay
│   ├── ditto_replay_gt2.json / ditto_oof.npz  # Ditto router
│   ├── baseline_routing_gt2.json / budget_table_gt2.json / ours_fine_grid_gt2.json # Equal-budget comparison
│   ├── oof_probabilities.npz         # OOF prediction matrix (frontier/budget analyses)
│   ├── feedback_convergence_gt2.json # Feedback simulation (Table 12)
│   ├── unified_coefficients_gt2.json / feature_importance.json / per_type_lr_coefficients.json # LR weights
│   └── hierarchy_comparison.json     # Unified/hybrid/per-type structure (fixed row stale, see notes)
│
├── dbp15k/                  # DBP15K cross-domain validation
├── models/                  # Trained LR classifiers (5-fold CV)
├── verification/            # Regression tests
├── deprecated/              # Stale artifacts, kept for provenance only
└── system/                  # MINEC system (open-source subset)
```

## Key Results (GLM-5-Turbo as primary L2, GT v2)

| Metric | Value |
|--------|-------|
| L1 accuracy (fixed weights) | 79.1% |
| L1 interception (fixed) | 38.6% |
| Pure L2 strict (GLM-5-Turbo, all 2,591) | 91.6% |
| L2 accuracy range on escalated pairs (4 models) | 83.7%–94.9% |
| Fixed cascade pipeline (strict) | 86.3% |
| Unified pipeline-aware (0.85/0.05) | 91.2% @ 71.0% LLM cost |
| Hybrid pipeline-aware | 90.3% |
| Unified/Hybrid L1-only utility | 82.4% |
| Judge swap: pure Qwen3.6-35B | 95.2% @ 100% |
| Judge swap: LR router + Qwen | 94.6% @ 71.0% |
| Ditto router (GLM / Qwen) | 91.4% / 93.1% @ 25.2% |
| Entity-disjoint CV strict | 90.7% @ 73.9% |
| Fragment ablation avg Δ (paired, n=290) | +3.1 pp |
| DBP15K L2 accuracy (GLM-5-Turbo) | 95.4% (4,843/4,876) |
| Human annotation subset (300 pairs) | 95.7% (GLM-5-Turbo) / 90.0% (pipeline) |

## Interpretation Notes (read before reusing artifacts)

1. **Two OOF probability matrices exist and are NOT interchangeable.**
   - `oof_probabilities.npz` — used by the equal-budget/frontier analyses (supports the 779-interception replay, L1-wins-36 / L2-wins-34 counts, and the budget table).
   - `classifier_operating_points_gt2.json` (`oof_probabilities` field) — used for the classifier-ablation table (reproduces 96.0 / 21.1 / 91.4 exactly).
   Both are internally self-consistent; they come from different CV runs. Reproduce a number from the same matrix that produced it.
2. **Fragment ablation mixed sources (paper Table 6).** GLM-family rows come from `fragment_ablation_v2_results.json`; the Qwen row comes from `fragment_rerun_qwen.json` (95.2 / 94.5 / +0.7). The table average (+3.1 pp) is over these four rows.
3. **`hierarchy_comparison.json`**: the `fixed.pipeline = 86.8` field is a stale pre-strict-convention number; the unified/hybrid/per-type rows are current. Paper fixed-cascade figure: 86.3 (strict, `threshold_selection_experiments.json`).
4. **`qwen_cascade_replay.json`**: `pure_qwen_pct = 95.45` is the replay-sample estimate; the paper cites 95.2 from the full run `qwen_l2_full.json` (95.21). Use the full-run number.
5. `pipeline_accuracy.json` (86.8, lenient unresolved handling) has been moved to `deprecated/pipeline_accuracy.stale.json`.
6. Sweep sizes: `dual_threshold_sweep_gt2.json` has 99 threshold configurations (0.01 grid); older "300+" descriptions refer to a pre-GT-v2 sweep.

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

# MINEC: OOF probabilities / operating points / entity-split CV
python3 compute_oof_probabilities.py
python3 classifier_operating_points.py
python3 entity_split_cv.py

# Qwen judge: full L2 run + cascade replay
python3 qwen_full_l2.py

# Paper figures
python3 generate_threshold_sensitivity.py
python3 generate_jaccard_sensitivity.py
python3 generate_pareto_frontier.py
```

## DBP15K Validation

```bash
cd dbp15k/
python3 run_dbp15k_glm5turbo.py   # merged checkpoint: 4,876 calls total
```

## Annotation Pipeline

MINEC labels are produced by a multi-stage annotation pipeline:
1. **GLM-5.2** provides provisional same/different labels (outside the evaluated cascade)
2. **Claude Opus 4.8** surfaces unresolved cases and disagreements for manual review
3. **Human adjudication** resolves the review queue (336 cases)

An independent human annotation set of 300 stratified-random pairs (`data/minec_human_annotation_300.json`) validates label quality: GLM-5-Turbo achieves 95.7% agreement with human labels. Six of the 300 blind labels (2.0%) disagree with the final ground truth (all expert-different vs corrected-same; four equipment subclasses, two event paraphrases) — retained as recorded.

## Citation

```bibtex
@article{cascaderule2026,
  title={Learning to Route Rule--LLM Cascades for Knowledge Graph Entity Resolution},
  author={Li, Shaoyong and Xin, Jiang and Feng, Yao},
  journal={Journal of Intelligent Information Systems (under review)},
  year={2026}
}
```
