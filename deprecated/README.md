# Deprecated — 论文工程废弃物归档（2026-11-04）

GT v2全链统一后的一次性/被取代文件。保留在此仅作溯源，不再被任何保留脚本引用。

## code/
| 文件 | 替代/原因 |
|---|---|
| qwen_probe_300.py | Qwen3.6端点验身探针，三重验身已完成（结论在skill） |
| alias_estimate.py / alias_screening.py | 别名漏网一次性筛查，结论已写入论文blocking段 |
| final_operating_points.py / select_l1_acc_max_points.py | 旧探索选点，被GT v2管线(run_experiments.py + baseline_routing.py)取代 |
| compute_prf1.py | 旧标签口径P/R/F1脚本（MINEC部分论文未引用其数字，DBP15K的P/R另有来源），读的三个输入json均已废弃 |

## results/
| 文件 | 替代 |
|---|---|
| dual_threshold_sweep.json | dual_threshold_sweep_gt2.json |
| jaccard_sensitivity.json | jaccard_sensitivity_gt2.json |
| pareto_frontier_data.json | pareto_frontier_data_gt2.json（实证重放版） |
| baseline_routing.json | baseline_routing_gt2.json |
| feedback_convergence.json | feedback_convergence_gt2.json |
| all_lr_coefficients.json | unified_coefficients_gt2.json |
| pipeline_accuracy_empirical.json / pipeline_vs_pure_l2.json / l1_only_empirical_turbo.json | threshold_selection_experiments.json + verify_pipeline_vs_pure_l2.py（GT v2） |
| final_operating_points.json / l1_acc_max_operating_points.json | ours_fine_grid_gt2.json + threshold_selection_experiments.json |
| prf1_metrics.json | 旧标签口径P/R/F1（论文MINEC未引用；fixed的strict口径如需GT v2版，可由threshold_selection_experiments.json重算） |
| *.png / *.log / ditto_progress.npz | 图预览、训练日志、断点中间物 |
| alias_screening.json / alias_sample_review.md | 一次性筛查产物 |
| backup_pre_gt2_rerun/ | GT v2重跑前备份（旧标签存档快照） |

恢复：`mv deprecated/<path> ../<path>`。

## pipeline_accuracy.stale.json (moved 2026-11-05)
Pre-GT-v2 artifact: `pipeline_fixed=86.8` counts unresolved pairs leniently.
Paper-final fixed-cascade figure is 86.3 strict (`threshold_selection_experiments.json`).
Retained for provenance only; do not cite.
