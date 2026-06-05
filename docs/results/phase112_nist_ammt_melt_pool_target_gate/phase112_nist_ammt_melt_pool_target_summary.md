# Phase 112 NIST AMMT Melt-Pool Target Gate

- Status: `phase112_melt_pool_target_gap_ready_focused_review`
- Row count: `63`
- Selected target: `target_mp_temporal_mean_range`
- Focused review allowed: `True`
- Model training allowed: `false`
- A100 training allowed now: `false`

| Target | Status | Profile | Method | Val RMSE | Val NRMSE | Layer/time val RMSE |
|---|---|---|---|---:|---:|---:|
| target_mp_mean_mean | candidate_melt_pool_target_gap_ready_for_focused_review | source_all_no_layer | extra_trees | 0.889177 | 0.201950 | 1.365462 |
| target_mp_mean_std | blocked_no_baseline_visible_gap | source_geometry | hist_gradient_boosting | 0.466135 | 0.289281 | 0.466135 |
| target_mp_q90_mean | candidate_melt_pool_target_gap_ready_for_focused_review | source_geometry | knn | 1.461908 | 0.226848 | 1.755627 |
| target_mp_max_mean | candidate_melt_pool_target_gap_ready_for_focused_review | source_all_no_layer | extra_trees | 25.758577 | 0.306245 | 29.418688 |
| target_mp_max_range | candidate_melt_pool_target_gap_ready_for_focused_review | source_geometry | knn | 0.288675 | 0.288675 | 0.383331 |
| target_mp_temporal_mean_range | candidate_melt_pool_target_gap_ready_for_focused_review | source_geometry | knn | 1.666605 | 0.243057 | 1.904477 |
| target_mp_early_late_mean_delta | blocked_no_baseline_visible_gap | source_power_path | extra_trees | 1.158370 | 0.363232 | 1.170720 |
| target_mp_peak_frame_position | blocked_no_baseline_visible_gap | source_geometry | hist_gradient_boosting | 0.246305 | 0.394088 | 0.246305 |

Next action: review target_mp_temporal_mean_range and its source profile before any model training
