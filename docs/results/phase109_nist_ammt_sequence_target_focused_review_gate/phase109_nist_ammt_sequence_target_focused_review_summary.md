# Phase 109 NIST AMMT Sequence Target Focused Review Gate

- Status: `phase109_sequence_target_focused_review_closed_camera_shortcut`
- Reviewed target: `target_cp_camera_pair_delta`
- Camera shortcut detected: `True`
- Model mechanism allowed: `false`
- Model training allowed: `false`
- A100 training allowed now: `false`

| Profile | Status | Method | Val RMSE | Test RMSE | Delta vs full val |
|---|---|---|---:|---:|---:|
| mean_guard | reference_mean_guard | mean | 1.940769 | 1.846333 | 0.131319 |
| full_phase108 | reference_full_phase108_profile | hist_gradient_boosting | 1.809450 | 1.809990 | 0.000000 |
| no_camera | no_validation_gain_over_mean_guard | extra_trees | 1.940769 | 1.846333 | 0.131319 |
| camera_only | profile_has_independent_validation_signal | extra_trees | 1.808794 | 1.809849 | -0.000656 |
| source_only | no_validation_gain_over_mean_guard | extra_trees | 1.940769 | 1.846333 | 0.131319 |
| layer_time_camera | shortcut_matches_full_validation | hist_gradient_boosting | 1.809450 | 1.809990 | 0.000000 |
| layer_time_only | no_validation_gain_over_mean_guard | extra_trees | 1.940769 | 1.846333 | 0.131319 |

Next action: close Phase 108 camera-pair delta target as diagnostic; do not train
