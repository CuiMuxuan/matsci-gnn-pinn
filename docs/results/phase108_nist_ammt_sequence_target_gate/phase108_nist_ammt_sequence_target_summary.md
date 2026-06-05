# Phase 108 NIST AMMT Sequence Target Gate

- Status: `phase108_sequence_target_gap_ready_focused_review`
- Base target: `target_center_periphery_contrast`
- Selected target: `target_cp_camera_pair_delta`
- Focused review allowed: `True`
- Model training allowed: `false`
- A100 training allowed now: `false`

| Target | Status | Method | Val RMSE | Val NRMSE | Mean val RMSE | Val gain vs mean |
|---|---|---|---:|---:|---:|---:|
| target_cp_layer_mean | candidate_sequence_target_gap_ready_for_review | hist_gradient_boosting | 0.940098 | 0.409992 | 1.262500 | 0.255368 |
| target_cp_camera_pair_delta | candidate_sequence_target_gap_ready_for_review | hist_gradient_boosting | 1.809450 | 0.278516 | 1.940769 | 0.067663 |
| target_cp_abs_camera_pair_delta | blocked_no_baseline_visible_gap | mean | 0.781419 | 0.332048 | 0.781419 | 0.000000 |
| target_cp_layer_camera_range | blocked_no_baseline_visible_gap | mean | 0.781419 | 0.332048 | 0.781419 | 0.000000 |
| target_cp_deviation_from_layer_mean | candidate_sequence_target_gap_ready_for_review | hist_gradient_boosting | 0.904725 | 0.278516 | 0.970384 | 0.067663 |
| target_cp_prev_same_camera_delta | blocked_no_baseline_visible_gap | hist_gradient_boosting | 3.277696 | 0.173595 | 3.296014 | 0.005557 |
| target_cp_prev_layer_mean_delta | blocked_no_baseline_visible_gap | hist_gradient_boosting | 2.641939 | 0.214121 | 2.721532 | 0.029246 |
| target_cp_prev_same_camera_abs_delta | blocked_strong_baseline_solved_validation_target | hist_gradient_boosting | 2.686103 | 0.183650 | 2.898835 | 0.073385 |
| target_cp_prev_layer_mean_abs_delta | blocked_no_baseline_visible_gap | mean | 1.982028 | 0.226945 | 1.982028 | 0.000000 |

Next action: review target_cp_camera_pair_delta sequence target representation before any model training
