# Phase 106 NIST AMMT Spatial Target Representation Gate

- Status: `phase106_spatial_target_gap_ready_focused_no_training_validation`
- Representation: `registered_layer_camera_spatial_statistics_v1`
- Selected target: `target_center_periphery_contrast`
- Focused validation allowed: `True`
- Model training allowed: `false`
- A100 training allowed now: `false`

| Target | Status | Method | Val RMSE | Val NRMSE | Mean val RMSE | Val gain vs mean |
|---|---|---|---:|---:|---:|---:|
| target_center_mean | blocked_no_baseline_visible_gap | mean | 1.467979 | 0.247314 | 1.467979 | 0.000000 |
| target_periphery_mean | blocked_no_baseline_visible_gap | mean | 0.802996 | 0.322944 | 0.802996 | 0.000000 |
| target_center_periphery_contrast | candidate_spatial_target_gap_ready_for_focused_validation | hist_gradient_boosting | 1.174314 | 0.255740 | 1.592342 | 0.262524 |
| target_hot_fraction_q90 | blocked_no_baseline_visible_gap | mean | 0.005015 | 0.279134 | 0.005015 | 0.000000 |
| target_top_half_mean | blocked_no_baseline_visible_gap | mean | 1.067997 | 0.226969 | 1.067997 | 0.000000 |
| target_bottom_half_mean | blocked_no_baseline_visible_gap | mean | 1.014427 | 0.463111 | 1.014427 | 0.000000 |
| target_left_half_mean | blocked_no_baseline_visible_gap | mean | 1.970630 | 0.286761 | 1.970630 | 0.000000 |
| target_right_half_mean | candidate_spatial_target_gap_ready_for_focused_validation | hist_gradient_boosting | 0.771698 | 0.237545 | 1.073832 | 0.281360 |
| target_vertical_contrast | blocked_no_baseline_visible_gap | mean | 1.465468 | 0.462225 | 1.465468 | 0.000000 |
| target_horizontal_contrast | blocked_no_baseline_visible_gap | hist_gradient_boosting | 2.783770 | 0.330164 | 2.807353 | 0.008400 |
| target_quadrant_contrast | candidate_spatial_target_gap_ready_for_focused_validation | hist_gradient_boosting | 1.717069 | 0.212827 | 2.947129 | 0.417376 |
| target_grid_max_mean | candidate_spatial_target_gap_ready_for_focused_validation | hist_gradient_boosting | 2.500625 | 0.217866 | 2.971366 | 0.158426 |
| target_grid_min_mean | blocked_no_baseline_visible_gap | mean | 1.280491 | 0.381802 | 1.280491 | 0.000000 |
| target_grid_mean_range | candidate_spatial_target_gap_ready_for_focused_validation | hist_gradient_boosting | 2.872929 | 0.251717 | 3.763895 | 0.236714 |
| target_local_variance_mean | blocked_strong_baseline_solved_validation_target | extra_trees | 4.572187 | 0.061242 | 31.318544 | 0.854010 |
| target_gradient_mean | blocked_strong_baseline_solved_validation_target | hist_gradient_boosting | 0.162704 | 0.065548 | 1.096638 | 0.851634 |
| target_gradient_q90 | blocked_strong_baseline_solved_validation_target | hist_gradient_boosting | 0.100008 | 0.040372 | 1.167104 | 0.914311 |
| target_camera_pair_mean_delta | blocked_no_baseline_visible_gap | hist_gradient_boosting | 1.552134 | 0.252771 | 1.578333 | 0.016599 |
| target_camera_pair_std_delta | blocked_strong_baseline_solved_validation_target | hist_gradient_boosting | 0.443558 | 0.046974 | 3.633037 | 0.877910 |
| target_camera_pair_gradient_delta | blocked_strong_baseline_solved_validation_target | hist_gradient_boosting | 0.155436 | 0.032559 | 2.134205 | 0.927169 |

Next action: enter seed-7 focused validation or low-capacity mechanism design only after reviewing target_center_periphery_contrast against route guards
