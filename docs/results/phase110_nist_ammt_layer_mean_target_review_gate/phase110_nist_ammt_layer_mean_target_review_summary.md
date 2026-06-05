# Phase 110 NIST AMMT Layer-Mean Target Review Gate

- Status: `phase110_layer_mean_target_review_closed_layer_time_shortcut`
- Reviewed target: `target_cp_layer_mean`
- Layer/time shortcut detected: `True`
- Model mechanism allowed: `false`
- Model training allowed: `false`
- A100 training allowed now: `false`

| Profile | Status | Method | Val RMSE | Test RMSE | Delta vs full val |
|---|---|---|---:|---:|---:|
| mean_guard | reference_mean_guard | mean | 1.262500 | 1.327112 | 0.322402 |
| full_phase108 | reference_full_phase108_profile | hist_gradient_boosting | 0.940098 | 1.060691 | 0.000000 |
| no_camera | profile_has_validation_signal | hist_gradient_boosting | 0.940098 | 1.060691 | 0.000000 |
| camera_only | profile_has_validation_signal | hist_gradient_boosting | 1.262500 | 1.327112 | 0.322402 |
| source_only | no_independent_source_validation_gain | extra_trees | 1.257470 | 1.622680 | 0.317372 |
| layer_time_only | layer_time_shortcut_matches_or_beats_full_validation | hist_gradient_boosting | 0.716473 | 1.137170 | -0.223625 |
| layer_time_camera | layer_time_shortcut_matches_or_beats_full_validation | hist_gradient_boosting | 0.716473 | 1.137170 | -0.223625 |

Next action: close NIST AMMT sequence target branch as diagnostic; do not train
