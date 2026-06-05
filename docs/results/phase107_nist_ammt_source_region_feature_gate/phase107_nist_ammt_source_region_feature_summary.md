# Phase 107 NIST AMMT Source-Region Feature Gate

- Status: `phase107_source_region_feature_gate_blocked_no_phase106_gain`
- Target: `target_center_periphery_contrast`
- Phase 106 guard: `hist_gradient_boosting` val RMSE `1.174314337940004`
- Selected feature profile: `None`
- Focused review allowed: `False`
- Model training allowed: `false`
- A100 training allowed now: `false`

| Feature profile | Status | Method | Val RMSE | Test RMSE | Val gain vs guard | Test gain vs guard |
|---|---|---|---:|---:|---:|---:|
| phase106_guard_replay | phase106_guard_replay_reference | hist_gradient_boosting | 1.174314 | 1.382751 | 0.000000 | 0.000000 |
| source_center_periphery | blocked_no_validation_gain_over_phase106_guard | hist_gradient_boosting | 1.174314 | 1.382751 | 0.000000 | 0.000000 |
| source_grid_region | blocked_test_reversal_against_phase106_guard | hist_gradient_boosting | 1.015222 | 1.537635 | 0.159092 | -0.154884 |
| source_moment_drift | blocked_no_validation_gain_over_phase106_guard | hist_gradient_boosting | 1.186112 | 1.741800 | -0.011798 | -0.359049 |
| source_region_all | blocked_no_validation_gain_over_phase106_guard | hist_gradient_boosting | 1.187121 | 1.704267 | -0.012806 | -0.321517 |
| source_region_only | blocked_no_validation_gain_over_phase106_guard | hist_gradient_boosting | 1.227306 | 1.796377 | -0.052992 | -0.413626 |

Next action: close sampled source-region path features as diagnostic; do not train
