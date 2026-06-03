# Phase 105 NIST AMMT Source/Path Feature Gate

- Status: `phase105_source_path_feature_gate_blocked_no_hgb_gain`
- Target: `target_intensity_std`
- Phase 104 guard: `hist_gradient_boosting` val RMSE `0.5594973535400991`
- Selected feature profile: `None`
- CPU smoke allowed: `False`
- A100 training allowed now: `false`

| Feature profile | Status | Method | Val RMSE | Test RMSE | Val gain vs guard | Test gain vs guard |
|---|---|---|---:|---:|---:|---:|
| base_guard_replay | guard_replay_reference | hist_gradient_boosting | 0.559497 | 0.713065 | 0.000000 | 0.000000 |
| base_energy | blocked_no_validation_gain_over_hgb_guard | hist_gradient_boosting | 0.559497 | 0.713065 | 0.000000 | 0.000000 |
| base_density | blocked_no_validation_gain_over_hgb_guard | hist_gradient_boosting | 0.559497 | 0.713065 | 0.000000 | 0.000000 |
| base_green | blocked_no_validation_gain_over_hgb_guard | hist_gradient_boosting | 0.559497 | 0.713065 | 0.000000 | 0.000000 |
| base_all_physics | blocked_no_validation_gain_over_hgb_guard | hist_gradient_boosting | 0.561350 | 0.712704 | -0.001853 | 0.000361 |
| physics_only | blocked_no_validation_gain_over_hgb_guard | hist_gradient_boosting | 0.584469 | 0.740072 | -0.024971 | -0.027007 |

Next action: close deterministic source/path proxy features as diagnostic or refine registered targets
