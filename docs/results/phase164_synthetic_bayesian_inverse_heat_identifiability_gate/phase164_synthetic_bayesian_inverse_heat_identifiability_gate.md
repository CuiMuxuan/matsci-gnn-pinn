# Phase 164 Synthetic Bayesian Inverse-Heat Identifiability Gate

## Gate
- Status: `phase164_synthetic_bayesian_inverse_heat_identifiability_ready_phase165_sampler_gate`
- Selected method: `calibrated_bayesian_grid_posterior`
- Best control method: `grid_least_squares_control`
- Validation score gain vs best control: `0.0813259569`
- Test reversal ratio vs best control: `0.9340940991`
- Phase 165 adaptive sampler gate allowed: `true`
- Phase 164 model training allowed: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
The synthetic inverse task is intentionally physics-matched and local. A positive gate means hidden diffusivity/source-width parameters are identifiable against strong non-neural controls on this controlled task. It does not permit AM-Bench Bayesian PINN training.

## Validation and Test Metrics
| method | method_family | split | case_count | diffusivity_rmse | source_width_rmse | joint_normalized_rmse | coverage90_mean | calibration_gap | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bayesian_grid_posterior | bayesian_candidate | val | 5 | 0.0004748541 | 0.0010056931 | 0.0136452258 | 0.3 | 0.6 | 0.0736452258 |
| bayesian_grid_posterior | bayesian_candidate | test | 4 | 0.0005827661 | 0.0011719119 | 0.015988897 | 0.375 | 0.525 | 0.068488897 |
| calibrated_bayesian_grid_posterior | bayesian_candidate | val | 5 | 0.0004748541 | 0.0010056931 | 0.0136452258 | 1 | 0.1 | 0.0236452258 |
| calibrated_bayesian_grid_posterior | bayesian_candidate | test | 4 | 0.0005827661 | 0.0011719119 | 0.015988897 | 0.875 | 0.025 | 0.018488897 |
| extra_trees_sensor_control | control | val | 5 | 0.0086473476 | 0.0141988019 | 0.199071113 | 0 | 0.9 | 0.289071113 |
| extra_trees_sensor_control | control | test | 4 | 0.0078633662 | 0.0195130541 | 0.26105523 | 0 | 0.9 | 0.35105523 |
| grid_least_squares_control | control | val | 5 | 0.0004591253 | 0.0011173521 | 0.0149711827 | 0 | 0.9 | 0.1049711827 |
| grid_least_squares_control | control | test | 4 | 0.0005341224 | 0.0012755576 | 0.0171170089 | 0 | 0.9 | 0.1071170089 |
| moment_linearized_control | control | val | 5 | 0.0490684429 | 0.0673435862 | 0.9760421965 | 0 | 0.9 | 1.066042196 |
| moment_linearized_control | control | test | 4 | 0.0404890483 | 0.0716044633 | 0.9928345659 | 0 | 0.9 | 1.082834566 |
