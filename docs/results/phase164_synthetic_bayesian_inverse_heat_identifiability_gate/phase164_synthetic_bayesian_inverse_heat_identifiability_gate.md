# Phase 164 Synthetic Bayesian Inverse-Heat Identifiability Gate

## Gate
- Status: `phase164_synthetic_bayesian_inverse_heat_identifiability_ready_phase165_sampler_gate`
- Selected method: `calibrated_bayesian_grid_posterior`
- Best control method: `grid_least_squares_control`
- Validation score gain vs best control: `0.081325956949`
- Test reversal ratio vs best control: `0.934094099052`
- Phase 165 adaptive sampler gate allowed: `true`
- Phase 164 model training allowed: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
The synthetic inverse task is intentionally physics-matched and local. A positive gate means hidden diffusivity/source-width parameters are identifiable against strong non-neural controls on this controlled task. It does not permit AM-Bench Bayesian PINN training.

## Validation and Test Metrics
| method | method_family | split | case_count | diffusivity_rmse | source_width_rmse | joint_normalized_rmse | coverage90_mean | calibration_gap | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bayesian_grid_posterior | bayesian_candidate | val | 5 | 0.000474854143429 | 0.00100569310452 | 0.0136452257958 | 0.3 | 0.6 | 0.0736452257958 |
| bayesian_grid_posterior | bayesian_candidate | test | 4 | 0.000582766058195 | 0.00117191193781 | 0.0159888969818 | 0.375 | 0.525 | 0.0684888969818 |
| calibrated_bayesian_grid_posterior | bayesian_candidate | val | 5 | 0.000474854143429 | 0.00100569310452 | 0.0136452257958 | 1 | 0.1 | 0.0236452257958 |
| calibrated_bayesian_grid_posterior | bayesian_candidate | test | 4 | 0.000582766058195 | 0.00117191193781 | 0.0159888969818 | 0.875 | 0.025 | 0.0184888969818 |
| extra_trees_sensor_control | control | val | 5 | 0.00864734762648 | 0.0141988019326 | 0.199071112985 | 0 | 0.9 | 0.289071112985 |
| extra_trees_sensor_control | control | test | 4 | 0.00786336623676 | 0.0195130541399 | 0.261055230001 | 0 | 0.9 | 0.351055230001 |
| grid_least_squares_control | control | val | 5 | 0.000459125311053 | 0.00111735208711 | 0.0149711827448 | 0 | 0.9 | 0.104971182745 |
| grid_least_squares_control | control | test | 4 | 0.000534122368446 | 0.0012755575703 | 0.0171170088731 | 0 | 0.9 | 0.107117008873 |
| moment_linearized_control | control | val | 5 | 0.0490684429288 | 0.0673435861671 | 0.976042196478 | 0 | 0.9 | 1.06604219648 |
| moment_linearized_control | control | test | 4 | 0.0404890482665 | 0.0716044633155 | 0.992834565934 | 0 | 0.9 | 1.08283456593 |
