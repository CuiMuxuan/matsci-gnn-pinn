# Phase 169 Hidden-Source/Closure Identifiability Gate

## Gate
- Status: `phase169_hidden_source_closure_identifiability_ready_phase170_low_budget_mechanism_design`
- Candidate method: `calibrated_bayesian_hidden_source_closure_posterior`
- Best control method: `grid_least_squares_source_closure_control`
- Validation score gain vs best control: `0.0707838724`
- Test reversal ratio vs best control: `1.005124092`
- Phase 170 low-budget mechanism design allowed: `true`
- Phase 169 model training allowed: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
This gate tests source/closure identifiability only. A positive result does not train a PINN and does not support AM-Bench or Bayesian neural claims; it only allows a later low-budget mechanism smoke design.

## Validation and Test Metrics
| method | method_family | split | case_count | center_shift_rmse | source_width_rmse | closure_coeff_rmse | joint_normalized_rmse | coverage90_mean | calibration_gap | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bayesian_hidden_source_closure_posterior | bayesian_candidate | val | 16 | 0.000436518 | 0.0004003389 | 0.0154361866 | 0.0407750999 | 0.1041666667 | 0.7958333333 | 0.1044417666 |
| bayesian_hidden_source_closure_posterior | bayesian_candidate | test | 16 | 0.0004185159 | 0.000376861 | 0.0228727705 | 0.0601864682 | 0.0625 | 0.8375 | 0.1271864682 |
| calibrated_bayesian_hidden_source_closure_posterior | bayesian_candidate | val | 16 | 0.000436518 | 0.0004003389 | 0.0154361866 | 0.0407750999 | 0.9375 | 0.0375 | 0.0437750999 |
| calibrated_bayesian_hidden_source_closure_posterior | bayesian_candidate | test | 16 | 0.0004185159 | 0.000376861 | 0.0228727705 | 0.0601864682 | 0.8958333333 | 0.0041666667 | 0.0605198015 |
| extra_trees_sensor_control | control | val | 16 | 0.0142437151 | 0.0182446847 | 0.0346006126 | 0.215067247 | 0 | 0.9 | 0.287067247 |
| extra_trees_sensor_control | control | test | 16 | 0.0130354847 | 0.0179028727 | 0.0303439531 | 0.205032466 | 0 | 0.9 | 0.277032466 |
| grid_least_squares_source_closure_control | control | val | 16 | 0.0004549617 | 0.00041943 | 0.0161110494 | 0.0425589723 | 0 | 0.9 | 0.1145589723 |
| grid_least_squares_source_closure_control | control | test | 16 | 0.0004256575 | 0.0004158941 | 0.0227457132 | 0.0598796394 | 0 | 0.9 | 0.1318796394 |
| moment_linearized_closure_control | control | val | 16 | 0.032761037 | 0.0555424906 | 0.2102597964 | 0.792552538 | 0 | 0.9 | 0.864552538 |
| moment_linearized_closure_control | control | test | 16 | 0.0326945544 | 0.0554800648 | 0.2085212978 | 0.7888786404 | 0 | 0.9 | 0.8608786404 |
| no_closure_source_control | control | val | 16 | 0.0004293786 | 0.0004597453 | 0.0829156198 | 0.2176570405 | 0 | 0.9 | 0.2896570405 |
| no_closure_source_control | control | test | 16 | 0.0004256575 | 0.0004500053 | 0.0829156198 | 0.2176548968 | 0 | 0.9 | 0.2896548968 |
