# Phase 171 Hidden-Closure Low-Budget Smoke

## Gate
- Status: `phase171_hidden_closure_low_budget_smoke_ready_phase172_trainable_design`
- Selected variant: `calibrated_hidden_source_closure_parameter_head`
- Best control variant: `posterior_only_calibrated_bayesian_no_neural`
- Validation score gain vs best control: `0.0011359227`
- Posterior validation closure gain: `0.0015145636`
- Posterior test closure gain: `0.0031637456`
- Phase 172 trainable hidden-closure design allowed: `true`
- Phase 171 model training allowed: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
This smoke is synthetic and NumPy-only. A positive gate means the explicit calibrated closure head improves the interpretable closure metric over the posterior-only control without degrading field prediction; it is not PINN training, AM-Bench evidence, or an A100-80GB justification.

## Variants
| variant_id | family | source_estimator | closure_estimator | executed | is_control | description |
| --- | --- | --- | --- | --- | --- | --- |
| calibrated_hidden_source_closure_parameter_head | mechanism_candidate | phase169_calibrated_posterior_center_width | train_split_linear_calibration_head | true | false | explicit low-dimensional closure head calibrated from train cases |
| posterior_only_calibrated_bayesian_no_neural | control | phase169_calibrated_posterior_center_width | posterior_point_estimate | true | true | strong no-neural posterior-only control |
| grid_least_squares_source_closure_control | control | grid_search_best_sse_center_width | least_squares_closure_coefficient | true | true | required non-Bayesian inverse control |
| no_closure_source_control | control | grid_search_without_closure_term | zero_closure | true | true | tests whether the closure term is necessary |
| wrong_source_prior_control | control | deliberately_shifted_source_prior | least_squares_under_wrong_source | true | true | tests whether a wrong source prior can solve the task |
| data_only_sensor_regression_control | control | polynomial_sensor_regression_no_physics | zero_closure | true | true | NumPy data-only proxy for the registered data-only neural control |
| fixed_nominal_source_closure_control | control | fixed_nominal_center_width | least_squares_closure_coefficient | true | true | tests whether hidden source learning is necessary |
| failure_sampler_retrain_block | blocked_control | not_executed | not_executed | false | true | registered block against repeating the Phase 167 sampler-retuning route |

## Summary Metrics
| variant_id | family | split | seed_count | case_count | field_rmse_mean | hot_q90_rmse_mean | gradient_q90_rmse_mean | closure_abs_error_mean | coverage90_mean | selection_score_mean | selection_score_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibrated_hidden_source_closure_parameter_head | mechanism_candidate | val | 3 | 48 | 0.0244252778 | 0.0411811 | 0.0572416905 | 0.0263759868 | 0.8819444444 | 0.0611121014 | 0.0214938526 |
| calibrated_hidden_source_closure_parameter_head | mechanism_candidate | test | 3 | 48 | 0.0256532769 | 0.0455446368 | 0.0612577044 | 0.0172555913 | 0.9375 | 0.0563296682 | 0.0220509477 |
| data_only_sensor_regression_control | control | val | 3 | 48 | 2.066629422 | 4.347394932 | 2.750988124 | 0.075 | 0 | 3.287457221 | 0.655957135 |
| data_only_sensor_regression_control | control | test | 3 | 48 | 2.066800347 | 4.321894152 | 2.745832829 | 0.075 | 0 | 3.28201246 | 0.6458761195 |
| fixed_nominal_source_closure_control | control | val | 3 | 48 | 0.7381966286 | 1.302007316 | 1.622326603 | 0.8270726066 | 0 | 1.801135207 | 0.8769404657 |
| fixed_nominal_source_closure_control | control | test | 3 | 48 | 0.7072711833 | 1.235917585 | 1.565312042 | 0.827246484 | 0 | 1.751420767 | 0.7141901629 |
| grid_least_squares_source_closure_control | control | val | 3 | 48 | 0.0263732727 | 0.0444826802 | 0.0616376905 | 0.0302465546 | 0 | 0.0841184938 | 0.0231563872 |
| grid_least_squares_source_closure_control | control | test | 3 | 48 | 0.0265095193 | 0.0468583864 | 0.0630571244 | 0.0225972062 | 0 | 0.0791348137 | 0.0254641775 |
| no_closure_source_control | control | val | 3 | 48 | 0.0280890029 | 0.0453651882 | 0.0629806427 | 0.075 | 0 | 0.1197101048 | 0.031989168 |
| no_closure_source_control | control | test | 3 | 48 | 0.0290625452 | 0.0494574023 | 0.0659084149 | 0.075 | 0 | 0.1217948671 | 0.0370843219 |
| posterior_only_calibrated_bayesian_no_neural | control | val | 3 | 48 | 0.0244252778 | 0.0411811 | 0.0572416905 | 0.0278905504 | 0.8819444444 | 0.0622480241 | 0.0252675588 |
| posterior_only_calibrated_bayesian_no_neural | control | test | 3 | 48 | 0.0256532769 | 0.0455446368 | 0.0612577044 | 0.0204193369 | 0.9375 | 0.0587024774 | 0.0254347314 |
| wrong_source_prior_control | control | val | 3 | 48 | 0.7759218074 | 1.241204414 | 1.686848984 | 0.8082501097 | 0 | 1.819035171 | 0.2635866881 |
| wrong_source_prior_control | control | test | 3 | 48 | 0.7737491665 | 1.259373681 | 1.676945851 | 0.816953174 | 0 | 1.826033368 | 0.2564821358 |

## Seed Summary
| seed | variant_id | split | case_count | field_rmse_mean | closure_abs_error_mean | selection_score_mean |
| --- | --- | --- | --- | --- | --- | --- |
| 171 | calibrated_hidden_source_closure_parameter_head | val | 16 | 0.0245230333 | 0.0250702117 | 0.0604787397 |
| 171 | calibrated_hidden_source_closure_parameter_head | test | 16 | 0.0246828665 | 0.0178274375 | 0.0554067997 |
| 171 | data_only_sensor_regression_control | val | 16 | 2.066617329 | 0.075 | 3.286978925 |
| 171 | data_only_sensor_regression_control | test | 16 | 2.066771954 | 0.075 | 3.281779511 |
| 171 | fixed_nominal_source_closure_control | val | 16 | 0.7381969586 | 0.8301852046 | 1.803450157 |
| 171 | fixed_nominal_source_closure_control | test | 16 | 0.7072638775 | 0.8374386004 | 1.758906848 |
| 171 | grid_least_squares_source_closure_control | val | 16 | 0.0266390154 | 0.0282463834 | 0.0830642092 |
| 171 | grid_least_squares_source_closure_control | test | 16 | 0.0264025016 | 0.0257209561 | 0.0812317175 |
| 171 | no_closure_source_control | val | 16 | 0.0283440386 | 0.075 | 0.1200962177 |
| 171 | no_closure_source_control | test | 16 | 0.0291535945 | 0.075 | 0.121912791 |
| 171 | posterior_only_calibrated_bayesian_no_neural | val | 16 | 0.0245230333 | 0.0263031789 | 0.0614034651 |
| 171 | posterior_only_calibrated_bayesian_no_neural | test | 16 | 0.0246828665 | 0.0210502497 | 0.0578239088 |
| 171 | wrong_source_prior_control | val | 16 | 0.7767623361 | 0.7971526025 | 1.812249076 |
| 171 | wrong_source_prior_control | test | 16 | 0.7736275975 | 0.8084723704 | 1.819380889 |
| 172 | calibrated_hidden_source_closure_parameter_head | val | 16 | 0.0245879018 | 0.0248454496 | 0.0600441783 |
| 172 | calibrated_hidden_source_closure_parameter_head | test | 16 | 0.0261130535 | 0.0166381303 | 0.0563894501 |
| 172 | data_only_sensor_regression_control | val | 16 | 2.066613174 | 0.075 | 3.28726739 |
| 172 | data_only_sensor_regression_control | test | 16 | 2.066799318 | 0.075 | 3.28190392 |
| 172 | fixed_nominal_source_closure_control | val | 16 | 0.7381911253 | 0.8325791006 | 1.80522333 |
| 172 | fixed_nominal_source_closure_control | test | 16 | 0.7072813596 | 0.8209217235 | 1.74675904 |
| 172 | grid_least_squares_source_closure_control | val | 16 | 0.0263593053 | 0.0298920754 | 0.0837698416 |
| 172 | grid_least_squares_source_closure_control | test | 16 | 0.0264721547 | 0.0204001969 | 0.077412466 |
| 172 | no_closure_source_control | val | 16 | 0.0277117033 | 0.075 | 0.119048317 |
| 172 | no_closure_source_control | test | 16 | 0.0290442869 | 0.075 | 0.1216667374 |
| 172 | posterior_only_calibrated_bayesian_no_neural | val | 16 | 0.0245879018 | 0.0252303145 | 0.0603328269 |
| 172 | posterior_only_calibrated_bayesian_no_neural | test | 16 | 0.0261130535 | 0.0194877949 | 0.0585266985 |
| 172 | wrong_source_prior_control | val | 16 | 0.7769708457 | 0.8181741098 | 1.827685794 |
| 172 | wrong_source_prior_control | test | 16 | 0.7736254997 | 0.815147635 | 1.824001714 |
| 173 | calibrated_hidden_source_closure_parameter_head | val | 16 | 0.0241648982 | 0.029212299 | 0.0628133861 |
| 173 | calibrated_hidden_source_closure_parameter_head | test | 16 | 0.0261639107 | 0.0173012061 | 0.0571927549 |
| 173 | data_only_sensor_regression_control | val | 16 | 2.066657765 | 0.075 | 3.288125349 |
| 173 | data_only_sensor_regression_control | test | 16 | 2.066829768 | 0.075 | 3.282353949 |
| 173 | fixed_nominal_source_closure_control | val | 16 | 0.7382018019 | 0.8184535147 | 1.794732134 |
| 173 | fixed_nominal_source_closure_control | test | 16 | 0.7072683129 | 0.823379128 | 1.748596414 |
| 173 | grid_least_squares_source_closure_control | val | 16 | 0.0261214975 | 0.032601205 | 0.0855214306 |
| 173 | grid_least_squares_source_closure_control | test | 16 | 0.0266539018 | 0.0216704657 | 0.0787602577 |
| 173 | no_closure_source_control | val | 16 | 0.0282112666 | 0.075 | 0.1199857796 |
| 173 | no_closure_source_control | test | 16 | 0.028989754 | 0.075 | 0.1218050729 |
| 173 | posterior_only_calibrated_bayesian_no_neural | val | 16 | 0.0241648982 | 0.0321381578 | 0.0650077802 |
| 173 | posterior_only_calibrated_bayesian_no_neural | test | 16 | 0.0261639107 | 0.0207199662 | 0.0597568249 |
| 173 | wrong_source_prior_control | val | 16 | 0.7740322404 | 0.8094236169 | 1.817170643 |
| 173 | wrong_source_prior_control | test | 16 | 0.7739944024 | 0.8272395165 | 1.834717502 |
