# Phase 173 Trainable Hidden-Closure Low-Budget Smoke

## Gate
- Status: `phase173_trainable_hidden_closure_low_budget_smoke_ready_phase174_low_capacity_hidden_closure_design`
- Selected variant: `tiny_explicit_latent_hidden_closure_smoke`
- Best control variant: `uniform_grid_latent_trainable_control`
- Validation score gain vs best control: `0.0166680534`
- Phase 171 validation score gain: `0.0372880265`
- Phase 171 test closure gain: `0.0032139974`
- Phase 174 low-capacity hidden-closure design allowed: `true`
- Phase 173 model training allowed: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
This is a tiny synthetic trainable-latent smoke. A positive gate means continuous explicit source/closure latents beat the Phase 171 closure head and strong posterior/grid controls under validation-only selection. It is not AM-Bench evidence, not Bayesian PINN training, not a GCN/CNN/operator route, and not an A100-80GB justification.

## Variants
| variant_id | family | source_estimator | closure_estimator | executed | is_control | trainable | description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny_explicit_latent_hidden_closure_smoke | mechanism_candidate | continuous_center_width_coordinate_search | train_split_calibrated_optimized_closure_head | true | false | true | bounded trainable explicit source/closure latent smoke |
| phase171_numpy_closure_head_control | control | phase169_calibrated_posterior_center_width | phase171_train_split_linear_closure_head | true | true | false | required Phase 171 non-trainable closure-head control |
| posterior_only_calibrated_bayesian_no_neural | control | phase169_calibrated_posterior_center_width | posterior_point_estimate | true | true | false | strong no-neural calibrated posterior control |
| grid_least_squares_source_closure_control | control | coarse_grid_best_sse_center_width | least_squares_closure_coefficient | true | true | false | required non-Bayesian inverse grid control |
| no_closure_source_control | control | grid_search_without_closure_term | zero_closure | true | true | false | tests whether the hidden closure remains necessary |
| wrong_source_prior_control | control | deliberately_shifted_source_prior | least_squares_under_wrong_source | true | true | false | tests whether a wrong source prior can solve the task |
| data_only_tiny_control | control | polynomial_sensor_regression_no_physics | zero_closure | true | true | false | NumPy proxy for the registered data-only tiny control |
| uniform_grid_latent_trainable_control | control | single_nominal_start_coordinate_search | train_split_calibrated_optimized_closure_head | true | true | true | uniform/nominal-start trainable latent control |
| failure_sampler_retrain_block | blocked_control | not_executed | not_executed | false | true | false | registered block against repeating the Phase 167 sampler route |

## Summary Metrics
| variant_id | family | split | seed_count | case_count | field_rmse_mean | hot_q90_rmse_mean | gradient_q90_rmse_mean | closure_abs_error_mean | coverage90_mean | selection_score_mean | selection_score_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data_only_tiny_control | control | val | 3 | 48 | 2.066629422 | 4.347394932 | 2.750988124 | 0.075 | 0 | 3.287457221 | 0.655957135 |
| data_only_tiny_control | control | test | 3 | 48 | 2.066800347 | 4.321894152 | 2.745832829 | 0.075 | 0 | 3.28201246 | 0.6458761195 |
| grid_least_squares_source_closure_control | control | val | 3 | 48 | 0.0263732727 | 0.0444826802 | 0.0616376905 | 0.0302465546 | 0 | 0.0841184938 | 0.0231563872 |
| grid_least_squares_source_closure_control | control | test | 3 | 48 | 0.0265095193 | 0.0468583864 | 0.0630571244 | 0.0225972062 | 0 | 0.0791348137 | 0.0254641775 |
| no_closure_source_control | control | val | 3 | 48 | 0.0280890029 | 0.0453651882 | 0.0629806427 | 0.075 | 0 | 0.1197101048 | 0.031989168 |
| no_closure_source_control | control | test | 3 | 48 | 0.0290625452 | 0.0494574023 | 0.0659084149 | 0.075 | 0 | 0.1217948671 | 0.0370843219 |
| phase171_numpy_closure_head_control | control | val | 3 | 48 | 0.0244252778 | 0.0411811 | 0.0572416905 | 0.0263759868 | 0.8819444444 | 0.0611121014 | 0.0214938526 |
| phase171_numpy_closure_head_control | control | test | 3 | 48 | 0.0256532769 | 0.0455446368 | 0.0612577044 | 0.0172555913 | 0.9375 | 0.0563296682 | 0.0220509477 |
| posterior_only_calibrated_bayesian_no_neural | control | val | 3 | 48 | 0.0244252778 | 0.0411811 | 0.0572416905 | 0.0278905504 | 0.8819444444 | 0.0622480241 | 0.0252675588 |
| posterior_only_calibrated_bayesian_no_neural | control | test | 3 | 48 | 0.0256532769 | 0.0455446368 | 0.0612577044 | 0.0204193369 | 0.9375 | 0.0587024774 | 0.0254347314 |
| tiny_explicit_latent_hidden_closure_smoke | mechanism_candidate | val | 3 | 48 | 0.006271794 | 0.0085512203 | 0.0089651896 | 0.0160014312 | 0.8819444444 | 0.0238240749 | 0.0111767491 |
| tiny_explicit_latent_hidden_closure_smoke | mechanism_candidate | test | 3 | 48 | 0.0059616267 | 0.0074503416 | 0.0087659511 | 0.0140415939 | 0.9375 | 0.0213594856 | 0.0091051469 |
| uniform_grid_latent_trainable_control | control | val | 3 | 48 | 0.006272017 | 0.0085552192 | 0.0089679272 | 0.0154830329 | 0 | 0.0404921283 | 0.0106266041 |
| uniform_grid_latent_trainable_control | control | test | 3 | 48 | 0.0059694886 | 0.0074575433 | 0.0087938693 | 0.013016002 | 0 | 0.0381023857 | 0.0089839272 |
| wrong_source_prior_control | control | val | 3 | 48 | 0.7759218074 | 1.241204414 | 1.686848984 | 0.8082501097 | 0 | 1.819035171 | 0.2635866881 |
| wrong_source_prior_control | control | test | 3 | 48 | 0.7737491665 | 1.259373681 | 1.676945851 | 0.816953174 | 0 | 1.826033368 | 0.2564821358 |

## Seed Summary
| seed | variant_id | split | case_count | field_rmse_mean | closure_abs_error_mean | selection_score_mean |
| --- | --- | --- | --- | --- | --- | --- |
| 171 | data_only_tiny_control | val | 16 | 2.066617329 | 0.075 | 3.286978925 |
| 171 | data_only_tiny_control | test | 16 | 2.066771954 | 0.075 | 3.281779511 |
| 171 | grid_least_squares_source_closure_control | val | 16 | 0.0266390154 | 0.0282463834 | 0.0830642092 |
| 171 | grid_least_squares_source_closure_control | test | 16 | 0.0264025016 | 0.0257209561 | 0.0812317175 |
| 171 | no_closure_source_control | val | 16 | 0.0283440386 | 0.075 | 0.1200962177 |
| 171 | no_closure_source_control | test | 16 | 0.0291535945 | 0.075 | 0.121912791 |
| 171 | phase171_numpy_closure_head_control | val | 16 | 0.0245230333 | 0.0250702117 | 0.0604787397 |
| 171 | phase171_numpy_closure_head_control | test | 16 | 0.0246828665 | 0.0178274375 | 0.0554067997 |
| 171 | posterior_only_calibrated_bayesian_no_neural | val | 16 | 0.0245230333 | 0.0263031789 | 0.0614034651 |
| 171 | posterior_only_calibrated_bayesian_no_neural | test | 16 | 0.0246828665 | 0.0210502497 | 0.0578239088 |
| 171 | tiny_explicit_latent_hidden_closure_smoke | val | 16 | 0.006663995 | 0.0183809643 | 0.026613519 |
| 171 | tiny_explicit_latent_hidden_closure_smoke | test | 16 | 0.0058561301 | 0.0151544788 | 0.0221824065 |
| 171 | uniform_grid_latent_trainable_control | val | 16 | 0.0066640643 | 0.017654697 | 0.0429012018 |
| 171 | uniform_grid_latent_trainable_control | test | 16 | 0.0058593134 | 0.0121950494 | 0.0373030865 |
| 171 | wrong_source_prior_control | val | 16 | 0.7767623361 | 0.7971526025 | 1.812249076 |
| 171 | wrong_source_prior_control | test | 16 | 0.7736275975 | 0.8084723704 | 1.819380889 |
| 172 | data_only_tiny_control | val | 16 | 2.066613174 | 0.075 | 3.28726739 |
| 172 | data_only_tiny_control | test | 16 | 2.066799318 | 0.075 | 3.28190392 |
| 172 | grid_least_squares_source_closure_control | val | 16 | 0.0263593053 | 0.0298920754 | 0.0837698416 |
| 172 | grid_least_squares_source_closure_control | test | 16 | 0.0264721547 | 0.0204001969 | 0.077412466 |
| 172 | no_closure_source_control | val | 16 | 0.0277117033 | 0.075 | 0.119048317 |
| 172 | no_closure_source_control | test | 16 | 0.0290442869 | 0.075 | 0.1216667374 |
| 172 | phase171_numpy_closure_head_control | val | 16 | 0.0245879018 | 0.0248454496 | 0.0600441783 |
| 172 | phase171_numpy_closure_head_control | test | 16 | 0.0261130535 | 0.0166381303 | 0.0563894501 |
| 172 | posterior_only_calibrated_bayesian_no_neural | val | 16 | 0.0245879018 | 0.0252303145 | 0.0603328269 |
| 172 | posterior_only_calibrated_bayesian_no_neural | test | 16 | 0.0261130535 | 0.0194877949 | 0.0585266985 |
| 172 | tiny_explicit_latent_hidden_closure_smoke | val | 16 | 0.0055883085 | 0.0126834291 | 0.0199192285 |
| 172 | tiny_explicit_latent_hidden_closure_smoke | test | 16 | 0.0059120355 | 0.0168341667 | 0.0233378705 |
| 172 | uniform_grid_latent_trainable_control | val | 16 | 0.0055923404 | 0.0126816573 | 0.0372583387 |
| 172 | uniform_grid_latent_trainable_control | test | 16 | 0.0059138408 | 0.0161140658 | 0.0404636374 |
| 172 | wrong_source_prior_control | val | 16 | 0.7769708457 | 0.8181741098 | 1.827685794 |
| 172 | wrong_source_prior_control | test | 16 | 0.7736254997 | 0.815147635 | 1.824001714 |
| 173 | data_only_tiny_control | val | 16 | 2.066657765 | 0.075 | 3.288125349 |
| 173 | data_only_tiny_control | test | 16 | 2.066829768 | 0.075 | 3.282353949 |
| 173 | grid_least_squares_source_closure_control | val | 16 | 0.0261214975 | 0.032601205 | 0.0855214306 |
| 173 | grid_least_squares_source_closure_control | test | 16 | 0.0266539018 | 0.0216704657 | 0.0787602577 |
| 173 | no_closure_source_control | val | 16 | 0.0282112666 | 0.075 | 0.1199857796 |
| 173 | no_closure_source_control | test | 16 | 0.028989754 | 0.075 | 0.1218050729 |
| 173 | phase171_numpy_closure_head_control | val | 16 | 0.0241648982 | 0.029212299 | 0.0628133861 |
| 173 | phase171_numpy_closure_head_control | test | 16 | 0.0261639107 | 0.0173012061 | 0.0571927549 |
| 173 | posterior_only_calibrated_bayesian_no_neural | val | 16 | 0.0241648982 | 0.0321381578 | 0.0650077802 |
| 173 | posterior_only_calibrated_bayesian_no_neural | test | 16 | 0.0261639107 | 0.0207199662 | 0.0597568249 |
| 173 | tiny_explicit_latent_hidden_closure_smoke | val | 16 | 0.0065630785 | 0.0169399003 | 0.0249394772 |
| 173 | tiny_explicit_latent_hidden_closure_smoke | test | 16 | 0.0061167146 | 0.0101361362 | 0.0185581798 |
| 173 | uniform_grid_latent_trainable_control | val | 16 | 0.0065596464 | 0.0161127444 | 0.0413168442 |
| 173 | uniform_grid_latent_trainable_control | test | 16 | 0.0061353116 | 0.0107388909 | 0.0365404332 |
| 173 | wrong_source_prior_control | val | 16 | 0.7740322404 | 0.8094236169 | 1.817170643 |
| 173 | wrong_source_prior_control | test | 16 | 0.7739944024 | 0.8272395165 | 1.834717502 |
