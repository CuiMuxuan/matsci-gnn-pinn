# Phase 170 Hidden-Closure Mechanism Smoke Design Gate

## Gate
- Status: `phase170_hidden_closure_mechanism_smoke_design_ready_phase171_low_budget_smoke`
- Candidate mechanism: `calibrated_hidden_source_closure_parameter_head_design`
- Phase 171 low-budget hidden-closure smoke allowed: `true`
- Phase 170 model training allowed: `false`
- Phase 171 training allowed now: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
This is a design gate only. It does not execute training, does not reopen the Phase 167 sampler-retuning route, and does not support AM-Bench, Bayesian PINN, GCN, CNN, or neural-operator claims.

## Evidence
| evidence_id | source | finding | design_implication | claim_boundary |
| --- | --- | --- | --- | --- |
| P170-EVID-001 | Phase 167 low-budget synthetic PINN smoke | validation selected uniform_grid_pinn and blocked the failure-informed sampler route | Do not retune the same sampler-PINN route. | Sampler coverage is not model-error evidence. |
| P170-EVID-002 | Phase 169 hidden-source/closure identifiability gate | selected calibrated_bayesian_hidden_source_closure_posterior with validation gain 0.0707838724 | Use hidden source/closure parameters as the next mechanism target. | Still synthetic and no-training. |
| P170-EVID-003 | Phase 169 best control | best control was grid_least_squares_source_closure_control; point estimates are competitive but calibration is absent | Keep grid least-squares as a required non-neural control. | Bayesian language must be tied to calibrated intervals, not point RMSE only. |
| P170-EVID-004 | B-PINN and efficient Bayesian PINN literature | Bayesian inverse-PDE uncertainty is plausible but full neural posterior training is heavier. | Use calibrated posterior warm-start/regularization only in the next smoke. | Do not claim Bayesian PINN training until a later trained gate passes. |
| P170-EVID-005 | AM thermal PINN literature | Thermal PINNs are relevant to AM, but AM data claims require separate source guards. | Keep Phase 171 synthetic before any AM-Bench or NIST AMMT route. | No AM-Bench model claim from Phase 170. |
| P170-EVID-006 | Phase 148 and Phase 151 route closures | Current path-graph/GCN and dense CNN/operator routes remain blocked. | Do not reopen graph or operator mechanisms inside this smoke design. | No GCN/CNN/operator success claim. |

## Mechanism
| mechanism_id | component | design_choice | bound | phase169_evidence | opens_training_now |
| --- | --- | --- | --- | --- | --- |
| P170-MECH-001 | task_scope | synthetic_sparse_sensor_inverse_heat_only | no AM-Bench, no NIST AMMT, no raw data | Phase 169 is synthetic and no-training. | false |
| P170-MECH-002 | hidden_parameter_head | learn_center_shift_source_width_closure_coeff_as_explicit_latents | three scalar latents with bounded physical ranges | center/width/closure were identifiable under sparse sensors. | false |
| P170-MECH-003 | posterior_warm_start | use_calibrated_bayesian_hidden_source_closure_posterior_as_initialization_or_teacher_diagnostic | warm-start/reporting only; no full BNN, HMC, VI, or EKI training now | calibrated posterior beat the non-Bayesian guard on validation score. | false |
| P170-MECH-004 | physics_residual | add_hidden_closure_term_to_heat_residual_with_bounded_weight | closure term is scalar and interpretable; no free residual MLP | no-closure source control was much worse than calibrated closure inference. | false |
| P170-MECH-005 | sampler_policy | uniform_grid_primary_with_optional_fixed_quota_hot_gradient_ablation | no retuning of Phase 167 failure-informed sampler as the main candidate | Phase 167 selected the uniform-grid PINN control. | false |
| P170-MECH-006 | loss_balancing | bounded_adaptive_weights_between_data_residual_and_latent_prior_terms | weights clipped to registered ranges and reported in artifacts | parameter calibration matters; loss weights must not hide parameter error. | false |
| P170-MECH-007 | selection_protocol | validation_only_selection_with_shifted_test_once | no test tuning and no promotion from one seed | Phase 169 used validation-only selection and shifted-test reversal guard. | false |

## Controls
| control_id | control_name | role | required_metric | promotion_requirement |
| --- | --- | --- | --- | --- |
| P170-CTRL-001 | posterior_only_calibrated_bayesian_no_neural | tests whether training adds value beyond Phase 169 inference | field RMSE, closure RMSE, coverage, selection score | trained mechanism must beat or complement posterior-only on validation |
| P170-CTRL-002 | grid_least_squares_source_closure_control | required non-Bayesian inverse control | joint normalized parameter RMSE and field RMSE | candidate must beat validation score and avoid test reversal >1.05 |
| P170-CTRL-003 | no_closure_source_control | tests whether closure term is necessary | closure coefficient RMSE and field RMSE | candidate must reduce validation error and closure error |
| P170-CTRL-004 | uniform_grid_pinn_control | equal-budget trained PINN control from Phase 167 lesson | same optimizer steps, architecture width, and collocation budget | candidate must beat uniform validation score |
| P170-CTRL-005 | data_only_tiny_mlp_no_residual | tests whether physics residual helps | field RMSE, hot q90, gradient q90 | physics candidate must beat data-only on validation |
| P170-CTRL-006 | wrong_source_prior_control | tests hidden-source interpretability | parameter RMSE and field RMSE | candidate must not be solved by a wrong prior |
| P170-CTRL-007 | failure_sampler_retrain_block | prevents repeating Phase 167 failed route | registered as blocked unless used only as fixed ablation | cannot be the main selected candidate in Phase 171 |
| P170-CTRL-008 | seed_stability_control | prevents single-seed promotion | at least three seeds when the smoke remains cheap | mean gain positive and worst seed not worse than best control by >5% |

## Losses And Metrics
| metric_id | metric_or_loss | split_use | guard | rationale |
| --- | --- | --- | --- | --- |
| P170-METRIC-001 | validation_selection_score | validation only | primary selection; lower is better | Keeps model selection off the shifted test split. |
| P170-METRIC-002 | field_rmse | train/val/test | candidate validation gain >=0.03 vs best trained control | Tests actual model error, unlike Phase 169 parameter-only inference. |
| P170-METRIC-003 | closure_coeff_rmse | validation and test | validation <=0.020 and test <=0.025 unless field error gain is clearly larger | Preserves the interpretable closure signal from Phase 169. |
| P170-METRIC-004 | posterior_or_interval_coverage | validation and test | coverage in [0.65, 1.0] for any Bayesian-calibrated report | Prevents overclaiming Bayesian point estimates. |
| P170-METRIC-005 | hot_q90_and_gradient_q90_rmse | validation and test | must not degrade both region metrics while improving only global RMSE | Matches AM thermal-region concerns without using AM data. |
| P170-METRIC-006 | bounded_adaptive_loss_weights | artifact audit | weights must remain within registered ranges | Loss balancing is a mechanism only if weights remain interpretable. |
| P170-METRIC-007 | runtime_and_memory | artifact audit | small smoke must finish without 80GB hardware | A100-SXM4-80GB remains unjustified until measured 40GB blockage. |

## Compute
| resource_id | resource | phase170_use | limit | training_allowed_now | escalation_rule |
| --- | --- | --- | --- | --- | --- |
| P170-COMPUTE-001 | local_python | design artifact generation and tests | no torch training and no raw data | false | none |
| P170-COMPUTE-002 | A800_40GB | reproduce design artifacts and tests only | runner must leave training locks false | false | use for Phase 171 smoke only if that later phase explicitly runs it |
| P170-COMPUTE-003 | future_phase171_small_smoke | planned only | tiny synthetic, three seeds, no AM raw data | false | must be implemented in Phase 171, not Phase 170 |
| P170-COMPUTE-004 | A100_SXM4_80GB | not allowed | not justified | false | request only after a later seed-positive branch has measured 40GB blockage |

## Promotion Rules
| rule_id | rule | threshold | failure_action |
| --- | --- | --- | --- |
| P170-PROMOTE-001 | phase171_entry | Phase 170 gate passes with all training/A100 locks false | repair design before any training |
| P170-PROMOTE-002 | validation_gain | future Phase 171 candidate validation score gain >=0.03 vs best trained control | close as diagnostic |
| P170-PROMOTE-003 | test_reversal | future shifted-test reversal ratio <=1.05 | close and do not tune test |
| P170-PROMOTE-004 | closure_interpretability | future closure coefficient RMSE remains within Phase 169-scale guard | write as predictive diagnostic only, not interpretable mechanism |
| P170-PROMOTE-005 | claim_boundary | future result remains synthetic until separate AM data guard | do not write AM-Bench or general GNN-PINN claim |

## Risks
| risk_id | risk | guard | closure_action |
| --- | --- | --- | --- |
| P170-RISK-001 | repeating Phase 167 sampler route | failure_sampler_retrain_block is a required control | close if the main candidate is sampler retuning |
| P170-RISK-002 | Bayesian overclaim | posterior calibration must be reported separately from neural training | write as calibrated inference diagnostic only |
| P170-RISK-003 | grid-search control solves the problem | grid least-squares remains a required non-neural baseline | do not train if controls leave no modeling gap |
| P170-RISK-004 | synthetic-to-AM overreach | no raw AM data and no AM claim in this route | require a later AM data gate before AM-Bench claims |
| P170-RISK-005 | compute creep | all Phase 170 training and A100/80GB locks remain false | stop and redesign instead of scaling |
