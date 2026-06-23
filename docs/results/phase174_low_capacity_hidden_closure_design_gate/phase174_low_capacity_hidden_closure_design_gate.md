# Phase 174 Low-Capacity Hidden-Closure Design Gate

## Gate
- Status: `phase174_low_capacity_hidden_closure_design_ready_phase175_low_capacity_smoke`
- Candidate low-capacity route: `low_capacity_explicit_latent_hidden_closure_head`
- Phase 175 low-capacity smoke allowed: `true`
- Phase 174 model training allowed: `false`
- Phase 175 training allowed now: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
This is a design gate only. It translates the Phase 173 synthetic trainable-latent positive into a future bounded low-capacity smoke protocol, but it does not execute training and does not support AM-Bench, Bayesian PINN, GCN, CNN, operator, or A100-80GB claims.

## Design
| design_id | component | decision | bound | phase173_evidence | opens_training_now |
| --- | --- | --- | --- | --- | --- |
| P174-DESIGN-001 | task_scope | synthetic_sparse_sensor_inverse_heat_only | no AM-Bench, no NIST AMMT, no external raw data | Phase 173 positive is synthetic and NumPy-only. | false |
| P174-DESIGN-002 | candidate_mechanism | low_capacity_explicit_latent_closure_head | two explicit source latents plus one bounded closure head; no free residual field | tiny_explicit_latent_hidden_closure_smoke | false |
| P174-DESIGN-003 | initialization | posterior_warm_start_plus_phase173_latent_solution | initialization and prior audit only; not Bayesian neural posterior training | Phase 173 beat posterior-only and Phase 171 controls. | false |
| P174-DESIGN-004 | model_capacity | tiny_residual_free_parameter_head_or_numpy_differentiable_loop | max 3 seeds, max 800 tiny steps in later smoke; Phase 174 does not train | Phase 173 max rounds per start and function evaluations stayed bounded. | false |
| P174-DESIGN-005 | loss_policy | data_fit_plus_closure_interpretability_plus_source_prior_guard | no adaptive sampler retuning, no residual MLP, no test tuning | Phase 173 selected explicit latent route, not sampler route. | false |
| P174-DESIGN-006 | selection_protocol | validation_only_selection_shifted_test_once | must beat Phase 173, Phase 171, posterior, grid, and uniform-start controls | Phase 173 seed stability pass rate was 1.0. | false |
| P174-DESIGN-007 | claim_boundary | may_open_phase175_smoke_but_not_training_now | all Phase 174 training and A100 locks remain false | Phase 173 opened design only. | false |

## Controls
| control_id | control_name | role | required_metric | promotion_requirement |
| --- | --- | --- | --- | --- |
| P174-CTRL-001 | phase173_tiny_explicit_latent_hidden_closure_smoke | must beat or explain the current trainable-latent positive | selection score, field RMSE, closure error | future low-capacity smoke must improve validation or close as unnecessary |
| P174-CTRL-002 | phase171_numpy_closure_head_control | non-trainable mechanism floor | closure_abs_error and selection_score | candidate must preserve Phase 171 closure interpretability gains |
| P174-CTRL-003 | posterior_only_calibrated_bayesian_no_neural | strong no-neural posterior control | field RMSE, closure error, coverage | candidate must improve validation without degrading coverage |
| P174-CTRL-004 | grid_least_squares_source_closure_control | non-Bayesian inverse control | field RMSE and closure error | candidate must beat validation score and avoid test reversal |
| P174-CTRL-005 | no_closure_source_control | closure necessity control | field RMSE and closure error | candidate must show closure term remains necessary |
| P174-CTRL-006 | wrong_source_prior_control | hidden-source interpretability control | field RMSE and closure error | candidate must not be solved by wrong source prior |
| P174-CTRL-007 | data_only_tiny_control | tests whether physics/closure adds value | field RMSE, hot q90, gradient q90 | candidate must beat data-only validation score |
| P174-CTRL-008 | uniform_grid_latent_trainable_control | same trainable search budget without posterior/grid starts | selection_score and coverage penalty | candidate must beat or justify this control |
| P174-CTRL-009 | failure_sampler_retrain_block | prevents repeating Phase 167 | must remain non-selected | cannot be the selected route |
| P174-CTRL-010 | seed_stability_control | prevents single-seed promotion | three seeds when cheap | all seeds pass or gate closes |

## Metrics
| metric_id | metric_or_guard | threshold | selection_use | rationale |
| --- | --- | --- | --- | --- |
| P174-METRIC-001 | validation_selection_score | future candidate must improve vs Phase 173 and best control | validation only | prevents test tuning and controls solve-it artifacts |
| P174-METRIC-002 | field_rmse | must not trade closure gain for large field degradation | validation/test audit | keeps the route predictive, not only interpretable |
| P174-METRIC-003 | closure_abs_error | must improve or match Phase 173 within tolerance | validation/test audit | preserves hidden-closure interpretability |
| P174-METRIC-004 | coverage90_mean | must remain in [0.65, 1.0] if intervals are reported | validation/test audit | keeps calibrated inference language bounded |
| P174-METRIC-005 | test_reversal_ratio | future shifted-test ratio <= 1.02 vs best control | test once after validation selection | closes unstable split behavior |
| P174-METRIC-006 | hot_q90_gradient_q90_rmse | must not degrade both region metrics | validation/test audit | retains thermal hot/gradient relevance |
| P174-METRIC-007 | budget_and_seed_stability | max 3 seeds and bounded tiny steps; all seeds pass | gate audit | prevents compute creep and single-seed promotion |

## Compute
| resource_id | resource | allowed_now | limit | escalation_rule |
| --- | --- | --- | --- | --- |
| P174-COMPUTE-001 | local_cpu_numpy_or_tiny_torch | false | design only in Phase 174; Phase 175 may run only if explicitly opened | prefer NumPy/local smoke before any GPU use |
| P174-COMPUTE-002 | A800_40GB | false | reproduce design only now; future smoke must stay tiny | allowed only by a later Phase 175 runner |
| P174-COMPUTE-003 | A100_SXM4_80GB | false | not justified | request only after a seed-positive branch hits measured 40GB blockage |

## Promotion Rules
| rule_id | rule | threshold | failure_action |
| --- | --- | --- | --- |
| P174-PROMOTE-001 | phase175_entry | Phase 174 design gate passes with all training locks false | repair design before training |
| P174-PROMOTE-002 | low_capacity_validation_gain | future candidate improves validation score vs Phase 173 and controls | close as unnecessary low-capacity mechanism |
| P174-PROMOTE-003 | closure_interpretability | future closure error improves or matches Phase 173 without field degradation | write as predictive diagnostic only |
| P174-PROMOTE-004 | test_reversal | future shifted-test reversal ratio <=1.02 | close and do not tune test |
| P174-PROMOTE-005 | claim_boundary | future result remains synthetic until separate AM data gate | do not claim AM-Bench, Bayesian PINN, GNN/CNN/operator, or 80GB success |

## Risks
| risk_id | risk | guard | closure_action |
| --- | --- | --- | --- |
| P174-RISK-001 | low-capacity model adds no value over Phase 173 explicit latents | Phase 173 candidate is a required control | close if validation improvement is absent |
| P174-RISK-002 | overclaiming Bayesian PINN | posterior warm-start is initialization only | write as latent mechanism design only |
| P174-RISK-003 | sampler retuning repeats Phase 167 failure | failure_sampler_retrain_block remains required | do not select sampler route |
| P174-RISK-004 | capacity or compute creep | Phase 174 is design-only and future budget is bounded | stop instead of scaling |
| P174-RISK-005 | synthetic-to-AM overreach | no raw data and no AM claim | require later baseline-first AM data gate |
