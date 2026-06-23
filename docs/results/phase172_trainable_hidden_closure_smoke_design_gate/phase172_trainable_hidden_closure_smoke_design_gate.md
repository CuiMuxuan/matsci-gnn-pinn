# Phase 172 Trainable Hidden-Closure Smoke Design Gate

## Gate
- Status: `phase172_trainable_hidden_closure_smoke_design_ready_phase173_low_budget_trainable_smoke`
- Candidate trainable route: `tiny_explicit_latent_hidden_closure_smoke`
- Phase 173 low-budget trainable smoke allowed: `true`
- Phase 172 model training allowed: `false`
- Phase 173 training allowed now: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
This is a design gate only. It may allow a later tiny trainable synthetic smoke, but it does not execute training and does not support AM-Bench, Bayesian PINN, GCN, CNN, operator, or A100-80GB claims.

## Design
| design_id | component | decision | bound | phase171_evidence | opens_training_now |
| --- | --- | --- | --- | --- | --- |
| P172-DESIGN-001 | task_scope | synthetic_sparse_sensor_inverse_heat_only | no AM-Bench, no NIST AMMT, no external raw data | Phase 171 positive is synthetic and NumPy-only. | false |
| P172-DESIGN-002 | trainable_mechanism | tiny_explicit_latent_head_for_center_width_closure | three scalar latents plus bounded correction head; no residual MLP | calibrated_hidden_source_closure_parameter_head | false |
| P172-DESIGN-003 | initialization | initialize_from_phase171_calibrated_closure_head_and_phase169_posterior | initialization only; not a Bayesian neural posterior | posterior closure gain was positive across seeds. | false |
| P172-DESIGN-004 | model_budget | tiny_width_16_or_numpy_differentiable_optimizer_smoke | max 3 seeds, max 600 steps, no AM data, no GPU requirement | Phase 171 runner is small and deterministic. | false |
| P172-DESIGN-005 | sampler_policy | uniform_sensor_and_collocation_primary | failure-informed sampler remains blocked except as non-selected audit row | Phase 167 sampler retuning failed; Phase 171 blocked it. | false |
| P172-DESIGN-006 | selection_protocol | validation_only_selection_shifted_test_once | no test tuning, no single-seed promotion | Phase 171 seed stability pass rate was positive. | false |
| P172-DESIGN-007 | claim_boundary | design_may_open_phase173_smoke_but_not_training_now | all Phase 172 training and A100 locks remain false | Phase 171 opened design only. | false |

## Controls
| control_id | control_name | role | required_metric | promotion_requirement |
| --- | --- | --- | --- | --- |
| P172-CTRL-001 | phase171_numpy_closure_head | must beat or complement the non-trainable mechanism | selection score, closure error, field RMSE | trainable smoke must improve validation score or close as unnecessary |
| P172-CTRL-002 | posterior_only_calibrated_bayesian_no_neural | strong no-neural posterior control | field RMSE, closure error, coverage | candidate must preserve field RMSE and improve interpretable closure |
| P172-CTRL-003 | grid_least_squares_source_closure_control | non-Bayesian inverse control | field RMSE, closure error | candidate must beat validation score and avoid test reversal |
| P172-CTRL-004 | no_closure_source_control | closure necessity control | field RMSE and closure error | candidate must show closure term remains necessary |
| P172-CTRL-005 | data_only_tiny_control | tests whether physics/closure adds value | field RMSE, hot q90, gradient q90 | candidate must beat data-only validation score |
| P172-CTRL-006 | wrong_source_prior_control | hidden-source interpretability control | field RMSE and closure error | candidate must not be solved by wrong source prior |
| P172-CTRL-007 | uniform_grid_pinn_control | future tiny trained baseline if Phase 173 executes | same trainable budget | candidate must beat or match it on validation |
| P172-CTRL-008 | failure_sampler_retrain_block | prevents repeating Phase 167 | must remain non-selected | cannot be the selected route |
| P172-CTRL-009 | seed_stability_control | prevents single-seed promotion | three seeds when cheap | all seeds pass or gate closes |

## Losses
| loss_id | loss_or_metric | weight_or_guard | selection_use | rationale |
| --- | --- | --- | --- | --- |
| P172-LOSS-001 | sensor_data_loss | primary data fit | train only | keeps the trainable smoke tied to observed sparse sensors |
| P172-LOSS-002 | heat_residual_with_explicit_closure | bounded registered range [0.01, 0.10] | train only | tests a physical closure mechanism without a free residual MLP |
| P172-LOSS-003 | latent_prior_penalty | posterior warm-start prior, not full Bayesian training | train audit | uses Phase 171 inference as a bounded prior |
| P172-LOSS-004 | validation_selection_score | candidate must beat controls | validation only | prevents test tuning |
| P172-LOSS-005 | closure_abs_error | must improve vs posterior-only | validation and test audit | preserves interpretability |
| P172-LOSS-006 | coverage90_mean | must remain in [0.65, 1.0] if intervals are reported | validation/test audit | keeps Bayesian language calibrated |
| P172-LOSS-007 | hot_q90_gradient_q90_rmse | must not degrade both region metrics | validation/test audit | retains thermal hot/gradient relevance |

## Compute
| resource_id | resource | allowed_now | limit | escalation_rule |
| --- | --- | --- | --- | --- |
| P172-COMPUTE-001 | local_cpu_numpy_or_tiny_torch | false | design only in Phase 172; Phase 173 may run if explicitly opened | prefer local if torch import works; otherwise A800 tiny smoke |
| P172-COMPUTE-002 | A800_40GB | false | reproduce design only now; future smoke must stay tiny | allowed only by a later Phase 173 runner |
| P172-COMPUTE-003 | A100_SXM4_80GB | false | not justified | request only after a seed-positive branch hits measured 40GB blockage |

## Promotion Rules
| rule_id | rule | threshold | failure_action |
| --- | --- | --- | --- |
| P172-PROMOTE-001 | phase173_entry | Phase 172 design gate passes with all training locks false | repair design before training |
| P172-PROMOTE-002 | trainable_validation_gain | future trainable candidate validation score improves vs Phase 171 and posterior-only controls | close as unnecessary trainable mechanism |
| P172-PROMOTE-003 | test_reversal | future shifted-test reversal ratio <=1.05 | close and do not tune test |
| P172-PROMOTE-004 | closure_interpretability | future closure error improves or matches Phase 171 without field degradation | write as predictive diagnostic only |
| P172-PROMOTE-005 | claim_boundary | future result remains synthetic until separate AM data gate | do not claim AM-Bench, Bayesian PINN, or full GNN-PINN success |

## Risks
| risk_id | risk | guard | closure_action |
| --- | --- | --- | --- |
| P172-RISK-001 | trainable smoke adds no value over Phase 171 | Phase 171 closure head is a required control | close if trainable variant does not improve validation |
| P172-RISK-002 | overclaiming Bayesian PINN | posterior warm-start is not neural posterior inference | write as initialized latent model only |
| P172-RISK-003 | sampler retuning repeats Phase 167 failure | failure_sampler_retrain_block remains required | do not select sampler route |
| P172-RISK-004 | compute creep | no training in Phase 172 and future max budget registered | stop instead of scaling |
| P172-RISK-005 | synthetic-to-AM overreach | no raw data and no AM claim | require later AM data gate |
