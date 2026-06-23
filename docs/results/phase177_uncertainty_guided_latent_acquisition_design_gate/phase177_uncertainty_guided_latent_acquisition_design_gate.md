# Phase 177 Uncertainty-Guided Latent Acquisition Design Gate

## Gate
- Status: `phase177_uncertainty_guided_latent_acquisition_design_ready_phase178_no_training_smoke`
- Candidate mechanism: `posterior_ensemble_uncertainty_guided_latent_acquisition`
- Materially different from Phase 175: `true`
- Phase 178 no-training acquisition smoke allowed: `true`
- Phase 177 model training allowed: `false`
- Phase 178 training allowed now: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
This is a design-only gate. It pivots away from the closed Phase 175 low-capacity head toward uncertainty-guided sparse observation acquisition for explicit source/closure latents. It does not train a PINN, does not read AM data, and does not open graph, CNN, operator, or 80GB claims.

## Evidence
| evidence_id | source | finding | design_implication | claim_boundary |
| --- | --- | --- | --- | --- |
| P177-EVID-001 | Phase 176 evidence refresh | synthetic hidden-closure claims are allowed only narrowly; Phase 177 design allowed=True | The next route must be materially different and design-only. | No second-paper core claim or training permission yet. |
| P177-EVID-002 | Phase 169 calibrated posterior | validation gain 0.0707838724 with coverage 0.9375 | Use posterior uncertainty as a candidate acquisition signal. | Still no Bayesian neural training claim. |
| P177-EVID-003 | Phase 173 explicit latent smoke | explicit latent route beat uniform_grid_latent_trainable_control with validation gain 0.0166680534 | Use latent disagreement, not a larger head, to choose new observations. | Tiny synthetic result only. |
| P177-EVID-004 | Phase 175 closure | low-capacity candidate low_capacity_explicit_latent_hidden_closure_head closed with gain -0.0004223396 | Block low-capacity head retuning and avoid capacity expansion. | The low-capacity head is a negative route. |
| P177-EVID-005 | Phase 167 sampler-to-PINN closure | sampler coverage alone did not become a trained PINN gain. | Acquisition must target posterior contraction and latent error, not only hot-gradient coverage. | No adaptive PINN training claim. |
| P177-EVID-006 | Phase 148/151 route closures | current graph/path and dense operator routes remain blocked. | Do not add GCN/CNN/operator machinery in this design gate. | No graph/operator success claim. |

## Mechanism
| mechanism_id | component | design_choice | bound | materially_different_from_phase175 | opens_training_now |
| --- | --- | --- | --- | --- | --- |
| P177-MECH-001 | task_scope | controlled_synthetic_inverse_heat_sparse_observation_acquisition | no AM-Bench, no NIST AMMT, no registered camera target, no raw data | true | false |
| P177-MECH-002 | candidate_route | posterior_ensemble_uncertainty_guided_latent_acquisition | choose new observation locations before any model update; no low-capacity head | true | false |
| P177-MECH-003 | uncertainty_source | combine_phase169_posterior_variance_and_phase173_latent_disagreement | uncertainty is computed from train/validation-safe synthetic sensors only | true | false |
| P177-MECH-004 | objective | maximize_expected_closure_posterior_contraction_under_sparse_budget | registered acquisition budget; no test-target or shifted-test peeking | true | false |
| P177-MECH-005 | physics_guard | retain_explicit_center_width_closure_latents_with_bounded_ranges | no free residual field, no residual MLP, no density proxy retuning | true | false |
| P177-MECH-006 | selection_protocol | validation_only_acquisition_policy_selection_shifted_test_once | Phase 178 may run only a no-training acquisition utility smoke if opened | true | false |
| P177-MECH-007 | claim_boundary | mechanism_design_not_model_training | all training/A100/80GB locks remain false | true | false |

## Acquisition Policies
| acquisition_id | policy | input_signal | selection_rule | leakage_guard | candidate_for_phase178 |
| --- | --- | --- | --- | --- | --- |
| P177-ACQ-001 | posterior_entropy_reduction_candidate | Phase 169 calibrated posterior variance over center, width, closure coefficient | rank candidate sensor locations by expected entropy reduction | candidate pool generated from train/validation synthetic coordinates only | true |
| P177-ACQ-002 | latent_ensemble_disagreement_candidate | Phase 173 explicit latent ensemble prediction disagreement | rank by disagreement in field and closure-sensitive regions | no shifted-test error or target residual is used | true |
| P177-ACQ-003 | hybrid_uncertainty_hot_gradient_candidate | posterior contraction score plus fixed hot/gradient quota | allocate a bounded quota to high-gradient regions after uncertainty ranking | hot/gradient field computed analytically in synthetic train split only | true |
| P177-ACQ-004 | uniform_budget_control | none | same number of new observation points on a uniform grid | same candidate pool and budget | false |
| P177-ACQ-005 | random_budget_control | seeded random coordinate order | same number of new observation points by deterministic random seed | same candidate pool and budget | false |
| P177-ACQ-006 | oracle_test_target_block | test error, shifted-test labels, or target residuals | prohibited | any use closes the gate | false |

## Controls
| control_id | control_name | role | required_metric | promotion_requirement |
| --- | --- | --- | --- | --- |
| P177-CTRL-001 | phase173_tiny_explicit_latent_hidden_closure_smoke | current strongest synthetic mechanism floor | selection score, field RMSE, closure error | future acquisition smoke must improve latent/closure metrics over this floor |
| P177-CTRL-002 | phase169_posterior_only_calibrated_bayesian_no_neural | uncertainty baseline | posterior contraction, interval coverage, closure RMSE | candidate must improve uncertainty or closure recovery beyond posterior-only |
| P177-CTRL-003 | grid_least_squares_source_closure_control | strong deterministic inverse control | joint normalized RMSE and closure coefficient error | candidate must beat validation score or close as solved |
| P177-CTRL-004 | uniform_budget_acquisition_control | same budget acquisition baseline | posterior contraction and validation selection score | candidate must beat uniform acquisition |
| P177-CTRL-005 | random_budget_acquisition_control | seeded random observation baseline | mean and worst-seed contraction | candidate must beat random mean and avoid worst-seed collapse |
| P177-CTRL-006 | no_new_observation_control | tests whether acquisition itself adds value | posterior contraction and closure error delta | candidate must improve beyond no-acquisition |
| P177-CTRL-007 | phase175_low_capacity_head_retrain_block | blocks the closed Phase 175 route | must remain non-selected | any reuse of the low-capacity head closes the design |
| P177-CTRL-008 | failure_sampler_retrain_block | blocks Phase 167 sampler retuning | must remain diagnostic-only | candidate cannot be only hot-gradient sampler retuning |
| P177-CTRL-009 | oracle_target_leakage_block | prevents test/target leakage | must be absent from acquisition features | any oracle target use closes the branch |
| P177-CTRL-010 | seed_stability_control | prevents single-seed acquisition promotion | three deterministic seeds when Phase 178 runs | mean gain positive and worst seed within registered tolerance |

## Metrics
| metric_id | metric_or_guard | split_use | threshold | rationale |
| --- | --- | --- | --- | --- |
| P177-METRIC-001 | validation_selection_score | validation only | future acquisition candidate must beat best control | prevents promotion from design text alone |
| P177-METRIC-002 | posterior_entropy_or_variance_contraction | validation audit | candidate contraction gain > uniform and random controls | tests the acquisition mechanism directly |
| P177-METRIC-003 | closure_abs_error_delta | validation/test audit | must improve or match Phase 173 closure error | keeps hidden-closure recovery as the target |
| P177-METRIC-004 | field_rmse_delta | validation/test audit | must not degrade field RMSE while improving uncertainty | prevents pure uncertainty-only gains |
| P177-METRIC-005 | coverage90_mean | validation/test audit | remain in [0.65, 1.0] if intervals are reported | bounds Bayesian interpretation |
| P177-METRIC-006 | acquisition_diversity | artifact audit | no duplicate points; bounded boundary fraction | prevents collapse into one local hot spot |
| P177-METRIC-007 | test_reversal_ratio | shifted test once | <=1.02 vs best acquisition control | prevents validation-only overfit |
| P177-METRIC-008 | no_training_no_raw_data_lock | gate audit | all training/raw-data/A100-80GB locks false | keeps Phase 177 as design-only |

## Compute
| resource_id | resource | phase177_use | limit | training_allowed_now | escalation_rule |
| --- | --- | --- | --- | --- | --- |
| P177-COMPUTE-001 | local_python | design artifact generation and tests | no torch training, no raw data | false | none |
| P177-COMPUTE-002 | future_phase178_no_training_smoke | planned only | analytic/synthetic acquisition utility only if Phase 177 passes | false | must be implemented in Phase 178, not Phase 177 |
| P177-COMPUTE-003 | A800_40GB | reproduce design artifacts and tests only | no training; server optional for reproduction | false | do not use as evidence for larger hardware |
| P177-COMPUTE-004 | A100_SXM4_80GB | not allowed | not justified | false | request only after later seed-positive branch hits measured 40GB blockage |

## Promotion Rules
| rule_id | rule | threshold | failure_action |
| --- | --- | --- | --- |
| P177-PROMOTE-001 | phase178_entry | Phase 177 design passes with all training/A100 locks false | repair design before any smoke |
| P177-PROMOTE-002 | materially_different_route | candidate route is posterior/ensemble acquisition, not Phase 175 low-capacity head | close as retune attempt |
| P177-PROMOTE-003 | acquisition_utility_gain | future no-training smoke must improve posterior contraction vs uniform/random/no-acquisition | close as solved by controls |
| P177-PROMOTE-004 | closure_and_field_guard | future smoke must improve closure without field/test reversal | write as diagnostic only |
| P177-PROMOTE-005 | claim_boundary | future smoke remains synthetic/no-training until a separate gate permits more | do not claim Bayesian PINN, AM-Bench, graph/operator, or 80GB success |

## Risks
| risk_id | risk | guard | closure_action |
| --- | --- | --- | --- |
| P177-RISK-001 | retuning the closed Phase 175 low-capacity head | explicit low-capacity head retrain block | close if selected route uses the Phase 175 head |
| P177-RISK-002 | sampler coverage repeats Phase 167 without model utility | posterior contraction and closure error are primary metrics | close if only hot-gradient coverage improves |
| P177-RISK-003 | oracle target leakage in acquisition | candidate pool excludes shifted-test labels and target residuals | close immediately on leakage |
| P177-RISK-004 | Bayesian overclaim | coverage and calibration are audit metrics only | do not write full Bayesian PINN success |
| P177-RISK-005 | synthetic-to-AM overreach | no raw AM data and no AM claim | require separate baseline-first AM data gate |
| P177-RISK-006 | compute creep | all Phase 177 training/A100/80GB locks are false | stop instead of scaling |
