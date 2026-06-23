# Phase 168 Hidden-Source/Closure Redesign Gate

## Gate
- Status: `phase168_hidden_source_closure_redesign_ready_phase169_identifiability_gate`
- Selected next route: `hidden_source_closure_identifiability_gate`
- Phase 169 identifiability gate allowed: `true`
- Retrain same sampler route allowed: `false`
- Phase 168 model training allowed: `false`
- Phase 169 training allowed now: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
Phase 168 redirects the PINN route away from sampler retuning. The next candidate must first prove hidden-source/closure identifiability against non-neural controls before any neural training is reconsidered.

## Evidence
| evidence_id | source_phase | finding | metric_anchor | route_consequence |
| --- | --- | --- | --- | --- |
| P168-EVID-001 | phase167 | validation_selected_uniform_control | selected=uniform_grid_pinn; best_control=uniform_grid_pinn | do_not_continue_same_failure_sampler_pinn |
| P168-EVID-002 | phase167 | adaptive_sampler_model_score_worse_than_uniform | adaptive_val_score=0.0980784314; uniform_val_score=0.0439669985 | sampler_coverage_alone_is_not_model_mechanism |
| P168-EVID-003 | phase167 | wrong_source_prior_control_failed_strongly | wrong_prior_val_score=0.2247392356; data_only_val_score=0.0962753229 | hidden_source_or_closure_identifiability_is_the_next_physical_pain_point |
| P168-EVID-004 | phase166_literature | bayesian_inverse_and_parametric_heat_pinn_references_remain_relevant | B-PINN, EKI-BPINN, AM thermal PINN, LPBF parametric PINN | use_for_identifiability_design_not_immediate_training |

## Routes
| route_id | route_name | mechanism | decision | why_after_phase167 | opens_training_now | opens_a100_now |
| --- | --- | --- | --- | --- | --- | --- |
| P168-ROUTE-001 | hidden_source_closure_identifiability_gate | infer low-dimensional moving-source and residual-closure parameters before training a PINN | selected_for_phase169_no_training_gate | Phase 167 shows sampler choice alone loses to uniform control, while wrong source prior is strongly harmful; the physical pain point is source/closure identifiability. | false | false |
| P168-ROUTE-002 | adaptive_loss_balancing_gate | balance data/residual/source losses after a source/closure candidate is identifiable | defer_until_phase169_identifiability_passes | loss balancing without a better source/closure target would retune the same failed smoke | false | false |
| P168-ROUTE-003 | lightweight_bayesian_neural_uq | ensemble or EKI-style Bayesian neural posterior for uncertainty | defer_until_deterministic_closure_gap_exists | full Bayesian neural inference is not justified before a deterministic mechanism gap | false | false |
| P168-ROUTE-004 | gcn_or_path_graph_pinn | graph-structured non-grid PDE residual route | blocked_by_prior_path_graph_guard | Phase 148 already closed current path-contact graph evidence | false | false |
| P168-ROUTE-005 | cnn_or_neural_operator_dense_route | fixed-grid dense field residual completion | blocked_by_prior_dense_baseline_guard | Phase 151 found no leakage-safe dense baseline gap | false | false |

## Phase 169 Design Contract
| design_id | component | phase169_requirement | control_or_guard | claim_boundary |
| --- | --- | --- | --- | --- |
| P168-DESIGN-001 | target_physics | recover source center, width, amplitude, diffusivity, and a bounded residual-closure coefficient | grid least-squares, wrong-source prior, and flexible non-physical regression controls | identifiability only; no trained Bayesian PINN claim |
| P168-DESIGN-002 | data_contract | synthetic sparse sensors with train/val/test parameter shifts and fixed random seeds | validation-only route selection and shifted test reporting | no AM-Bench or NIST AMMT data in Phase 169 |
| P168-DESIGN-003 | model_contract | no neural training; compare posterior/grid/linearized/strong-baseline estimators | candidate must beat best non-neural control on validation and avoid test reversal | opens at most a later low-budget mechanism smoke design |
| P168-DESIGN-004 | compute_contract | CPU/local or A800 reproduction only for no-training numerical grids | a100_training_allowed_now=false and a100_80gb_request_now=false | 80GB request requires a later seed-positive branch with measured 40GB blockage |
| P168-DESIGN-005 | failure_exit | close if source/closure parameters are solved by controls or not identifiable | blocking audits must remain explicit | do not proceed to adaptive loss, Bayesian neural, GCN, or CNN/operator training after a failed identifiability gate |
