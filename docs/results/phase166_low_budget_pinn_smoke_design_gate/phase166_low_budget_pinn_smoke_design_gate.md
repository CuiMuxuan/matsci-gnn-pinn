# Phase 166 Low-Budget PINN Smoke Design Gate

## Gate
- Status: `phase166_low_budget_pinn_smoke_design_ready_phase167_local_smoke`
- Phase 167 local low-budget PINN smoke allowed: `true`
- Phase 166 model training allowed: `false`
- Phase 167 training allowed now: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
This is a design gate only. It permits a later local synthetic smoke protocol if the design is complete, but it does not execute training or support AM-Bench claims.

## Design
| design_id | component | decision | bound | rationale | opens_training_now |
| --- | --- | --- | --- | --- | --- |
| P166-DESIGN-001 | task_scope | synthetic_1d_moving_heat_source_only | no AM-Bench, no NIST AMMT, no external raw data | The Phase 164/165 positives are synthetic and must not be promoted to AM data. | false |
| P166-DESIGN-002 | model_budget | tiny_mlp_pinn_2_hidden_layers_width_32 | max 3 seeds, max 1500 optimizer steps, max 512 collocation points per step | Smoke should test mechanism plausibility, not scale. | false |
| P166-DESIGN-003 | sampler_variants | compare_uniform_grid_control_vs_failure_informed_hot_gradient | equal collocation point budget and identical data sensors | Isolates the Phase 165 sampler contribution. | false |
| P166-DESIGN-004 | bayesian_inverse_variant | use_calibrated_grid_posterior_as_parameter_prior_diagnostic | posterior informs reporting/calibration only; no full Bayesian neural net | Matches the Phase 164 calibrated hidden-parameter result without expensive BNN training. | false |
| P166-DESIGN-005 | metrics | validation_only_selection_rmse_residual_hot_gradient_parameter_error | test metrics reported once; no hyperparameter tuning on test | Preserves the gate discipline used in earlier phases. | false |
| P166-DESIGN-006 | promotion_rule | promote_only_if_failure_sampler_beats_uniform_and_data_only_controls | >=0.03 validation relative RMSE gain, no test reversal >1.05, parameter error non-worse | Prevents sampler-only coverage from being misread as model performance. | false |

## Controls
| control_id | control_name | role | required_metric | promotion_requirement |
| --- | --- | --- | --- | --- |
| P166-CTRL-001 | train_mean_or_sensor_interpolation | non-neural target baseline | temperature RMSE and hot/gradient q90 RMSE | tiny PINN must beat this on validation and avoid test reversal |
| P166-CTRL-002 | data_only_tiny_mlp_no_residual | tests whether physics residual adds value | temperature RMSE, residual RMSE, parameter error | PINN residual variants must beat or match data-only MLP |
| P166-CTRL-003 | uniform_grid_pinn | equal-budget sampler control | same optimizer steps and collocation budget | failure-informed sampler must beat uniform on validation |
| P166-CTRL-004 | wrong_parameter_prior_control | tests hidden-parameter interpretability | diffusivity/source-width error and prediction RMSE | calibrated prior route must not be worse than wrong-prior control |
| P166-CTRL-005 | seed_stability_control | prevents single-seed promotion | three seeds when the smoke is cheap enough | mean gain positive and worst seed not worse than uniform by >5% |

## Compute
| resource_id | resource | limit | allowed_now | escalation_rule |
| --- | --- | --- | --- | --- |
| P166-COMPUTE-001 | local_cpu_or_existing_local_gpu | preferred for Phase 167 smoke if torch is available | true | use A800 only if local torch/runtime blocks the tiny synthetic smoke |
| P166-COMPUTE-002 | A800_40GB | allowed only for reproduction or if local environment cannot import torch | false | still no AM-Bench training and no large sweeps |
| P166-COMPUTE-003 | A100_SXM4_80GB | not justified | false | request only after a seed-positive branch hits measured 40GB memory/runtime blockage |

## Route References
| reference_id | title | year | doi | source_url | verification_source | route_implication | phase166_decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P166-REF-001 | B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data | 2020 | 10.1016/j.jcp.2020.109913 | https://www.osti.gov/pages/biblio/2282008 | OSTI record and DOI metadata | Bayesian treatment is appropriate for noisy inverse PDE settings, but full BNN/HMC is heavier than a smoke gate. | Use calibrated grid-posterior diagnostics first; do not claim Bayesian PINN training. |
| P166-REF-002 | Efficient Bayesian Physics Informed Neural Networks for inverse problems via Ensemble Kalman Inversion | 2024 | 10.1016/j.jcp.2024.113006 | https://dl.acm.org/doi/10.1016/j.jcp.2024.113006 | Publisher/Crossref-indexed DOI metadata | Efficient Bayesian inverse-PINN variants exist and can become a later lightweight UQ branch. | Defer EKI-style Bayesian neural inference until a tiny deterministic PINN smoke passes. |
| P166-REF-003 | A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks | 2023 | 10.1016/j.cma.2022.115671 | https://github.com/lu-group/pinn-sampling | arXiv record and official code repository citation | RAD/RAR-D-style sampling is a valid comparator for residual-point efficiency. | Keep uniform/jittered controls and failure-informed sampler as equal-budget variants. |
| P166-REF-004 | Failure-Informed Adaptive Sampling for PINNs | 2023 | 10.1137/22M1527763 | https://epubs.siam.org/doi/abs/10.1137/22M1527763 | SIAM DOI landing page | Failure probability/error-indicator sampling supports Phase 165's sampler choice. | Require model-error improvement, not sampler coverage alone, before promotion. |
| P166-REF-005 | Machine learning for metal additive manufacturing: predicting temperature and melt pool fluid dynamics using physics-informed neural networks | 2021 | 10.1007/s00466-020-01952-9 | https://link.springer.com/article/10.1007/s00466-020-01952-9 | University publication record with Springer DOI | AM thermal PINNs are plausible, but AM-Bench claims require separate guarded data evidence. | Run synthetic inverse-heat smoke before returning to AM-Bench or NIST AMMT. |
| P166-REF-006 | Single-track thermal analysis of laser powder bed fusion process: Parametric solution through physics-informed neural networks | 2023 | 10.1016/j.cma.2023.116019 | https://www.research-collection.ethz.ch/handle/20.500.11850/607001 | ETH repository and DOI metadata | Parametric heat PINNs align with the user's hidden-physics/parameter-inference goal. | Track diffusivity/source-width parameter error as a smoke metric. |
| P166-REF-007 | Physics-informed graph neural Galerkin networks: A unified framework for solving PDE-governed forward and inverse problems | 2022 | 10.1016/j.cma.2021.114502 | https://par.nsf.gov/biblio/10338460-physics-informed-graph-neural-galerkin-networks-unified-framework-solving-pde-governed-forward-inverse-problems | NSF public-access manuscript and DOI metadata | GCN/graph discretizations are relevant for non-grid PDE domains. | Defer graph-PINN training because current project path-graph routes were already closed by earlier guards. |
| P166-REF-008 | Physics-informed machine learning-based real-time long-horizon temperature fields prediction in metallic additive manufacturing | 2025 | 10.1038/s44172-025-00501-7 | https://www.nature.com/articles/s44172-025-00501-7 | Nature Communications Engineering article page | Hybrid recurrent/CNN physics-informed AM models support future dense-field routes. | Do not open neural-operator/CNN dense training until leakage-safe dense targets are reopened. |

## Risks
| risk_id | risk | guard | closure_action |
| --- | --- | --- | --- |
| P166-RISK-001 | synthetic overfitting | test_shifted scenario and wrong-prior control | close as synthetic diagnostic if shifted test reverses |
| P166-RISK-002 | sampler advantage without model advantage | uniform-grid PINN and data-only MLP controls | do not promote if coverage gain does not reduce validation error |
| P166-RISK-003 | Bayesian language overclaim | calibrated grid posterior only; no full BNN claim | write as posterior diagnostic, not Bayesian PINN success |
| P166-RISK-004 | compute creep | max steps/width/seeds and no AM-Bench raw data | stop and redesign instead of scaling |
