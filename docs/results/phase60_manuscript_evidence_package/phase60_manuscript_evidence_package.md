# AM-Bench Phase 60 Manuscript Evidence Package

## Purpose

Phase 60 consolidates the post-Phase-59 evidence boundary for manuscript drafting. It does not add training evidence and does not reopen a model branch from the density failure.

## Main Claim Floor

Current transfer gate: `seed_robust_transfer_positive`. The main paper-positive evidence remains the fixed-sampling Phase 55 `spot_size` result under `broad_process_v1` with seeds 7/1/2.

| Dataset | Metric | broad_process_v1 | Best strong baseline | Delta | Gate |
| --- | --- | --- | --- | --- | --- |
| broad12 | Test RMSE | 136.384781987 | 151.850577987 | -15.4657960003 | seed_robust_transfer_positive |
| broad12 | Hot q90 RMSE | 162.125337065 | 252.554439588 | -90.4291025223 | seed_robust_transfer_positive |
| broad12 | Gradient q90 RMSE | 165.282182272 | 233.119660009 | -67.8374777369 | seed_robust_transfer_positive |
| broad21 | Test RMSE | 146.002302637 | 149.185412106 | -3.18310946913 | seed_robust_transfer_positive |
| broad21 | Hot q90 RMSE | 164.31388799 | 251.976794482 | -87.662906492 | seed_robust_transfer_positive |
| broad21 | Gradient q90 RMSE | 174.735838739 | 231.072566056 | -56.3367273176 | seed_robust_transfer_positive |

## Route Guard Boundaries

Laser power, scan speed, full process, and no-process fallback axes remain boundary evidence unless a future branch passes the frozen-floor gate.

| Dataset | Split | Classification | Route | Use |
| --- | --- | --- | --- | --- |
| broad12 | laser_power | route_guard_positive | concat/global_standard | route-guard-only boundary evidence |
| broad12 | line | paper_claim_positive | none/none | route guard / no-process fallback evidence |
| broad12 | process | route_guard_positive | none/none | route-guard-only boundary evidence |
| broad12 | scan_speed | route_guard_positive | none/none | route-guard-only boundary evidence |
| broad21 | laser_power | route_guard_positive | concat/global_standard | route-guard-only boundary evidence |
| broad21 | line | paper_claim_positive | none/none | route guard / no-process fallback evidence |
| broad21 | process | route_guard_positive | none/none | route-guard-only boundary evidence |
| broad21 | scan_speed | route_guard_positive | none/none | route-guard-only boundary evidence |

## Stress and Residual Boundaries

Stronger-baseline stress supports the fixed-sampling floor, while alternate-density broad21 is a density-sensitive boundary. Phase 59 selected a mean fallback from validation, so the density failure is not a model-expansion signal.

| Scenario | Dataset | Metric | Status | Candidate | Comparator | Delta | Use |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stronger_baseline_stress | broad12 | Test RMSE | pass | 136.384782 | mean: 151.850578 | -15.465796 | supports fixed-sampling Phase 55 floor |
| stronger_baseline_stress | broad12 | Hot q90 RMSE | pass | 162.125337 | mean: 252.554440 | -90.429103 | supports fixed-sampling Phase 55 floor |
| stronger_baseline_stress | broad12 | Gradient q90 RMSE | pass | 165.282182 | mean: 233.119660 | -67.837478 | supports fixed-sampling Phase 55 floor |
| stronger_baseline_stress | broad21 | Test RMSE | pass | 146.002303 | mean: 149.185412 | -3.183109 | supports fixed-sampling Phase 55 floor |
| stronger_baseline_stress | broad21 | Hot q90 RMSE | pass | 164.313888 | mean: 251.976794 | -87.662906 | supports fixed-sampling Phase 55 floor |
| stronger_baseline_stress | broad21 | Gradient q90 RMSE | pass | 174.735839 | mean: 231.072566 | -56.336727 | supports fixed-sampling Phase 55 floor |
| alternate_density_stress | broad12 | Test RMSE | pass | 139.085217 | mean: 140.201362 | -1.116145 | density stress support |
| alternate_density_stress | broad12 | Hot q90 RMSE | pass | 235.696768 | mean: 253.374499 | -17.677731 | density stress support |
| alternate_density_stress | broad12 | Gradient q90 RMSE | pass | 221.604222 | mean: 234.028899 | -12.424677 | density stress support |
| alternate_density_stress | broad21 | Test RMSE | boundary | 153.259455 | mean: 139.725646 | 13.533809 | density-sensitive boundary |
| alternate_density_stress | broad21 | Hot q90 RMSE | boundary | 270.628922 | mean: 253.129723 | 17.499198 | density-sensitive boundary |
| alternate_density_stress | broad21 | Gradient q90 RMSE | boundary | 250.519935 | mean: 231.780894 | 18.739041 | density-sensitive boundary |
| auxiliary_process_panel | broad15 | Test RMSE | pass | 138.855456 | mean: 151.850578 | -12.995122 | auxiliary process-panel support |
| auxiliary_process_panel | broad15 | Hot q90 RMSE | pass | 158.622677 | mean: 252.554440 | -93.931762 | auxiliary process-panel support |
| auxiliary_process_panel | broad15 | Gradient q90 RMSE | pass | 165.869192 | mean: 233.732337 | -67.863145 | auxiliary process-panel support |
| residual_upper_bound_gate | broad21_density | Test RMSE | blocks_model_expansion | 153.259459 | mean: 139.725646 | 13.533813 | route-boundary evidence; do not claim density-invariant robustness |

## Next Branch Gates

| Branch | Status | Entry condition | Seed gate |
| --- | --- | --- | --- |
| Candidate A: physically constrained spot-size conditioning | paused | Requires a new validation-visible spot-size signal, not the Phase 58 density failure alone. | Seed-expand only if seed 7 is non-worse than broad_process_v1 on broad12, broad21 for rmse, hot_q90_rmse, gradient_q90_rmse. |
| Candidate B: validation-auditable route policy | blocked by Phase 59 density gate | Route choice must be selected from train/validation evidence and existing comparable artifacts. | Seed-expand only if seed 7 is non-worse than broad_process_v1 on broad12, broad21 for rmse, hot_q90_rmse, gradient_q90_rmse. |
| Candidate C: heat-kernel or Green's-function source features | blocked by registration data | Requires aligned single-track scan-path metadata or a defensible pad thermography target. | Seed-expand only if seed 7 is non-worse than broad_process_v1 on broad12, broad21 for rmse, hot_q90_rmse, gradient_q90_rmse. |
| External robustness / second dataset branch | deferred until manuscript package is draft-ready | Use after the current AM-Bench claim package is internally complete. | Define a new frozen floor before seed expansion on the added dataset. |

## Manuscript Placement

- Main table: `phase60_main_spot_size_seed_positive_table.csv`.
- Route-guard table: `phase60_route_guard_boundary_table.csv`.
- Stress/boundary table: `phase60_stress_boundary_table.csv`.
- Appendix negative/boundary diagnostics: `phase60_appendix_negative_diagnostic_table.csv`.

## Claim Guardrail

Phase 59 selected `blend:broad_process_v1->mean:alpha=1` with `uses_test_for_selection=False`. This blocks density-failure-driven model expansion until a new validation-visible signal appears.

Appendix rows after Phase 60: `14`.
