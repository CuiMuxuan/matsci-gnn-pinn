# Phase 56 Manuscript-Facing Table/Figure Package

## Status

Package complete.

This package converts Phase 55/54 reports into manuscript-facing artifacts. It does not introduce new training results. The main paper claim is a seed-robust broad-data process route guard with a stable `spot_size -> FiLM/global-standard` branch.

## Main Table: Seed-Robust Spot-Size Positive

| Dataset | Split | Route | Metric | broad_process_v1 mean | std | best strong baseline | baseline method | no-process mean | delta vs strong | delta vs no-process |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broad12 | spot_size | film/global_standard | Test RMSE | 136.384782 | 0.467526 | 151.850578 | mean | 238.093690 | -15.465796 | -101.708908 |
| broad12 | spot_size | film/global_standard | Hot q90 RMSE | 162.125337 | 4.909788 | 252.554440 | mean | 424.409003 | -90.429103 | -262.283666 |
| broad12 | spot_size | film/global_standard | Gradient q90 RMSE | 165.282182 | 5.270236 | 233.119660 | mean | 382.799174 | -67.837478 | -217.516992 |
| broad21 | spot_size | film/global_standard | Test RMSE | 146.002303 | 1.118699 | 149.185412 | mean | 217.922642 | -3.183109 | -71.920340 |
| broad21 | spot_size | film/global_standard | Hot q90 RMSE | 164.313888 | 3.548500 | 251.976794 | mean | 401.488520 | -87.662906 | -237.174632 |
| broad21 | spot_size | film/global_standard | Gradient q90 RMSE | 174.735839 | 2.301005 | 231.072566 | mean | 360.868300 | -56.336727 | -186.132461 |

## Route-Guard Boundary Table

| Dataset | Split | Class | Route | Paper use | Metrics summary |
| --- | --- | --- | --- | --- | --- |
| broad12 | laser_power | route_guard_positive | concat/global_standard | route-guard-only boundary evidence | Test RMSE: 140.753534 vs mean 132.965887 (d=7.787647); Hot q90 RMSE: 254.473291 vs mean 242.427068 (d=12.046223); Gradient q90 RMSE: 215.411533 vs mean 208.105836 (d=7.305697) |
| broad12 | line | paper_claim_positive | none/none | route guard / no-process fallback evidence | Test RMSE: 126.308616 vs mean 134.042138 (d=-7.733522); Hot q90 RMSE: 217.257126 vs knn_coords 228.525979 (d=-11.268853); Gradient q90 RMSE: 195.314294 vs knn_coords 210.675696 (d=-15.361402) |
| broad12 | process | route_guard_positive | none/none | route-guard-only boundary evidence | Test RMSE: 181.091525 vs mean 147.381589 (d=33.709936); Hot q90 RMSE: 325.205379 vs mean 251.032500 (d=74.172879); Gradient q90 RMSE: 266.149257 vs mean 213.464819 (d=52.684438) |
| broad12 | scan_speed | route_guard_positive | none/none | route-guard-only boundary evidence | Test RMSE: 186.173938 vs mean 145.115776 (d=41.058162); Hot q90 RMSE: 345.736994 vs mean 250.659348 (d=95.077645); Gradient q90 RMSE: 266.380605 vs mean 209.791354 (d=56.589251) |
| broad21 | laser_power | route_guard_positive | concat/global_standard | route-guard-only boundary evidence | Test RMSE: 178.040331 vs mean 131.741364 (d=46.298967); Hot q90 RMSE: 296.909567 vs mean 237.730958 (d=59.178609); Gradient q90 RMSE: 254.954359 vs mean 205.133029 (d=49.821330) |
| broad21 | line | paper_claim_positive | none/none | route guard / no-process fallback evidence | Test RMSE: 126.194921 vs mean 131.161929 (d=-4.967008); Hot q90 RMSE: 234.351122 vs mean 243.188033 (d=-8.836911); Gradient q90 RMSE: 205.642173 vs mean 214.217962 (d=-8.575789) |
| broad21 | process | route_guard_positive | none/none | route-guard-only boundary evidence | Test RMSE: 166.231596 vs mean 145.350346 (d=20.881250); Hot q90 RMSE: 308.389105 vs mean 248.754243 (d=59.634862); Gradient q90 RMSE: 251.049837 vs mean 216.442403 (d=34.607434) |
| broad21 | scan_speed | route_guard_positive | none/none | route-guard-only boundary evidence | Test RMSE: 227.128663 vs mean 144.014351 (d=83.114312); Hot q90 RMSE: 392.018079 vs mean 251.407139 (d=140.610940); Gradient q90 RMSE: 304.940054 vs mean 212.910489 (d=92.029565) |

## Negative Diagnostic Appendix

| Phase | Branch | Target | Result | Paper use | Evidence |
| --- | --- | --- | --- | --- | --- |
| 33 | Fixed Fourier spacetime representation | broad12 broad_process_v1, all grouped splits | Negative | Appendix diagnostic: fixed coordinate basis degraded all broad12 splits. | docs/results/ambench_multiline_process_fourier_spacetime_v1.md |
| 34 | Sparse closure and learned residual correction | broad12 spot_size | Negative | Appendix diagnostic: closure/residual heads over-traded hot/gradient metrics. | docs/results/ambench_multiline_process_residual_correction_v1.md |
| 35 | Train-split region-weighted data loss | broad12 spot_size | Negative after seed check | Appendix diagnostic: single-seed region gains did not survive paired seeds. | docs/results/ambench_multiline_process_region_weighted_loss_v1.md |
| 36 | Structured process-neighborhood RBF features | broad12/broad21 laser_power/spot_size diagnostics | Unstable | Appendix diagnostic: split-local process-neighborhood signal did not transfer. | docs/results/ambench_multiline_process_process_graph_rbf_v1.md |
| 37 | Strong-baseline residualized Macro PINN | broad12 spot_size and laser_power | Negative | Appendix diagnostic: ExtraTrees residualization left no useful neural residual. | docs/results/ambench_multiline_process_target_residual_v1.md |
| 38 | Residual Macro PINN backbone | broad12 spot_size and laser_power | Negative | Appendix diagnostic: backbone changes improved one metric only by hurting others. | docs/results/ambench_multiline_process_residual_backbone_v1.md |
| 39-40 | Process-conditioned output affine calibration | laser_power broad12/broad21 | Non-transferable | Appendix diagnostic: broad12 local positive failed broad21 transfer. | docs/results/ambench_multiline_process_output_affine_v1.md |
| 41-43 | Derived process features and process encoder | laser_power broad12/broad21 | Non-transferable | Appendix diagnostic: representation signals were broad21-positive but broad12-negative. | docs/results/ambench_multiline_process_process_encoder_v1.md |
| 44 | Process-condition group-balanced objective | laser_power broad12/broad21 | Negative | Appendix diagnostic: region gains on broad21 sacrificed global RMSE and broad12 failed. | docs/results/ambench_multiline_process_group_balance_v1.md |
| 45 | Baseline-guarded expert stack gate | laser_power broad12/broad21 | Negative | Appendix diagnostic: existing experts did not contain a validation-selectable transferable stack. | docs/results/ambench_multiline_process_baseline_guarded_expert_plan_v1.md |
| 46-51 | Bayesian inverse closure and moving-source inversion | synthetic and AM-Bench Line_0_1 sparse/dense gates | Synthetic-positive, AM-Bench-negative | Limitation/negative control: interpretable source inversion needs better AM-Bench path registration. | docs/results/ambench_dense_source_parameter_transfer_v1.md |
| 52-53 | Registered source-path features and data pivot | Line_0_1 and pad thermography diagnostics | Data-incompatible | Limitation: current bundle lacks single-track scan-path/camera registration metadata. | docs/results/ambench_source_path_data_pivot_gate_v1.md |

## Figure

- Editable SVG: `docs/results/phase58_clean_repro/phase56_manuscript_package/phase56_spot_size_seed_validation_figure.svg`
- PNG preview: `not generated; cairosvg unavailable`

Suggested caption:

> Seed-robust `spot_size` validation of the broad-data process route guard. Bars show three-seed mean RMSE on broad12 and broad21 for global test points, hot-zone q90 points, and gradient q90 points. The `broad_process_v1` FiLM/global-standard route is compared with the no-process Macro PINN and the best strong classical baseline selected independently for each metric. Lower is better.

## Manuscript Placement

- Main results table: use the main table above.
- Method/model contribution figure: use the figure with a short route-guard schematic in the text, or pair it with the Phase 54 claim-boundary table.
- Supplementary appendix: include the route-guard boundary and negative diagnostic tables.

## Source Trace

- `outputs/reports/phase55_spot_size_route_seed_check_summary.json`
- `outputs/reports/phase54_process_route_claim_boundary_summary.json`
- `docs/results/ambench_multiline_process_spot_size_seed_validation_v1.md`
- `docs/results/ambench_multiline_process_route_claim_boundary_v1.md`
