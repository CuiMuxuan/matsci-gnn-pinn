# Phase 91 Table, Figure, and Appendix Freeze

## Gate Decision

Status: `table_figure_appendix_frozen_venue_unresolved`.
Frozen: `true`.
Submission ready: `false`.
A100 training allowed now: `false`.

Phase 91 freezes manuscript-facing quantitative artifacts without adding results or venue-specific formatting.

## Table/Figure Manifest

| Item | Label | Type | Role | Status |
| --- | --- | --- | --- | --- |
| P91-TABLE-001 | Table 1 | main_table | main positive performance table | frozen |
| P91-TABLE-002 | Table 2 | route_guard_table | route-boundary table | frozen |
| P91-TABLE-003 | Table 3 | stress_boundary_table | stress support and density-boundary table | frozen |
| P91-TABLE-S001 | Table S1 | appendix_table | appendix negative diagnostic ledger | frozen |
| P91-TABLE-S002 | Table S2 | future_gate_table | future model branch gate summary | frozen |
| P91-FIG-001 | Figure 1 | figure | seed-stability visualization for fixed spot_size claim | frozen_existing_asset |

## Main Table Check

| Row | Dataset | Metric | Delta vs best strong | Status |
| --- | --- | --- | --- | --- |
| P91-MAIN-001 | broad12 | Test RMSE | -15.4657960003 | frozen_main_claim_row |
| P91-MAIN-002 | broad12 | Hot q90 RMSE | -90.4291025223 | frozen_main_claim_row |
| P91-MAIN-003 | broad12 | Gradient q90 RMSE | -67.8374777369 | frozen_main_claim_row |
| P91-MAIN-004 | broad21 | Test RMSE | -3.18310946913 | frozen_main_claim_row |
| P91-MAIN-005 | broad21 | Hot q90 RMSE | -87.662906492 | frozen_main_claim_row |
| P91-MAIN-006 | broad21 | Gradient q90 RMSE | -56.3367273176 | frozen_main_claim_row |

## Route Guard Check

| Row | Dataset | Split | Role | Guard |
| --- | --- | --- | --- | --- |
| P91-ROUTE-001 | broad12 | laser_power | route_guard_boundary | Do not write this row as a strong-baseline-positive main claim. |
| P91-ROUTE-002 | broad12 | line | route_guard_no_process_fallback | Do not write this row as process-conditioning improvement. |
| P91-ROUTE-003 | broad12 | process | route_guard_boundary | Do not write this row as a strong-baseline-positive main claim. |
| P91-ROUTE-004 | broad12 | scan_speed | route_guard_boundary | Do not write this row as a strong-baseline-positive main claim. |
| P91-ROUTE-005 | broad21 | laser_power | route_guard_boundary | Do not write this row as a strong-baseline-positive main claim. |
| P91-ROUTE-006 | broad21 | line | route_guard_no_process_fallback | Do not write this row as process-conditioning improvement. |
| P91-ROUTE-007 | broad21 | process | route_guard_boundary | Do not write this row as a strong-baseline-positive main claim. |
| P91-ROUTE-008 | broad21 | scan_speed | route_guard_boundary | Do not write this row as a strong-baseline-positive main claim. |

## Appendix Coverage

Frozen appendix rows: `18`.

## Next Action

enter Phase 92 internal benchmark review or provide target venue before final formatting
