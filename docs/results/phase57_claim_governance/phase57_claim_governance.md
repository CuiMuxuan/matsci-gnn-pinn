# AM-Bench Phase 57 Claim Governance

## Frozen floor

The frozen paper-facing floor is `broad_process_v1` on the `spot_size` holdout with seeds 7/1/2. Future branches must beat or preserve this floor before seed expansion.

| Dataset | Route | Seed status | RMSE | Hot q90 RMSE | Gradient q90 RMSE |
| --- | --- | --- | --- | --- | --- |
| broad12 | film/global_standard | seed_robust_transfer_positive | 136.384782 | 162.125337 | 165.282182 |
| broad21 | film/global_standard | seed_robust_transfer_positive | 146.002303 | 164.313888 | 174.735839 |

## Future branch contract

- Compare every candidate against mean, kNN, ExtraTrees, no-process Macro PINN, `process_axis_v1`, and `broad_process_v1` when artifacts are comparable.
- Select routes, hyperparameters, and feature families from train/validation evidence only.
- Do not use test metrics to decide whether to seed-expand a branch.
- A focused candidate may seed-expand only if it is non-worse than the frozen `broad_process_v1` floor on broad12 and broad21 for RMSE, hot q90 RMSE, and gradient q90 RMSE.
- A manuscript model claim requires at least seeds 7/1/2 and strong-baseline comparison, not a single focused run.

## Current process-axis ledger

| Dataset | Split | Status | Process claim | Route | Paper placement |
| --- | --- | --- | --- | --- | --- |
| broad12 | laser_power | route_guard_only | no | concat/global_standard | route_guard_boundary_table |
| broad12 | line | route_guard_no_process_positive | no | none/none | route_guard_boundary_table |
| broad12 | process | route_guard_only | no | none/none | route_guard_boundary_table |
| broad12 | scan_speed | route_guard_only | no | none/none | route_guard_boundary_table |
| broad12 | spot_size | paper_positive_seed_robust | yes | film/global_standard | main_table_and_main_figure |
| broad21 | laser_power | route_guard_only | no | concat/global_standard | route_guard_boundary_table |
| broad21 | line | route_guard_no_process_positive | no | none/none | route_guard_boundary_table |
| broad21 | process | route_guard_only | no | none/none | route_guard_boundary_table |
| broad21 | scan_speed | route_guard_only | no | none/none | route_guard_boundary_table |
| broad21 | spot_size | paper_positive_seed_robust | yes | film/global_standard | main_table_and_main_figure |

## Diagnostic branch counts

| Status | Count |
| --- | --- |
| blocked_by_data | 1 |
| diagnostic_negative | 11 |

## Claim wording boundary

The main process-conditioning claim is limited to the seed-robust `spot_size` route. `line` supports the no-process fallback route guard, not a process-conditioning improvement. `laser_power`, `scan_speed`, and full `process` remain route-guard-only unless a future candidate passes the Phase 57 gate.
