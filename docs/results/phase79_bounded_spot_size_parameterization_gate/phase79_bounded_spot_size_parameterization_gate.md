# Phase 79 Bounded Spot-Size Parameterization Gate

## Candidate

Candidate: `bounded_spot_size_parameterization_v1`.

Add a constrained, low-capacity spot-size response inside the existing `broad_process_v1` FiLM/global-standard route, with identity initialization and bounded modulation so the frozen `spot_size` floor cannot be silently replaced.

## Gate Decision

Status: `local_surrogate_required_before_a100`.
A100 seed-7 allowed: `false`.
Local surrogate allowed: `true`.
A100-SXM4-80GB request now: `false`.

fixed-sampling margins are positive, but density debt and Phase 69 still block direct A100 seed-7 validation

## Margin Rows

| Dataset | Metric | Fixed margin | Density debt | Debt / margin | Status |
| --- | --- | --- | --- | --- | --- |
| broad12 | rmse | 15.465796 | 0.000000 | 0.000000 | margin_preserved |
| broad12 | hot_q90_rmse | 90.429103 | 0.000000 | 0.000000 | margin_preserved |
| broad12 | gradient_q90_rmse | 67.837478 | 0.000000 | 0.000000 | margin_preserved |
| broad21 | rmse | 3.183109 | 13.533809 | 4.251757 | density_debt_exceeds_floor_margin |
| broad21 | hot_q90_rmse | 87.662906 | 17.499198 | 0.199619 | density_debt_within_floor_margin |
| broad21 | gradient_q90_rmse | 56.336727 | 18.739041 | 0.332626 | density_debt_within_floor_margin |

## Next Action

build a local surrogate gate for bounded_spot_size_parameterization_v1; do not run broad12/broad21 A100 training yet
