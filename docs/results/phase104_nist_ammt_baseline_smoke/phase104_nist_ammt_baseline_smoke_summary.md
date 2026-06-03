# Phase 104 NIST AMMT Baseline-First Tiny Smoke

- Status: `phase104_baseline_smoke_boundary_tiny_sample_mechanisms_locked`
- Baseline smoke completed: `True`
- Row count: `24`
- Sample size sufficient for Phase 105: `False`
- Phase 105 model mechanisms allowed: `false`
- A100 training allowed now: `false`

| Method | Split | N | RMSE | Hot q90 RMSE | Gradient q90 RMSE |
|---|---|---:|---:|---:|---:|
| mean | test | 6 | 3.460781 | 1.426275 | 3.460781 |
| mean | train | 14 | 1.213729 | 1.304880 | 1.213729 |
| mean | val | 4 | 3.390400 | 2.973377 | 3.390400 |
| knn | test | 6 | 2.713934 | 1.904951 | 2.713934 |
| knn | train | 14 | 0.869788 | 0.976733 | 0.869788 |
| knn | val | 4 | 2.764919 | 3.318393 | 2.764919 |
| extra_trees | test | 6 | 2.056080 | 1.801907 | 2.056080 |
| extra_trees | train | 14 | 0.000000 | 0.000000 | 0.000000 |
| extra_trees | val | 4 | 2.563618 | 3.065739 | 2.563618 |

Next action: expand the registered numeric table before Phase 105 mechanism testing
