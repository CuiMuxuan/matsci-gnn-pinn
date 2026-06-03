# Phase 104 NIST AMMT Baseline-First Tiny Smoke

- Status: `phase104_baseline_smoke_complete_mechanisms_review_required`
- Baseline smoke completed: `True`
- Row count: `128`
- Sample size sufficient for Phase 105: `True`
- Phase 105 model mechanisms allowed: `false`
- A100 training allowed now: `false`

| Method | Split | N | RMSE | Hot q90 RMSE | Gradient q90 RMSE |
|---|---|---:|---:|---:|---:|
| mean | test | 26 | 0.711468 | 1.426059 | 0.711468 |
| mean | train | 76 | 3.322762 | 4.709487 | 3.322762 |
| mean | val | 26 | 0.723440 | 1.094724 | 0.723440 |
| knn | test | 26 | 4.855808 | 9.663194 | 4.855808 |
| knn | train | 76 | 2.315174 | 1.768765 | 2.315174 |
| knn | val | 26 | 3.825270 | 9.165106 | 3.825270 |
| extra_trees | test | 26 | 9.826492 | 15.121652 | 9.826492 |
| extra_trees | train | 76 | 0.000000 | 0.000000 | 0.000000 |
| extra_trees | val | 26 | 9.083277 | 15.415926 | 9.083277 |

Next action: review validation/test baseline gap before opening Phase 105
