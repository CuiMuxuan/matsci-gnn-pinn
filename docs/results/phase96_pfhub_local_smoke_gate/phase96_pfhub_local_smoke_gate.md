# Phase 96 PFHub-Style Local Smoke Gate

## Gate Decision

Status: `local_smoke_positive_transfer_design_only`.
Phase 97 transfer design allowed: `true`.
A100 training allowed now: `false`.
A100-SXM4-80GB request now: `false`.

Phase 96 is a local smoke gate only. It can open transfer-design work, but it is not AM-Bench evidence.

## Mechanism Decisions

| Mechanism | Name | Comparator | Validation pass | Test audit | Next action |
| --- | --- | --- | --- | --- | --- |
| P96-MECH-001 | fixed_green_function_features | vanilla_deterministic_surrogate | true | true | open Phase 97 transfer design gate |
| P96-MECH-002 | bayesian_adaptive_collocation | random_collocation_same_budget | false | false | keep as diagnostic until global/coverage guards pass |

## Metric Summary

| Method | Val RMSE | Test RMSE | Residual RMSE | Hot q90 | Gradient q90 | Coverage |
| --- | --- | --- | --- | --- | --- | --- |
| low_order_interpolation | 0.134093698 | 0.130146139 | 1.065072717 | 0.174942683 | 0.077840696 |  |
| vanilla_deterministic_surrogate | 0.107310631 | 0.101878075 | 0.955218992 | 0.114737572 | 0.058954667 |  |
| fixed_green_function_features | 0.000184038 | 0.000170693 | 0.002728446 | 0.000084619 | 0.000162240 |  |
| random_collocation_same_budget | 0.000275759 | 0.000553404 | 0.007255836 | 0.000048083 | 0.000062399 | 0.955940205 |
| bayesian_adaptive_collocation | 0.000715552 | 0.006432292 | 0.033832406 | 0.000011655 | 0.001615367 | 0.948859166 |

## Next Action

enter Phase 97 AM-Bench/external transfer design gate; do not start A100 training
