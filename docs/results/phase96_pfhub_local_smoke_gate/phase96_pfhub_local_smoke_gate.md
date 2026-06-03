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
| low_order_interpolation | 0.134094 | 0.130146 | 1.065073 | 0.174943 | 0.077841 |  |
| vanilla_deterministic_surrogate | 0.107311 | 0.101878 | 0.955219 | 0.114738 | 0.058955 |  |
| fixed_green_function_features | 0.000184 | 0.000171 | 0.002728 | 0.000085 | 0.000162 |  |
| random_collocation_same_budget | 0.000276 | 0.000553 | 0.007256 | 0.000048 | 0.000062 | 0.955940 |
| bayesian_adaptive_collocation | 0.000716 | 0.006432 | 0.033832 | 0.000012 | 0.001615 | 0.948859 |

## Next Action

enter Phase 97 AM-Bench/external transfer design gate; do not start A100 training
