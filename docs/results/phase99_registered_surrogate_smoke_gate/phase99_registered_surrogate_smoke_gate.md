# Phase 99 Local Registered-Surrogate Baseline-First Smoke Gate

## Gate Decision

Status: `local_surrogate_positive_with_focused_baseline_boundary`.
Local mechanism package allowed: `true`.
AM-Bench transfer unlocked: `false`.
A100 training allowed now: `false`.
A100-SXM4-80GB request now: `false`.

Phase 99 is a local registered-surrogate smoke gate. It does not provide AM-Bench transfer evidence.

## Metrics

| Method | Budget | Test RMSE | Residual | Hot q90 | Gradient q90 |
| --- | --- | --- | --- | --- | --- |
| low_order_interpolation_full | full_grid | 0.130146 | 1.065073 | 0.174943 | 0.077841 |
| vanilla_deterministic_surrogate_full | full_grid | 0.101878 | 0.955219 | 0.114738 | 0.058955 |
| fixed_green_function_features_full | full_grid | 0.000171 | 0.002728 | 0.000085 | 0.000162 |
| random_collocation_same_budget | same_budget | 0.000258 | 0.004172 | 0.000085 | 0.000154 |
| source_quota_green_same_budget | same_budget | 0.000641 | 0.015039 | 0.000054 | 0.000419 |

## Comparison Audit

| Baseline | Metric | Delta | Pass | Scope |
| --- | --- | --- | --- | --- |
| low_order_interpolation_full | global_rmse | 0.129975 | true | full-grid baselines |
| low_order_interpolation_full | pde_residual_rmse | 1.062344 | true | full-grid baselines |
| low_order_interpolation_full | hot_q90_rmse | 0.174858 | true | full-grid baselines |
| low_order_interpolation_full | gradient_q90_rmse | 0.077678 | true | full-grid baselines |
| vanilla_deterministic_surrogate_full | global_rmse | 0.101707 | true | full-grid baselines |
| vanilla_deterministic_surrogate_full | pde_residual_rmse | 0.952491 | true | full-grid baselines |
| vanilla_deterministic_surrogate_full | hot_q90_rmse | 0.114653 | true | full-grid baselines |
| vanilla_deterministic_surrogate_full | gradient_q90_rmse | 0.058792 | true | full-grid baselines |
| random_collocation_same_budget | global_rmse | 0.000088 | true | same-budget boundary audit |
| random_collocation_same_budget | pde_residual_rmse | 0.001444 | true | same-budget boundary audit |
| random_collocation_same_budget | hot_q90_rmse | 0.000000 | true | same-budget boundary audit |
| random_collocation_same_budget | gradient_q90_rmse | -0.000008 | false | same-budget boundary audit |

## Next Action

package as local mechanism evidence and keep AM-Bench transfer blocked until registered data exists
