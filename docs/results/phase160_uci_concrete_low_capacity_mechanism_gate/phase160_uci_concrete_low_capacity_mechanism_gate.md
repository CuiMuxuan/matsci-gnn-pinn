# Phase 160 UCI Concrete Low-Capacity Mechanism Gate

## Gate
- Status: `phase160_uci_concrete_low_capacity_mechanism_closed_no_guarded_gain`
- Selected profile: `mechanism_full_low_capacity`
- Selected model: `lasso;alpha=0.0001`
- Focused validation allowed: `false`
- Model training allowed: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
This no-training gate asks whether interpretable concrete mechanism features can beat the Phase 159 HistGradientBoosting guard. A closure means the concrete source remains a robust source-level diagnostic, not a second-paper mechanism or model claim.

## Top Candidates
| profile | model_label | feature_count | val_rmse | test_rmse | relative_val_gain_vs_phase159 | test_reversal_ratio_vs_phase159 | selected_low_capacity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mechanism_full_low_capacity | lasso;alpha=0.0001 | 36 | 6.41897953369 | 6.2968192414 | -0.133056497497 | 1.17338346759 | true |
| mechanism_full_low_capacity | ordinary_least_squares | 36 | 6.41978825588 | 6.29469810235 | -0.133199250396 | 1.17298820303 | false |
| mechanism_full_low_capacity | ridge;alpha=0.0001 | 36 | 6.41983757133 | 6.29467470235 | -0.13320795539 | 1.17298384255 | false |
| mechanism_full_low_capacity | ridge;alpha=0.01 | 36 | 6.42505307856 | 6.29528180098 | -0.134128579038 | 1.1730969726 | false |
| mechanism_full_low_capacity | lasso;alpha=0.001 | 36 | 6.43508234118 | 6.29268004765 | -0.135898910463 | 1.17261214777 | false |
| mechanism_full_low_capacity | lasso;alpha=0.01 | 36 | 6.45617698179 | 6.49851166457 | -0.13962246488 | 1.21096792822 | false |
| mechanism_full_low_capacity | elastic_net;alpha=0.01;l1_ratio=0.2 | 36 | 6.4588887081 | 6.57903875813 | -0.140101129613 | 1.22597378382 | false |
| mechanism_full_low_capacity | ridge;alpha=10 | 36 | 6.46094554301 | 6.63428601295 | -0.140464195135 | 1.23626885709 | false |

## Audits
| audit | status | severity | value | threshold | reason |
| --- | --- | --- | --- | --- | --- |
| phase159_gate_consistency | pass | info | phase159_uci_concrete_focused_review_ready_low_capacity_mechanism_gate | phase159_uci_concrete_focused_review_ready_low_capacity_mechanism_gate | Phase 160 requires Phase 159 to allow a no-training mechanism gate |
| phase159_validation_guard_gain | block | blocking | -0.133056497497 | 0.01 | selected low-capacity mechanism must beat the Phase 159 HGB guard on validation |
| phase159_test_reversal_guard | block | blocking | 1.17338346759 | 1.05 | selected low-capacity mechanism must not substantially reverse on test |
| low_capacity_feature_budget | pass | info | 36 | 36 | mechanism candidate must stay low-capacity and interpretable |
