# Phase 157 UCI Superconductivity Low-Capacity Mechanism Gate

## Gate
- Status: `phase157_uci_superconductivity_low_capacity_mechanism_closed_no_guarded_gain`
- Selected profile: `mechanism_full_low_capacity`
- Selected model: `huber;alpha=0.1`
- Focused validation allowed: `false`
- Model training allowed: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
This no-training gate asks whether an interpretable low-capacity mechanism can beat the Phase 156 ExtraTrees guard. A closure means the UCI source remains a strong baseline-positive diagnostic, not a model claim.

## Top Candidates
| profile | model_label | feature_count | val_rmse | test_rmse | relative_val_gain_vs_phase156 | test_reversal_ratio_vs_phase156 | selected_low_capacity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mechanism_full_low_capacity | huber;alpha=0.1 | 36 | 18.4273626141 | 23.5135708266 | -0.485311400491 | 1.59634401747 | true |
| mechanism_full_low_capacity | huber;alpha=1e-05 | 36 | 18.4274037528 | 23.5225366786 | -0.485314716419 | 1.59695271209 | false |
| mechanism_full_low_capacity | huber;alpha=0.001 | 36 | 18.4277542814 | 23.5205174987 | -0.485342970281 | 1.59681562931 | false |
| mechanism_full_low_capacity | lasso;alpha=0.1 | 36 | 18.7394023271 | 23.3548087605 | -0.510462918526 | 1.58556560885 | false |
| mechanism_full_low_capacity | elastic_net;alpha=0.1;l1_ratio=0.8 | 36 | 18.7581618372 | 23.4151536679 | -0.511975002206 | 1.58966244436 | false |
| mechanism_full_low_capacity | elastic_net;alpha=0.1;l1_ratio=0.5 | 36 | 18.8192997337 | 23.5691381144 | -0.516902935551 | 1.60011649882 | false |
| mechanism_full_low_capacity | elastic_net;alpha=0.01;l1_ratio=0.2 | 36 | 18.8348245107 | 23.5125017671 | -0.518154288154 | 1.59627143867 | false |
| mechanism_full_low_capacity | ridge;alpha=100 | 36 | 18.8348962587 | 23.5137051068 | -0.518160071302 | 1.59635313379 | false |

## Audits
| audit | status | severity | value | threshold | reason |
| --- | --- | --- | --- | --- | --- |
| phase156_gate_consistency | pass | info | phase156_uci_superconductivity_focused_review_ready_low_capacity_mechanism_gate | phase156_uci_superconductivity_focused_review_ready_low_capacity_mechanism_gate | Phase 157 requires Phase 156 to allow a no-training mechanism gate |
| phase156_validation_guard_gain | block | blocking | -0.485311400491 | 0.01 | selected low-capacity mechanism must beat the Phase 156 ExtraTrees guard on validation |
| phase156_test_reversal_guard | block | blocking | 1.59634401747 | 1.05 | selected low-capacity mechanism must not substantially reverse on test |
| low_capacity_feature_budget | pass | info | 36 | 36 | mechanism candidate must stay low-capacity and interpretable |
