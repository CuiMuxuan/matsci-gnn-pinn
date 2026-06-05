# Phase 122 Matbench Steels Low-Capacity Mechanism Gate

- Status: `phase122_matbench_steels_low_capacity_mechanism_closed_no_guarded_gain`
- Selected low-capacity profile: `mechanism_interaction_full`
- Selected model: `lasso;alpha=1`
- Selected validation RMSE: `360.697`
- Phase 121 guard validation RMSE: `252.86`
- Focused validation allowed: `False`
- Model training allowed: `False`
- A100 training allowed now: `False`

## Selected Candidate

| Profile | Model | Features | Val RMSE | Test RMSE | Val gain vs Phase121 | Test ratio vs Phase121 |
| --- | --- | --- | --- | --- | --- | --- |
| mechanism_interaction_full | lasso;alpha=1 | 34 | 360.697 | 306.975 | -0.426471 | 1.36001 |

## Blocking Audits

| Audit | Value | Threshold | Reason |
| --- | --- | --- | --- |
| phase121_validation_guard_gain | -0.426471 | 0.01 | validation-only selected low-capacity mechanism must beat the Phase 121 ExtraTrees guard |
| phase121_test_reversal_guard | 1.36001 | 1.05 | validation gain must not reverse badly on the held-out test split |

## Top Validation Candidates

| Profile | Model | Val RMSE | Test RMSE | Features |
| --- | --- | --- | --- | --- |
| mechanism_interaction_full | lasso;alpha=1 | 360.697 | 306.975 | 34 |
| mechanism_interaction_full | ridge;alpha=1e-06 | 380.77 | 332.929 | 34 |
| mechanism_interaction_full | ridge;alpha=0.0001 | 380.859 | 332.752 | 34 |
| mechanism_interaction_full | lasso;alpha=1e-05 | 383.19 | 328.131 | 34 |
| mechanism_interaction_full | lasso;alpha=0.0001 | 383.191 | 328.128 | 34 |
| mechanism_interaction_full | lasso;alpha=0.001 | 383.199 | 328.086 | 34 |
| mechanism_interaction_full | lasso;alpha=0.01 | 383.449 | 327.274 | 34 |
| mechanism_interaction_full | elastic_net;alpha=0.0001;l1_ratio=0.8 | 385.297 | 326.517 | 34 |

## Selected Coefficients

| Rank | Feature | Std coef | Raw coef |
| --- | --- | --- | --- |
| 1 | phase122_Ni_x_Ti | 218.532 | 228986 |
| 2 | frac_C | 154.068 | 28983.8 |
| 3 | phase122_Co_x_Ni | 98.6043 | 19200 |
| 4 | frac_Al | 98.2508 | 19209.7 |
| 5 | phase122_C_x_Co | 93.8623 | 162759 |
| 6 | frac_Ti | -91.0882 | -15677.2 |
| 7 | frac_Ni | 88.9804 | 1574.06 |
| 8 | phase122_C_x_Cr | -87.2973 | -152751 |
| 9 | phase122_Mo_x_Ni | -55.8784 | -63176.9 |
| 10 | phase122_Cr_x_Mo | 51.2812 | 34316.4 |
| 11 | phase122_C_x_Ni | -45.1585 | -148205 |
| 12 | phase122_C_x_V | 44.4144 | 936744 |
