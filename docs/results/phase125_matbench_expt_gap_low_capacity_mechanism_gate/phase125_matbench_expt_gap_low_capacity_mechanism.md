# Phase 125 Matbench Experimental Gap Low-Capacity Mechanism Gate

- Status: `phase125_matbench_expt_gap_low_capacity_mechanism_closed_no_guarded_gain`
- Selected low-capacity profile: `mechanism_full_low_capacity`
- Selected model: `ridge;alpha=10`
- Selected validation RMSE: `1.07469`
- Phase 124 guard validation RMSE: `0.716673`
- Focused validation allowed: `False`
- Model training allowed: `False`
- A100 training allowed now: `False`

## Selected Candidate

| Profile | Model | Features | Val RMSE | Test RMSE | Val gain vs Phase124 | Test ratio vs Phase124 |
| --- | --- | --- | --- | --- | --- | --- |
| mechanism_full_low_capacity | ridge;alpha=10 | 35 | 1.07469 | 1.36631 | -0.499547 | 1.29742 |

## Blocking Audits

| Audit | Value | Threshold | Reason |
| --- | --- | --- | --- |
| phase124_validation_guard_gain | -0.499547 | 0.01 | validation-only selected low-capacity mechanism must beat the Phase 124 ExtraTrees guard |
| phase124_test_reversal_guard | 1.29742 | 1.05 | validation gain must not reverse badly on the held-out test split |

## Top Validation Candidates

| Profile | Model | Val RMSE | Test RMSE | Features |
| --- | --- | --- | --- | --- |
| mechanism_full_low_capacity | ridge;alpha=10 | 1.07469 | 1.36631 | 35 |
| mechanism_full_low_capacity | elastic_net;alpha=0.01;l1_ratio=0.2 | 1.07573 | 1.36809 | 35 |
| mechanism_full_low_capacity | lasso;alpha=0.001 | 1.07667 | 1.36663 | 35 |
| mechanism_full_low_capacity | elastic_net;alpha=0.001;l1_ratio=0.8 | 1.0769 | 1.36696 | 35 |
| mechanism_full_low_capacity | elastic_net;alpha=0.001;l1_ratio=0.5 | 1.07723 | 1.36744 | 35 |
| mechanism_full_low_capacity | ridge;alpha=100 | 1.07752 | 1.37151 | 35 |
| mechanism_full_low_capacity | elastic_net;alpha=0.001;l1_ratio=0.2 | 1.07755 | 1.3679 | 35 |
| mechanism_full_low_capacity | elastic_net;alpha=0.01;l1_ratio=0.5 | 1.07858 | 1.37173 | 35 |

## Selected Coefficients

| Rank | Feature | Std coef | Raw coef |
| --- | --- | --- | --- |
| 1 | phase125_nonmetal_fraction | 0.895931 | 2.9879 |
| 2 | phase125_chalcogen_metal_interaction | -0.578109 | -5.32259 |
| 3 | chalcogen_fraction | 0.518573 | 1.81482 |
| 4 | metalloid_fraction | 0.445585 | 2.10368 |
| 5 | anion_fraction | 0.332513 | 1.08269 |
| 6 | phase125_en_variance | 0.289812 | 0.768512 |
| 7 | mean_electronegativity | -0.257398 | -0.576178 |
| 8 | phase125_spin_orbit_proxy | -0.247472 | -1.66793 |
| 9 | entropy_fraction | -0.242674 | -1.12368 |
| 10 | pnictogen_fraction | -0.229793 | -1.40893 |
| 11 | mean_atomic_number | 0.210268 | 0.0138061 |
| 12 | phase125_post_transition_chalcogen | 0.206885 | 3.31277 |
