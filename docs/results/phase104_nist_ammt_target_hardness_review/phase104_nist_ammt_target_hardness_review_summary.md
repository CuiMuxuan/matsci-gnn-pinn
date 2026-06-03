# Phase 104 NIST AMMT Target-Hardness Review

- Status: `phase104_target_hardness_review_ready_phase105_design`
- Baseline gate status: `phase104_baseline_smoke_complete_mechanisms_review_required`
- Selected target: `target_intensity_std`
- Phase 105 low-capacity design allowed: `True`
- A100 training allowed now: `false`

| Target | Status | Selected method | Val RMSE | Test RMSE | Mean val RMSE | Mean test RMSE | Val improvement vs mean |
|---|---|---|---:|---:|---:|---:|---:|
| target_intensity_mean | blocked_mean_baseline_best | mean | 0.723440 | 0.711468 | 0.723440 | 0.711468 | 0.000000 |
| target_intensity_std | candidate_target_ready_for_phase105_design | knn | 1.011970 | 1.292170 | 2.104431 | 2.215968 | 0.519124 |
| target_intensity_min | candidate_target_ready_for_phase105_design | extra_trees | 3.410944 | 3.549128 | 7.606510 | 7.530013 | 0.551576 |
| target_intensity_max | blocked_zero_variance_target | extra_trees | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| target_intensity_q90 | blocked_mean_baseline_best | mean | 1.302775 | 1.253277 | 1.302775 | 1.253277 | 0.000000 |

Next action: enter Phase 105 low-capacity mechanism design on target_intensity_std without opening A100 training
