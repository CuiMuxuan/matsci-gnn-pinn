# Phase 113 NIST AMMT Melt-Pool Focused Review

- Status: `phase113_melt_pool_focused_review_closed_validation_test_reversal`
- Phase 112 selected target: `target_mp_temporal_mean_range`
- Mechanism allowed targets: `none`
- Model training allowed: `false`
- A100 training allowed now: `false`

| Target | Focused review | Val gain vs mean | Test gain vs mean | Reason |
|---|---|---:|---:|---|
| target_mp_mean_mean | blocked_validation_test_reversal | 0.348809 | -0.421075 | validation-selected profile is worse than mean guard on test |
| target_mp_mean_std | not_phase112_candidate | 0.000000 | 0.000000 | blocked_no_baseline_visible_gap |
| target_mp_q90_mean | blocked_validation_test_reversal | 0.167302 | -0.077129 | validation-selected profile is worse than mean guard on test |
| target_mp_max_mean | blocked_validation_test_reversal | 0.124414 | -0.604234 | validation-selected profile is worse than mean guard on test |
| target_mp_max_range | blocked_validation_test_reversal | 0.246930 | -0.030084 | validation-selected profile is worse than mean guard on test |
| target_mp_temporal_mean_range | blocked_validation_test_reversal | 0.132899 | -0.135767 | validation-selected profile is worse than mean guard on test |
| target_mp_early_late_mean_delta | not_phase112_candidate | 0.010549 | 0.155311 | blocked_no_baseline_visible_gap |
| target_mp_peak_frame_position | not_phase112_candidate | 0.000000 | 0.000000 | blocked_no_baseline_visible_gap |

## Boundaries

| Boundary | Blocked item | Reason |
|---|---|---|
| phase113_no_training_on_phase112_candidates | Phase 112 melt-pool selected/candidate targets | validation/test reversal targets: target_mp_mean_mean, target_mp_q90_mean, target_mp_max_mean, target_mp_max_range, target_mp_temporal_mean_range |
| phase113_no_a100_80gb_request | A100-SXM4-80GB escalation | no model mechanism or seed-positive branch is open |

Next action: close melt-pool target branch as diagnostic; do not train
