# Phase 119 Battery Failure Candidate Sweep

- Status: `phase119_battery_failure_candidate_sweep_closed_all_phase117_candidates`
- Reviewed targets: `5`
- Allowed targets: `none`
- Model training allowed: `False`
- A100 training allowed now: `False`

## Target Sweep

| Target | Status | Allowed | Split pass | Blockers | Reason |
| --- | --- | --- | --- | --- | --- |
| Corrected-Total-Energy-Yield-kJ | closed_focused_review_blocked | False | 1 | original_split_negative_control_dominance, target_family_dependency | blocked by original_split_negative_control_dominance, target_family_dependency |
| Baseline-Plus-Heat-Loss-Total-Energy-Yield-kJ | closed_focused_review_blocked | False | 1 | original_split_negative_control_dominance, target_family_dependency | blocked by original_split_negative_control_dominance, target_family_dependency |
| Baseline-Total-Energy-Yield-kJ | closed_focused_review_blocked | False | 1 | original_split_negative_control_dominance, target_family_dependency | blocked by original_split_negative_control_dominance, target_family_dependency |
| Energy-Percent-Positive-Ejecta-% | closed_focused_review_blocked | False | 0.4 | original_split_negative_control_dominance, split_sensitivity_pass_rate | blocked by original_split_negative_control_dominance, split_sensitivity_pass_rate |
| Post-Test-Mass-Unrecovered-g | closed_focused_review_blocked | False | 0.8 | original_split_negative_control_dominance, target_family_dependency | blocked by original_split_negative_control_dominance, target_family_dependency |
