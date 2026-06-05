# Phase 118 Battery Failure Focused Review

- Status: `phase118_battery_failure_focused_review_closed_leakage_or_split_sensitivity`
- Selected target: `Post-Test-Mass-Unrecovered-g`
- Viable split reviews: `10`
- Split pass rate: `0.8`
- Blocking audits: `original_split_negative_control_dominance, target_family_dependency`
- Low-capacity mechanism design allowed: `False`
- Model training allowed: `False`
- A100 training allowed now: `False`

## Blocking Audits

| Audit | Status | Value | Threshold | Reason |
| --- | --- | --- | --- | --- |
| original_split_negative_control_dominance | block | target_family_plus_shortcuts | negative val RMSE > admissible * 1.02 | target-family or shortcut negative controls must not dominate the selected safe profile |
| target_family_dependency | block | 0.726988 | 0.7 | selected target must not be tightly coupled to other derived target-family columns |

## Split Review Summary

| Split | Group | Pass | Best profile | Best val RMSE | Best test RMSE | Negative dominates |
| --- | --- | --- | --- | --- | --- | --- |
| phase117_registered_split | Cell-Description | True | cell_trigger_safe | 4.28454 | 2.47258 | True |
| cell_description_hash_0 | Cell-Description | True | cell_pretest | 4.96171 | 4.25987 | False |
| cell_description_hash_1 | Cell-Description | False | cell_pretest | 5.01994 | 3.16445 | False |
| cell_description_hash_2 | Cell-Description | True | cell_pretest | 4.00722 | 6.08137 | True |
| cell_description_hash_3 | Cell-Description | True | cell_trigger_safe | 4.04057 | 4.93976 | False |
| cell_description_hash_4 | Cell-Description | True | cell_trigger_safe | 3.02809 | 5.88886 | True |
| test_series_hash | Test-Series | True | cell_pretest | 3.50689 | 5.81044 | False |
| s_ftrc_generation_hash | S-FTRC-Generation | True | cell_pretest | 3.57477 | 3.38558 | False |
| trigger_mechanism_hash | Trigger-Mechanism | False | cell_pretest | 3.21405 | 14.3432 | False |
| cell_format_hash | Cell-Format | True | cell_pretest | 4.51422 | 8.89001 | False |
