# Phase 144 MPEA Mechanical Baseline Gate

- Status: `phase144_mpea_mechanical_ready_focused_review`
- Source rows: `1545`
- Selected target: `hardness_hv`
- Focused review allowed: `True`
- Model training allowed: `False`
- A100 training allowed now: `False`
- A100 80GB request now: `False`

## Target Review

| Target | Status | Rows | Best profile | Best method | Val RMSE | Test RMSE | Negative profile | Shortcut blocks | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hardness_hv | ready_focused_review | 530 | composition_process_context | hist_gradient_boosting | 92.386 | 138.129 | process_only_control | False | admissible profile beats mean, preserves test gain, and is not control-dominated |
| yield_strength_mpa | ready_focused_review | 1067 | composition_process_context | hist_gradient_boosting | 271.464 | 349.722 | process_only_control | False | admissible profile beats mean, preserves test gain, and is not control-dominated |
| ultimate_tensile_strength_mpa | blocked_shortcut_or_process_control_dominance | 539 | composition_process_context | extra_trees | 331.482 | 352.547 | process_only_control | True | best negative-control profile matches or beats the admissible validation RMSE |
| elongation_pct | blocked_shortcut_or_process_control_dominance | 619 | composition_process_context | extra_trees | 16.7538 | 15.6364 | process_only_control | True | best negative-control profile matches or beats the admissible validation RMSE |
| young_modulus_gpa | blocked_insufficient_rows_or_split | 145 |  |  |  |  |  |  | target or one split is below the minimum row count |
