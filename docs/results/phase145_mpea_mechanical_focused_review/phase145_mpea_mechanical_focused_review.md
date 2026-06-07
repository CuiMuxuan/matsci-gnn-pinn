# Phase 145 MPEA Mechanical Focused Review

- Status: `phase145_mpea_mechanical_focused_review_closed_split_sensitivity_or_shortcut`
- Selected target: `hardness_hv`
- Viable split reviews: `10`
- Split pass rate: `0.5`
- Model mechanism allowed: `False`
- Model training allowed: `False`
- A100 80GB request now: `False`

## Split Review

| Split | Status | Best profile | Best method | Val RMSE | Test RMSE | Negative | Target shift | Formula cross | Pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| registered_formula_hash | ready_focused_review | composition_process_context | hist_gradient_boosting | 93.2854 | 137.532 | process_only_control | 0.138126 | 0 | True |
| formula_hash_salt_0 | ready_focused_review | composition_process_context | extra_trees | 110.239 | 136.661 | process_only_control | 0.343125 | 0 | True |
| formula_hash_salt_1 | ready_focused_review | composition_process_context | hist_gradient_boosting | 105.08 | 109.663 | process_only_control | 0.306654 | 0 | True |
| formula_hash_salt_2 | ready_focused_review | composition_process_context | hist_gradient_boosting | 96.4945 | 102.032 | process_only_control | 0.588868 | 0 | True |
| chemistry_family | ready_focused_review | composition_process_context | hist_gradient_boosting | 118.353 | 157.192 | process_only_control | 0.320529 | 0 | True |
| reference_holdout | blocked_shortcut_or_process_control_dominance | common_element_fractions | hist_gradient_boosting | 144.684 | 161.855 | process_only_control | 0.57348 | 0.0530504 | False |
| formula_reference_holdout | blocked_shortcut_or_process_control_dominance | composition_process_context | hist_gradient_boosting | 96.2376 | 102.559 | process_only_control | 0.184136 | 0.0503979 | False |
| processing_method_holdout | blocked_shortcut_or_process_control_dominance | composition_descriptors | knn | 103.965 | 180.465 | process_only_control | 2.08943 | 0.0795756 | False |
| phase_family_holdout | blocked_insufficient_rows_or_split |  |  |  |  |  | 0.935352 | 0.0185676 | False |
| test_type_holdout | blocked_insufficient_rows_or_split |  |  |  |  |  | 1.82916 | 0.0238727 | False |
| microstructure_holdout | ready_focused_review | common_element_fractions | hist_gradient_boosting | 153.678 | 204.805 | process_only_control | 1.01377 | 0.0477454 | False |
| process_phase_holdout | ready_focused_review | composition_process_context | extra_trees | 148.683 | 68.2932 | process_only_control | 1.26276 | 0.0742706 | False |

## Audits

| Audit | Status | Severity | Value | Threshold | Reason |
| --- | --- | --- | --- | --- | --- |
| phase144_gate_consistency | pass | info | phase144_mpea_mechanical_ready_focused_review | phase144_mpea_mechanical_ready_focused_review | focused review requires a positive Phase 144 baseline-first gate |
| registered_split_replay | pass | info | ready_focused_review | registered split must remain a guarded candidate | Phase 145 must reproduce the Phase 144 selected split without training |
| stable_split_pass_rate | block | blocking | 0.5 | 0.75 | guarded gain should survive most viable formula/reference/process split reviews |
| shortcut_or_process_control_dominant_split_count | block | blocking | 3 | 0 | process-only, formula, reference, or dominant-element controls must not dominate viable splits |
| target_distribution_imbalanced_split_count | block | blocking | 3 | 0 | split target mean/median/q90 shifts must not dominate interpretation |
| formula_cross_split_fraction_count | pass | info | 0 | 0 | focused review should not rely on many formula identities crossing train/val/test |
