# Phase 155 UCI Superconductivity Baseline Gate

## Gate
- Status: `phase155_uci_superconductivity_ready_focused_review`
- Phase 156 focused review allowed: `true`
- Model training allowed: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
This is a no-training baseline-first intake for a possible second-paper positive mainline. A positive gate can only open a focused split/shortcut review; it does not open neural model training.

## Source Overview
| source_id | source_url | source_doi | raw_path | raw_bytes | raw_sha256 | train_rows | unique_rows | feature_columns | target | group_column | group_count | train_rows_split | val_rows_split | test_rows_split |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase155_uci_superconductivity | https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip | 10.24432/C53P47 | data/raw/external/phase155_uci_superconductivity/superconductivty_data.zip | 8300005 | 87f4490d73390ff94ee01dbf0d7d32abc80b22f2c803d471765cfc46a9f6371e | 21263 | 21263 | 173 | target_critical_temp_K | element_set_key | 3365 | 11644 | 3661 | 5958 |

## Review
| target | selected_profile | selected_method | selected_validation_rmse | selected_test_rmse | mean_validation_rmse | mean_test_rmse | best_shortcut_profile | best_shortcut_method | best_shortcut_validation_rmse | best_shortcut_test_rmse | validation_relative_improvement_over_mean | test_relative_improvement_over_mean | baseline_visible_gap | shortcut_dominant | phase156_focused_review_allowed | status | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| target_critical_temp_K | weighted_feature_core | extra_trees | 12.4063968054 | 14.7296388305 | 35.461176676 | 37.5865491094 | formula_shape_control | extra_trees | 23.337244344 | 25.8340057752 | 0.650141423146 | 0.608114094549 | true | false | true | phase155_uci_superconductivity_ready_focused_review |  |
