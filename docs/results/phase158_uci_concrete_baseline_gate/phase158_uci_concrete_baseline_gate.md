# Phase 158 UCI Concrete Baseline Gate

## Gate
- Status: `phase158_uci_concrete_ready_focused_review`
- Phase 159 focused review allowed: `true`
- Model training allowed: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
This is a no-training baseline-first intake for a possible second-paper positive mainline. The split groups rows by concrete mix design, excluding age, so the same formulation cannot appear across train/validation/test. A positive gate can only open a focused split/shortcut review.

## Source Overview
| source_id | source_url | source_doi | raw_path | raw_bytes | raw_sha256 | field_rows | feature_columns | target | group_column | group_count | train_rows_split | val_rows_split | test_rows_split |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase158_uci_concrete | https://cdn.uci-ics-mlr-prod.aws.uci.edu/165/concrete%2Bcompressive%2Bstrength.zip | 10.24432/C5PK67 | data/raw/external/phase158_uci_concrete/concrete_compressive_strength.zip | 34444 | dad85d14de8aee4e07479daa774e6b569a313715b71a3b92c95a07cf91c2c9a7 | 1030 | 30 | target_compressive_strength_mpa | mix_design_key | 428 | 652 | 186 | 192 |

## Review
| target | selected_profile | selected_method | selected_validation_rmse | selected_test_rmse | mean_validation_rmse | mean_test_rmse | best_shortcut_profile | best_shortcut_method | best_shortcut_validation_rmse | best_shortcut_test_rmse | validation_relative_improvement_over_mean | test_relative_improvement_over_mean | baseline_visible_gap | shortcut_dominant | phase159_focused_review_allowed | status | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| target_compressive_strength_mpa | full_concrete_features | hist_gradient_boosting | 5.665 | 5.366 | 16.95 | 17.09 | coarse_mix_presence_control | extra_trees | 8.12 | 8.213 | 0.6658 | 0.686 | true | false | true | phase158_uci_concrete_ready_focused_review |  |
