# Phase 161 UCI Steel Plates Faults Baseline Gate

## Gate
- Status: `phase161_uci_steel_plates_faults_closed_no_stable_guarded_gap`
- Phase 162 focused review allowed: `false`
- Model training allowed: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
This is a no-training baseline-first intake for a possible second-paper source. The split groups rows by coarse steel and defect-geometry context so closely related surface-defect contexts stay within a single split. A positive gate can only open a focused split/shortcut/class-balance review.

## Source Overview
| source_id | source_url | source_doi | raw_path | raw_bytes | raw_sha256 | field_rows | feature_columns | target | class_count | class_counts_json | group_column | group_count | train_rows_split | val_rows_split | test_rows_split |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase161_uci_steel_plates_faults | https://cdn.uci-ics-mlr-prod.aws.uci.edu/198/steel%2Bplates%2Bfaults.zip | 10.24432/C5J88N | data/raw/external/phase161_uci_steel_plates_faults/steel_plates_faults.zip | 100890 | cb8eb9859198b63f053e443513036b401746fa517ef58bd17c846c6741c93919 | 1941 | 39 | target_fault_class | 7 | {"Bumps": 402, "Dirtiness": 55, "K_Scatch": 391, "Other_Faults": 673, "Pastry": 158, "Stains": 72, "Z_Scratch": 190} | steel_geometry_context_key | 1023 | 1251 | 350 | 340 |

## Review
| target | selected_profile | selected_method | selected_validation_balanced_accuracy | selected_test_balanced_accuracy | selected_validation_macro_f1 | selected_test_macro_f1 | majority_validation_balanced_accuracy | majority_test_balanced_accuracy | majority_validation_macro_f1 | majority_test_macro_f1 | best_shortcut_profile | best_shortcut_method | best_shortcut_validation_balanced_accuracy | best_shortcut_test_balanced_accuracy | validation_balanced_accuracy_gain_over_majority | test_balanced_accuracy_gain_over_majority | validation_macro_f1_gain_over_majority | test_macro_f1_gain_over_majority | baseline_visible_gap | shortcut_dominant | phase162_focused_review_allowed | status | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| target_fault_class | geometry_luminosity_full | hist_gradient_boosting | 0.986628 | 0.971429 | 0.976404 | 0.974387 | 0.142857 | 0.142857 | 0.0673737 | 0.081203 | row_order_control | knn | 0.995355 | 0.98836 | 0.843771 | 0.828571 | 0.909031 | 0.893184 | true | true | false | phase161_uci_steel_plates_faults_closed_no_stable_guarded_gap | shortcut control is too close to or better than selected admissible profile |
