# Phase 162 UCI Steel Industry Energy Baseline Gate

## Gate
- Status: `phase162_uci_steel_industry_energy_closed_no_stable_guarded_gap`
- Phase 163 focused review allowed: `false`
- Model training allowed: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
This is a no-training baseline-first intake for a steel-industry energy source. The registered split holds out complete weeks. Synchronous electrical quantities, CO2, load labels, and row order are shortcut controls rather than model inputs for a publishable claim.

## Source Overview
| source_id | source_url | source_doi | raw_path | raw_bytes | raw_sha256 | field_rows | numeric_feature_columns | categorical_feature_columns | target | group_column | group_count | train_rows_split | val_rows_split | test_rows_split |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase162_uci_steel_industry_energy | https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip | 10.24432/C52G8C | data/raw/external/phase162_uci_steel_industry_energy/steel_industry_energy_consumption.zip | 481973 | d82d28b33780ff1582507fcf08ae764ff648af459d58234370c551e62aadeaef | 35040 | 16 | 6 | Usage_kWh | week_key | 53 | 20928 | 9408 | 4704 |

## Review
| target | selected_profile | selected_method | selected_validation_rmse | selected_test_rmse | mean_validation_rmse | mean_test_rmse | best_shortcut_profile | best_shortcut_method | best_shortcut_validation_rmse | best_shortcut_test_rmse | validation_relative_improvement_over_mean | test_relative_improvement_over_mean | baseline_visible_gap | shortcut_dominant | phase163_focused_review_allowed | status | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Usage_kWh | calendar_cycle_context | hist_gradient_boosting | 19.5257 | 21.7667 | 34.156 | 33.9662 | direct_electrical_proxy_control | extra_trees | 0.501293 | 0.648701 | 0.428338 | 0.359165 | true | true | false | phase162_uci_steel_industry_energy_closed_no_stable_guarded_gap | direct proxy/load-type/row-order shortcut is too close to or better than selected admissible profile |
