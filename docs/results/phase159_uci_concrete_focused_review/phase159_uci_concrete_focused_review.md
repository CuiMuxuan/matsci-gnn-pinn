# Phase 159 UCI Concrete Focused Review

## Gate
- Status: `phase159_uci_concrete_focused_review_ready_low_capacity_mechanism_gate`
- Viable split reviews: `6`
- Split pass rate: `1`
- Model mechanism allowed: `true`
- Model training allowed: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
This focused review checks whether the Phase 158 concrete source gate is stable enough to justify a later no-training low-capacity mechanism gate. It does not train a neural model or support a second-paper model claim by itself.

## Split Review
| split_id | gate_review_split | best_admissible_profile | best_admissible_method | best_admissible_validation_rmse | best_admissible_test_rmse | shortcut_dominant | nearest_neighbor_dominant | target_distribution_shift_z | phase159_split_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase158_registered_mix_design | true | full_concrete_features | hist_gradient_boosting | 5.665 | 5.366 | false | false | 0.3407 | true |
| mix_design_hash_0 | true | full_concrete_features | hist_gradient_boosting | 6.223 | 4.941 | false | false | 0.2528 | true |
| mix_design_hash_1 | true | binder_ratio_age_core | hist_gradient_boosting | 5.384 | 5.547 | false | false | 0.2665 | true |
| mix_design_hash_2 | true | raw_mix_age | hist_gradient_boosting | 4.797 | 6.139 | false | false | 0.3807 | true |
| mix_design_hash_3 | true | full_concrete_features | extra_trees | 5.724 | 5.331 | false | false | 0.2293 | true |
| mix_design_hash_4 | true | full_concrete_features | hist_gradient_boosting | 4.418 | 5.64 | false | false | 0.4137 | true |
| age_bucket_holdout | false | binder_ratio_age_core | extra_trees | 5.29 | 8.734 | false | false | 0.7049 | true |
| water_binder_bins | false |  |  |  |  |  |  |  | false |
| binder_mass_bins | false |  |  |  |  |  |  |  | false |

## Audits
| audit | status | severity | value | threshold | reason |
| --- | --- | --- | --- | --- | --- |
| phase158_gate_consistency | pass | info | phase158_uci_concrete_ready_focused_review | phase158_uci_concrete_ready_focused_review | Phase 159 requires a Phase 158 baseline-first gate that allowed focused review |
| stable_mix_design_split_pass_rate | pass | info | 1 | 0.75 | baseline-visible signal should survive most leakage-safe mix-design splits |
| registered_split_replay | pass | info | true | true | the Phase 158 registered split must replay as a guarded candidate |
| shortcut_dominant_split_count | pass | info | 0 | 0 | age-only, coarse mix, mix-design hash, or row-order controls must not dominate |
| nearest_neighbor_dominant_split_count | pass | info | 0 | 0 | nearest-neighbor concrete mixture identity control must not explain the signal |
| target_distribution_imbalanced_split_count | pass | info | 0 | 0 | target mean/median/q90 shifts must not dominate focused-review interpretation |
