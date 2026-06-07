# Phase 156 UCI Superconductivity Focused Review

## Gate
- Status: `phase156_uci_superconductivity_focused_review_ready_low_capacity_mechanism_gate`
- Viable split reviews: `6`
- Split pass rate: `1`
- Model mechanism allowed: `true`
- Model training allowed: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
This focused review checks whether the Phase 155 source gate is stable enough to justify a later no-training low-capacity mechanism gate. It does not train a neural model or support a second-paper model claim by itself.

## Split Review
| split_id | gate_review_split | best_admissible_profile | best_admissible_method | best_admissible_validation_rmse | best_admissible_test_rmse | shortcut_dominant | nearest_neighbor_dominant | target_distribution_shift_z | phase156_split_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase155_registered_element_set | true | weighted_feature_core | extra_trees | 12.4063968054 | 14.7296388305 | false | false | 0.451626985878 | true |
| element_set_hash_0 | true | element_fraction_vector | extra_trees | 12.2917680059 | 12.1567837025 | false | false | 0.118378835813 | true |
| element_set_hash_1 | true | element_fraction_vector | extra_trees | 11.5125994702 | 12.6320703037 | false | false | 0.198642825014 | true |
| element_set_hash_2 | true | element_fraction_vector | extra_trees | 11.9190503649 | 12.5290680805 | false | false | 0.117245080904 | true |
| element_set_hash_3 | true | element_fraction_vector | extra_trees | 11.787846633 | 12.8633313626 | false | false | 0.141607141904 | true |
| element_set_hash_4 | true | weighted_feature_core | extra_trees | 12.2388421839 | 14.25960772 | false | false | 0.202998070687 | true |
| dominant_element_holdout | false | element_fraction_vector | extra_trees | 4.69942109825 | 11.532930398 | false | false | 2.22620162352 | false |
| number_of_elements_bins | false | element_fraction_vector | extra_trees | 6.00880081102 | 17.6811963488 | false | false | 2.15157851525 | false |
| max_fraction_bins | false |  |  |  |  |  |  |  | false |

## Audits
| audit | status | severity | value | threshold | reason |
| --- | --- | --- | --- | --- | --- |
| phase155_gate_consistency | pass | info | phase155_uci_superconductivity_ready_focused_review | phase155_uci_superconductivity_ready_focused_review | Phase 156 requires a Phase 155 baseline-first gate that allowed focused review |
| stable_element_set_split_pass_rate | pass | info | 1 | 0.75 | baseline-visible signal should survive most leakage-safe element-set splits |
| registered_split_replay | pass | info | true | true | the Phase 155 registered split must replay as a guarded candidate |
| shortcut_dominant_split_count | pass | info | 0 | 0 | formula-shape, element-set hash, or row-order controls must not dominate |
| nearest_neighbor_dominant_split_count | pass | info | 0 | 0 | nearest-neighbor composition identity control must not explain the signal |
| target_distribution_imbalanced_split_count | pass | info | 0 | 0 | target mean/median/q90 shifts must not dominate focused-review interpretation |
