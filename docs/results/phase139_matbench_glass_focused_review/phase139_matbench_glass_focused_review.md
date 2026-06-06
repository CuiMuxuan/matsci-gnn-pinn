# Phase 139 Matbench Glass Focused Review

- Status: `phase139_matbench_glass_focused_review_closed_split_sensitivity_or_shortcut`
- Split pass rate: `1`
- Blocking audits: `nearest_neighbor_dominant_split_count, class_balance_imbalanced_split_count`
- Low-capacity mechanism design allowed: `False`
- Model training allowed: `False`

## Blocking Audits

| Audit | Value | Threshold | Reason |
| --- | --- | --- | --- |
| nearest_neighbor_dominant_split_count | 3 | 0 | no viable split may be dominated by nearest-neighbor composition identity control |
| class_balance_imbalanced_split_count | 2 | 0 | no viable split may have severe train/validation/test glass class-balance imbalance |

## Split Reviews

| Split | Pass | Best profile | Val BA | Test BA | Shortcut | NN | Class shift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase138_registered_split | True | composition_descriptors | 0.76912 | 0.684187 | False | False | 0.0619597 |
| chemistry_family_hash_0 | True | common_plus_descriptors | 0.732189 | 0.771873 | False | False | 0.0798682 |
| chemistry_family_hash_1 | True | common_plus_descriptors | 0.753212 | 0.703957 | False | False | 0.0643657 |
| chemistry_family_hash_2 | True | composition_descriptors | 0.732181 | 0.711409 | False | False | 0.0923375 |
| dominant_element_hash | True | common_plus_descriptors | 0.646896 | 0.629109 | False | True | 0.128431 |
| element_count_bins | True | common_plus_descriptors | 0.644727 | 0.713269 | False | True | 0.208451 |
| entropy_bins | True | all_element_fractions | 0.743788 | 0.760077 | False | False | 0.163732 |
| transition_metal_bins | True | common_plus_descriptors | 0.695877 | 0.547444 | False | True | 0.157394 |
| metalloid_bins | True | common_plus_descriptors | 0.691029 | 0.711119 | False | False | 0.154577 |
| max_fraction_bins | True | common_plus_descriptors | 0.789496 | 0.687686 | False | False | 0.241197 |
