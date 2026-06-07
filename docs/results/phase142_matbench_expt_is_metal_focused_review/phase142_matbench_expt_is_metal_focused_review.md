# Phase 142 Matbench Experimental Is-Metal Focused Review

- Status: `phase142_matbench_expt_is_metal_focused_review_closed_split_sensitivity_or_shortcut`
- Split pass rate: `1`
- Blocking audits: `nearest_neighbor_dominant_split_count, class_balance_imbalanced_split_count`
- Low-capacity mechanism design allowed: `False`
- Model training allowed: `False`

## Blocking Audits

| Audit | Value | Threshold | Reason |
| --- | --- | --- | --- |
| nearest_neighbor_dominant_split_count | 1 | 0 | no viable split may be dominated by nearest-neighbor composition identity control |
| class_balance_imbalanced_split_count | 5 | 0 | no viable split may have severe train/validation/test experimental is-metal class-balance imbalance |

## Split Reviews

| Split | Pass | Best profile | Val BA | Test BA | Shortcut | NN | Class shift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase141_registered_split | True | composition_descriptors | 0.897955 | 0.883607 | False | False | 0.0113252 |
| chemistry_family_hash_0 | True | composition_descriptors | 0.88445 | 0.893708 | False | False | 0.0858087 |
| chemistry_family_hash_1 | True | composition_descriptors | 0.892456 | 0.869901 | False | True | 0.0470955 |
| chemistry_family_hash_2 | True | common_plus_descriptors | 0.890851 | 0.881771 | False | False | 0.073716 |
| dominant_element_hash | True | composition_descriptors | 0.898352 | 0.813078 | False | False | 0.331915 |
| element_count_bins | True | composition_descriptors | 0.869211 | 0.841432 | False | False | 0.286338 |
| entropy_bins | True | composition_descriptors | 0.891321 | 0.822112 | False | False | 0.200175 |
| transition_metal_bins | True | composition_descriptors | 0.869124 | 0.833399 | False | False | 0.35 |
| metalloid_bins | True | common_plus_descriptors | 0.892998 | 0.8995 | False | False | 0.0499772 |
| anion_fraction_bins | True | common_plus_descriptors | 0.842008 | 0.737401 | False | False | 0.338365 |
| max_fraction_bins | True | common_plus_descriptors | 0.923474 | 0.813685 | False | False | 0.139211 |
