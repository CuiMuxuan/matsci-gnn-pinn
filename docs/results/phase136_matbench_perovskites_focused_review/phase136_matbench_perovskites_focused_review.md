# Phase 136 Matbench Perovskites Focused Review

- Status: `phase136_matbench_perovskites_focused_review_closed_split_sensitivity_or_shortcut`
- Split pass rate: `0.888889`
- Blocking audits: `target_distribution_imbalanced_split_count`
- Low-capacity mechanism design allowed: `False`
- Model training allowed: `False`

## Blocking Audits

| Audit | Value | Threshold | Reason |
| --- | --- | --- | --- |
| target_distribution_imbalanced_split_count | 3 | 0 | no viable split may have severe train/validation/test perovskites target mean, median, or q90 imbalance |

## Split Reviews

| Split | Pass | Best profile | Val RMSE | Test RMSE | Shortcut | NN | Target shift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase135_registered_split | True | composition_lattice_descriptors | 0.487052 | 0.412936 | False | False | 0.190736 |
| chemistry_family_hash_0 | True | composition_lattice_descriptors | 0.425859 | 0.412082 | False | False | 0.388505 |
| chemistry_family_hash_1 | True | composition_lattice_descriptors | 0.467763 | 0.463902 | False | False | 0.298223 |
| chemistry_family_hash_2 | True | composition_lattice_descriptors | 0.440947 | 0.400151 | False | False | 0.655669 |
| dominant_element_hash | False | composition_lattice_descriptors | 0.610171 | 0.868408 | False | False | 0.957174 |
| lattice_volume_bins | True | composition_lattice_descriptors | 0.515785 | 0.427273 | False | False | 0.833657 |
| volume_per_site_bins | True | composition_lattice_descriptors | 0.413559 | 0.649108 | False | False | 1.44064 |
| element_count_bins | True | composition_lattice_descriptors | 0.457187 | 0.390596 | False | False | 0.337916 |
| density_anisotropy_bins | True | composition_lattice_descriptors | 0.38195 | 0.415456 | False | False | 0.571198 |
