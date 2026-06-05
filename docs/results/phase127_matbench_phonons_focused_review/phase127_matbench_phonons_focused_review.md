# Phase 127 Matbench Phonons Focused Review

- Status: `phase127_matbench_phonons_focused_review_closed_split_sensitivity_or_shortcut`
- Split pass rate: `1`
- Blocking audits: `original_split_shortcut_dominance, shortcut_dominant_split_count, target_distribution_imbalanced_split_count`
- Low-capacity mechanism design allowed: `False`
- Model training allowed: `False`

## Blocking Audits

| Audit | Value | Threshold | Reason |
| --- | --- | --- | --- |
| original_split_shortcut_dominance | dominant_element_shortcut | negative val RMSE > admissible * 1.02 | composition, chemistry-family, or dominant-element shortcuts must not dominate |
| shortcut_dominant_split_count | 1 | 0 | no viable split may be dominated by composition, chemistry-family, or dominant-element shortcuts |
| target_distribution_imbalanced_split_count | 5 | 0 | no viable split may have severe train/validation/test phonon target mean, median, or q90 imbalance |

## Split Reviews

| Split | Pass | Best profile | Val RMSE | Test RMSE | Shortcut | NN | Target shift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase126_registered_split | True | composition_descriptors | 299.918 | 313.777 | True | False | 0.479841 |
| chemistry_family_hash_0 | True | composition_lattice_descriptors | 290.418 | 298.016 | False | False | 0.47349 |
| chemistry_family_hash_1 | True | composition_lattice_descriptors | 250.335 | 209.851 | False | False | 0.508655 |
| chemistry_family_hash_2 | True | composition_descriptors | 458.123 | 434.982 | False | False | 1.11142 |
| lattice_volume_bins | True | composition_lattice_descriptors | 365.641 | 311.994 | False | False | 0.900208 |
| volume_per_site_bins | True | composition_lattice_descriptors | 432.943 | 378.946 | False | False | 2.52433 |
| element_count_bins | True | composition_descriptors | 403.649 | 361.604 | False | False | 0.972225 |
| density_anisotropy_bins | True | composition_descriptors | 210.221 | 191.23 | False | False | 0.826417 |
