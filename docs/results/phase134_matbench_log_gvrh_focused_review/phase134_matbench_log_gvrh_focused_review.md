# Phase 134 Matbench Log GVRH Focused Review

- Status: `phase134_matbench_log_gvrh_focused_review_closed_split_sensitivity_or_shortcut`
- Split pass rate: `1`
- Blocking audits: `target_distribution_imbalanced_split_count`
- Low-capacity mechanism design allowed: `False`
- Model training allowed: `False`

## Blocking Audits

| Audit | Value | Threshold | Reason |
| --- | --- | --- | --- |
| target_distribution_imbalanced_split_count | 2 | 0 | no viable split may have severe train/validation/test GVRH target mean, median, or q90 imbalance |

## Split Reviews

| Split | Pass | Best profile | Val RMSE | Test RMSE | Shortcut | NN | Target shift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase129_registered_split | True | composition_lattice_descriptors | 0.159455 | 0.169272 | False | False | 0.144027 |
| chemistry_family_hash_0 | True | composition_lattice_descriptors | 0.170128 | 0.165569 | False | False | 0.201707 |
| chemistry_family_hash_1 | True | composition_lattice_descriptors | 0.1708 | 0.175799 | False | False | 0.218457 |
| chemistry_family_hash_2 | True | composition_lattice_descriptors | 0.170536 | 0.157826 | False | False | 0.259773 |
| dominant_element_hash | True | composition_lattice_descriptors | 0.207408 | 0.192699 | False | False | 0.318163 |
| lattice_volume_bins | True | composition_lattice_descriptors | 0.169918 | 0.21087 | False | False | 0.792824 |
| volume_per_site_bins | True | composition_lattice_descriptors | 0.23726 | 0.285468 | False | False | 1.31132 |
| element_count_bins | True | composition_lattice_descriptors | 0.159399 | 0.206507 | False | False | 0.225818 |
| density_anisotropy_bins | True | composition_lattice_descriptors | 0.165383 | 0.15686 | False | False | 0.274958 |
