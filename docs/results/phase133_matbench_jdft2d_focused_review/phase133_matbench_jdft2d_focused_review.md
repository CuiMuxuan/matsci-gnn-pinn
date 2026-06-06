# Phase 133 Matbench JDFT2D Focused Review

- Status: `phase133_matbench_jdft2d_focused_review_closed_split_sensitivity_or_shortcut`
- Split pass rate: `0.714286`
- Blocking audits: `split_sensitivity_pass_rate, shortcut_dominant_split_count, target_distribution_imbalanced_split_count`
- Low-capacity mechanism design allowed: `False`
- Model training allowed: `False`

## Blocking Audits

| Audit | Value | Threshold | Reason |
| --- | --- | --- | --- |
| split_sensitivity_pass_rate | 0.714286 | 0.75 | JDFT2D target gain must survive deterministic chemistry, dominant-element, and lattice split perturbations |
| shortcut_dominant_split_count | 1 | 0 | no viable split may be dominated by composition, chemistry-family, or dominant-element shortcuts |
| target_distribution_imbalanced_split_count | 3 | 0 | no viable split may have severe train/validation/test JDFT2D target mean, median, or q90 imbalance |

## Split Reviews

| Split | Pass | Best profile | Val RMSE | Test RMSE | Shortcut | NN | Target shift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase132_registered_split | True | composition_lattice_descriptors | 96.9471 | 179.919 | False | False | 0.624223 |
| chemistry_family_hash_1 | True | composition_lattice_descriptors | 77.4852 | 112.603 | False | False | 0.387357 |
| chemistry_family_hash_2 | True | composition_lattice_descriptors | 86.2932 | 111.04 | False | False | 0.356284 |
| lattice_volume_bins | True | lattice_descriptors | 84.0807 | 98.6195 | False | False | 0.191785 |
| volume_per_site_bins | True | composition_lattice_descriptors | 69.6202 | 160.501 | True | False | 1.12966 |
| element_count_bins | False | composition_lattice_descriptors | 144.29 | 64.2945 | False | False | 0.776209 |
| density_anisotropy_bins | False | composition_lattice_descriptors | 101.031 | 93.316 | False | False | 1.14034 |
