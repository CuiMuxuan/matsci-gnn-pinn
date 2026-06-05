# Phase 131 Matbench Log KVRH Focused Review

- Status: `phase131_matbench_log_kvrh_focused_review_closed_split_sensitivity_or_shortcut`
- Split pass rate: `1`
- Blocking audits: `target_distribution_imbalanced_split_count`
- Low-capacity mechanism design allowed: `False`
- Model training allowed: `False`

## Blocking Audits

| Audit | Value | Threshold | Reason |
| --- | --- | --- | --- |
| target_distribution_imbalanced_split_count | 3 | 0 | no viable split may have severe train/validation/test KVRH target mean, median, or q90 imbalance |

## Split Reviews

| Split | Pass | Best profile | Val RMSE | Test RMSE | Shortcut | NN | Target shift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase130_registered_split | True | composition_lattice_descriptors | 0.145212 | 0.148538 | False | False | 0.196506 |
| chemistry_family_hash_0 | True | composition_lattice_descriptors | 0.156137 | 0.139455 | False | False | 0.307082 |
| chemistry_family_hash_1 | True | composition_lattice_descriptors | 0.160348 | 0.149887 | False | False | 0.269826 |
| chemistry_family_hash_2 | True | composition_lattice_descriptors | 0.145963 | 0.153837 | False | False | 0.144954 |
| dominant_element_hash | True | composition_lattice_descriptors | 0.178558 | 0.168787 | False | False | 0.386818 |
| lattice_volume_bins | True | composition_lattice_descriptors | 0.206896 | 0.163782 | False | False | 1.03518 |
| volume_per_site_bins | True | composition_lattice_descriptors | 0.185698 | 0.425483 | False | False | 1.99119 |
| element_count_bins | True | composition_lattice_descriptors | 0.16341 | 0.161638 | False | False | 0.129173 |
| density_anisotropy_bins | True | composition_lattice_descriptors | 0.159633 | 0.196195 | False | False | 0.876407 |
