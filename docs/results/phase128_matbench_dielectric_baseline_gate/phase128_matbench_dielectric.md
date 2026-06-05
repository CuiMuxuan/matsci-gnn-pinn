# Phase 128 Matbench Dielectric Baseline Gate

- Status: `phase128_matbench_dielectric_n_closed_no_stable_guarded_gap`
- Target: `refractive_index_n`
- Rows: `4764`
- Group split: `chemistry_family_key` with `909` groups
- Selected profile/method: `composition_lattice_descriptors` / `knn`
- Selected validation/test RMSE: `1.36571` / `1.80212`
- Mean validation/test RMSE: `1.49972` / `1.88481`
- Best negative control: `dominant_element_shortcut` / `extra_trees`
- Focused review allowed: `False`
- Model training allowed: `False`
- A100 training allowed now: `False`

## Review Reason

composition, family, or dominant-element shortcut dominates the selected safe profile
