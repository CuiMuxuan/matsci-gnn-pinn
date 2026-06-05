# Phase 130 Matbench Log KVRH Baseline Gate

- Status: `phase130_matbench_log_kvrh_ready_focused_review`
- Target: `log10_k_vrh`
- Rows: `10985`
- Group split: `chemistry_family_key` with `1716` groups
- Selected profile/method: `composition_lattice_descriptors` / `hist_gradient_boosting`
- Selected validation/test RMSE: `0.14529` / `0.148177`
- Mean validation/test RMSE: `0.369918` / `0.375825`
- Best negative control: `dominant_element_shortcut` / `extra_trees`
- Focused review allowed: `True`
- Model training allowed: `False`
- A100 training allowed now: `False`

## Review Reason

safe structure/composition profile beats mean and shortcut controls
