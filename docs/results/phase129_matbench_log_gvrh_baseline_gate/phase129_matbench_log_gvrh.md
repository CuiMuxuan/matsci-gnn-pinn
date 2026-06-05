# Phase 129 Matbench Log GVRH Baseline Gate

- Status: `phase129_matbench_log_gvrh_ready_focused_review`
- Target: `log10_g_vrh`
- Rows: `10985`
- Group split: `chemistry_family_key` with `1716` groups
- Selected profile/method: `composition_lattice_descriptors` / `hist_gradient_boosting`
- Selected validation/test RMSE: `0.160091` / `0.169371`
- Mean validation/test RMSE: `0.336058` / `0.374112`
- Best negative control: `dominant_element_shortcut` / `extra_trees`
- Focused review allowed: `True`
- Model training allowed: `False`
- A100 training allowed now: `False`

## Review Reason

safe structure/composition profile beats mean and shortcut controls
