# Phase 114 NIST AMMT G-code Strategy Source Gate

- Status: `phase114_gcode_strategy_source_gate_closed_no_guarded_baseline_gap`
- Row count: `128`
- Strategy count: `12`
- Selected target: `None`
- Focused review allowed: `False`
- Model training allowed: `false`
- A100 training allowed now: `false`

| Target | Status | Profile | Method | Val RMSE | XYPT guard val RMSE | Test gain vs guard |
|---|---|---|---|---:|---:|---:|
| target_intensity_std | blocked_layer_time_strategy_shortcut | gcode_all | knn | 1.926496 | 1.991487 | 0.077710 |
| target_center_periphery_contrast | blocked_no_gain_over_xypt_guard | gcode_strategy_params | extra_trees | 1.519917 | 1.325651 | -0.113991 |
| target_grid_mean_range | blocked_no_baseline_visible_gap | gcode_strategy_params | hist_gradient_boosting | 3.671388 | 3.889820 | -0.008420 |
| target_quadrant_contrast | blocked_no_baseline_visible_gap | gcode_strategy_params | hist_gradient_boosting | 2.988414 | 3.047936 | 0.049404 |

Next action: close G-code strategy source gate as diagnostic; do not train
