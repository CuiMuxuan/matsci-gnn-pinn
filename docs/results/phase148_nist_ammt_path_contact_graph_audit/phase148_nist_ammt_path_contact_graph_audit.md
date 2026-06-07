# Phase 148 NIST AMMT Path-Contact Graph Audit

- Status: `phase148_path_contact_graph_audit_closed_no_guarded_graph_gap`
- Target: `target_center_periphery_contrast`
- Selected feature profile: `None`
- Best control profile: `camera_layer_time_control`
- Focused review allowed: `False`
- Model mechanism allowed: `false`
- Model training allowed: `false`
- A100 training allowed now: `false`

| Feature profile | Role | Status | Method | Val RMSE | Test RMSE | Val gain vs guard | Val gain vs control |
|---|---|---|---|---:|---:|---:|---:|
| phase106_guard_replay | guard_replay | control_or_guard_profile | hist_gradient_boosting | 1.174314337940004 | 1.3827508688560513 | 0.0 | -0.20224168944910625 |
| scalar_source_control | control | control_or_guard_profile | hist_gradient_boosting | 1.174314337940004 | 1.3827508688560513 | 0.0 | -0.20224168944910625 |
| layer_time_control | control | control_or_guard_profile | hist_gradient_boosting | 1.206224989467648 | 1.4647156871692277 | -0.03191065152764394 | -0.2341523409767502 |
| camera_layer_time_control | control | control_or_guard_profile | hist_gradient_boosting | 0.9720726484908978 | 1.535568940135318 | 0.20224168944910625 | 0.0 |
| path_contact_graph_shuffled | control | control_or_guard_profile | hist_gradient_boosting | 1.0673110025154144 | 1.4205136430245549 | 0.10700333542458962 | -0.09523835402451664 |
| path_contact_graph_ordered | candidate | blocked_no_validation_gain_over_phase106_guard | hist_gradient_boosting | 1.255095011610485 | 1.3581150777081423 | -0.08078067367048103 | -0.2830223631195873 |
| path_contact_graph_full | candidate | blocked_no_validation_gain_over_phase106_guard | hist_gradient_boosting | 1.2837133762778044 | 1.341997214561371 | -0.10939903833780029 | -0.31164072778690655 |

Next action: close CAPL-inspired path-contact graph descriptors as diagnostic; try the next literature route only through a fresh no-training gate
