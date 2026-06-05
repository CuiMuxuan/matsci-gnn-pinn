# Phase 111 NIST AMMT Registered-Target Closure Package

- Status: `phase111_registered_target_closure_package_ready_sequence_branch_closed`
- NIST AMMT sequence branch closed: `True`
- Main paper floor: `Phase 55/60/74 broad_process_v1 fixed-sampling spot_size`
- Model training allowed: `false`
- A100 training allowed now: `false`

## Evidence

| Phase | Status | Target/Profile | Val metric | Test metric | Interpretation |
|---|---|---|---:|---:|---|
| 106 | phase106_spatial_target_gap_ready_focused_no_training_validation | target_center_periphery_contrast | 1.174314 | 1.382751 | spatial target gap found; opens review only |
| 107 | phase107_source_region_feature_gate_blocked_no_phase106_gain |  |  |  | source-region features did not clear Phase 106 guard |
| 108 | phase108_sequence_target_gap_ready_focused_review | target_cp_camera_pair_delta | 1.809450 | 1.809990 | sequence target candidate required focused review |
| 109 | phase109_sequence_target_focused_review_closed_camera_shortcut | target_cp_camera_pair_delta | 1.809450 | 1.809990 | selected sequence target closed as camera/layer-time shortcut |
| 110 | phase110_layer_mean_target_review_closed_layer_time_shortcut | target_cp_layer_mean | 0.940098 | 1.060691 | alternate layer-mean target closed as layer/time shortcut |

## Claim Use

| Claim | Use | Evidence status |
|---|---|---|
| nist_ammt_registered_target_intake_reproducible | appendix_methods_or_data_diagnostic | allowed_appendix_only |
| spatial_target_gap_diagnostic | appendix_future_work_diagnostic | diagnostic_only |
| sequence_target_branch_negative | appendix_negative_result | closed_negative |
| main_paper_floor_remains_phase55_60_74 | main_text_floor | unchanged_positive_floor |

## Boundaries

| Boundary | Blocked item | Reason |
|---|---|---|
| no_training_on_phase106_spatial_gap_alone | A100 model training on target_center_periphery_contrast | spatial target gap required mechanism review and Phase 107 source-region gate failed |
| no_training_on_source_region_features | sampled source-region feature model branch | close sampled source-region path features as diagnostic; do not train |
| no_training_on_camera_pair_delta | target_cp_camera_pair_delta | camera/layer-time shortcut matched full validation result |
| no_training_on_layer_mean_sequence_target | target_cp_layer_mean | layer_time_only and layer_time_camera beat full validation; source-only had no independent gain |
| nist_ammt_sequence_branch_closed | NIST AMMT sequence target branch | selected and alternate sequence targets collapsed under shortcut review |

Next action: return to manuscript/appendix packaging or open a new registered target/data source; do not train on Phase 106-110 targets
