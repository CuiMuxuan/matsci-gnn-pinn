# Phase 115 NIST AMMT Diagnostic Closure Package

- Status: `phase115_nist_ammt_diagnostic_closure_package_ready_all_new_branches_closed`
- Main paper floor: `Phase 55/60/74 broad_process_v1 fixed-sampling spot_size`
- Model mechanism allowed: `false`
- Model training allowed: `false`
- A100 training allowed now: `false`

## Evidence

| Phase | Branch | Status | Row count | Selected item | Closure reason |
|---|---|---|---:|---|---|
| 111 | registered_layer_camera_sequence | phase111_registered_target_closure_package_ready_sequence_branch_closed |  | Phase 106-110 closure package | sequence branch already closed as diagnostic |
| 112 | melt_pool_camera_target | phase112_melt_pool_target_gap_ready_focused_review | 63 | target_mp_temporal_mean_range | opened focused review only; training stayed locked |
| 113 | melt_pool_camera_target | phase113_melt_pool_focused_review_closed_validation_test_reversal |  | target_mp_temporal_mean_range | all Phase 112 candidates reversed versus mean guard on test |
| 114 | gcode_strategy_source | phase114_gcode_strategy_source_gate_closed_no_guarded_baseline_gap | 128 |  | no guarded baseline gap beyond XYPT guard and shortcut checks |

## Claim Use

| Claim | Use | Evidence status |
|---|---|---|
| nist_ammt_registered_intake_diagnostic_package | appendix_methods_and_negative_results | allowed_appendix_only |
| melt_pool_target_branch_closed | appendix_negative_result | closed_negative |
| gcode_strategy_source_branch_closed | appendix_negative_result | closed_negative |
| main_paper_floor_unchanged | main_text_floor | unchanged_positive_floor |

## Boundaries

| Boundary | Branch | Blocked item | Reason |
|---|---|---|---|
| phase115_no_training_on_phase112_melt_pool_targets | melt_pool_camera_target | target_mp_mean_mean, target_mp_q90_mean, target_mp_max_mean, target_mp_max_range, target_mp_temporal_mean_range | Phase 113 focused review found validation/test reversal versus mean guard |
| phase115_no_training_on_phase114_gcode_features | gcode_strategy_source | target_intensity_std:blocked_layer_time_strategy_shortcut; target_center_periphery_contrast:blocked_no_gain_over_xypt_guard; target_grid_mean_range:blocked_no_baseline_visible_gap; target_quadrant_contrast:blocked_no_baseline_visible_gap | Phase 114 found no candidate target after XYPT guard and shortcut checks |
| phase115_no_new_nist_ammt_main_claim | nist_ammt_diagnostics | new NIST AMMT main-text model claim | Phase 111, Phase 113, and Phase 114 all close as diagnostic or negative branches |
| phase115_no_a100_80gb_request | compute | A100-SXM4-80GB escalation | no seed-positive model branch or 40GB memory/runtime blockage exists |

Next action: do not train on Phase 112-114 NIST AMMT branches; return to manuscript consolidation or a fresh baseline-first data-source intake
