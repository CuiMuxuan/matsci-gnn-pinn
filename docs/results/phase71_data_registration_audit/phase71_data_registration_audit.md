# Phase 71 Data-Registration Audit

## Purpose

Phase 71 implements the Phase 68 `P68-DATA-REGISTRATION` action. It checks whether Candidate C can reopen heat-kernel, Green's-function, or source-path features before any A100 training.

## Candidate C Gate

Status: `blocked_by_registration_data`.
Open aligned feature gate: `false`.
Fixed feature gate allowed: `false`.
A100 training allowed now: `false`.
A100-SXM4-80GB request now: `false`.

Current AM-Bench evidence has pad XYPT but no paper-facing single-track registration or pad camera-to-galvo mapping.

## Audit Rows

| Source | Target | Source path | Registration | Feature route | Status | Blocker |
| --- | --- | --- | --- | --- | --- | --- |
| phase52_registered_source_path_gate | single_track_thermography | pad_xypt_only | not_registered | do_not_build_source_path_features | blocked_single_track_source_path | pad XYPT groups cannot be used as registered source paths for Line_* thermography without a documented mapping |
| phase53_source_path_inventory | broad_single_track_thermography | pad_xypt_only | not_registered | do_not_run_broad_source_path_validation | blocked_broad_source_path | no single-track scan-path groups and no HDF5 camera-pixel to galvo-mm registration metadata |
| phase53_pad_inventory | pad_thermography | pad_xypt | not_paper_registered | diagnostic_only | blocked_pad_registration | pad thermography and pad XYPT exist, but no HDF5 registration metadata was found |
| phase53_x_pad1_rescale_diagnostic | pad_thermography_x_pad1 | pad_xypt_xpad | not_paper_registered | failed_combined_metric_gate | diagnostic_global_regression | X_pad1 rescale diagnostic improves focused regions but worsens global RMSE |
| phase53_y_pad1_rescale_diagnostic | pad_thermography_y_pad1 | pad_xypt_ypad | not_paper_registered | failed_combined_metric_gate | diagnostic_all_metric_regression | Y_pad1 rescale diagnostic worsens global, hot, gradient, and/or coverage |
| phase60_next_branch_gate | candidate_c | heat_kernel_or_green_function_features | blocked_by_prior_evidence | Start with no-training-change feature gates before Macro PINN integration. | phase60_blocks_candidate_c | Requires aligned single-track scan-path metadata or a defensible pad thermography target. |
| phase68_candidate_signal_scorecard | candidate_c | data_aligned_physics_features | blocked_by_registration_data | inventory or add aligned scan-path/pad-thermography data and pass coordinate/unit/coverage checks before model changes | phase68_blocks_candidate_c | Phase 52/53 registration blocker plus Phase 60 next-branch gate |

## Next Action

do not train Candidate C; continue manuscript v0 audit or external data-registration planning
