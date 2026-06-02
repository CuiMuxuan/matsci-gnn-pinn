# Phase 81 Registered-Target Intake Gate

## Gate Decision

Status: `blocked_no_registered_target`.
Phase 82 baseline smoke allowed: `false`.
Phase 83 registered feature gate allowed: `false`.
A100 training allowed now: `false`.
A100-SXM4-80GB request now: `false`.
Preferred next route: `ambench_mds2_2716_pad_thermography_xypt`.

No current route has public reproducibility, split readiness, process metadata, and coordinate-compatible source/target registration at the same time.

## Intake Routes

| Route | Target | Source | Registration | Status | Paper use | Next action |
| --- | --- | --- | --- | --- | --- | --- |
| ambench_mds2_2716_single_track_scan_path | ThermalData/Line_* single-track thermography | single-track scan path or camera-to-galvo registration | no single-track scan-path group or camera-pixel to galvo-mm mapping is available | blocked_missing_registration | appendix_or_future_data_requirement | do not build source-path features for Line_* thermography; acquire aligned scan-path metadata or registration |
| ambench_mds2_2716_pad_thermography_xypt | ThermalData/X_pad* or Y_pad* pad thermography | XYPT/Xpad or XYPT/Ypad scan strategy | pad thermography and pad XYPT exist, but current evidence has only independent-rescale diagnostics | blocked_missing_registration | highest_priority_data_followup | search for documented pad camera-to-galvo registration before fixed source-path features |
| ambench_mds2_2718_exact_line_microstructure | single-track optical microscopy and melt-pool cross-section measurements | exact-line P3/P4 Line_0_1 TIFF panel | not_registered_to_thermal_pixels_or_source_path | diagnostic_prior_unstable | appendix_diagnostic_or_separate_microstructure_branch | do not open GCN/image-encoder training without stronger physical alignment or a separate data card |
| external_public_registered_thermal_process_dataset | public registered thermal/process target | aligned scan path, source command, or camera-to-galvo calibration | must_be_verified | blocked_no_data_card | future_registered_target_or_second_paper_branch | create a public source manifest and registration data card before any model work |

## Next Action

do not run A100 model training; pursue pad camera-to-galvo registration or an external registered-target data card
