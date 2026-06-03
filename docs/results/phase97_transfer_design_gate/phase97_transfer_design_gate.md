# Phase 97 AM-Bench / External Transfer Design Gate

## Gate Decision

Status: `blocked_no_registered_transfer_target`.
Phase 98 local smoke allowed: `false`.
A100 training allowed now: `false`.
A100-SXM4-80GB request now: `false`.

Phase 97 is a transfer-design gate. It cannot turn Phase 96 synthetic evidence into A100 training.

## Transfer Routes

| Route | Target | Registration | Status | Phase 98 | Next action |
| --- | --- | --- | --- | --- | --- |
| phase97_ambench_mds2_2716_pad_thermography_xypt | ThermalData/X_pad* or Y_pad* pad thermography | pad thermography and pad XYPT exist, but current evidence has only independent-rescale diagnostics | blocked_missing_registration | false | search for documented pad camera-to-galvo registration before fixed source-path features |
| phase97_external_public_registered_thermal_process_dataset | public registered thermal/process target | must_be_verified | blocked_no_data_card | false | create a public source manifest and registration data card before any model work |
| phase97_ambench_mds2_2716_single_track_scan_path | ThermalData/Line_* single-track thermography | no single-track scan-path group or camera-pixel to galvo-mm mapping is available | blocked_missing_registration | false | do not build source-path features for Line_* thermography; acquire aligned scan-path metadata or registration |
| phase97_current_ambench_spot_size_process_kernel | current broad12/broad21 spot_size holdout | blocked_no_source_path_mapping | blocked_no_physical_mapping | false | do not inject heat-kernel/source-path features into broad spot_size without a registered source path or equivalent physical mapping |
| phase97_ambench_mds2_2718_exact_line_microstructure | single-track optical microscopy and melt-pool cross-section measurements | not_registered_to_thermal_pixels_or_source_path | diagnostic_prior_unstable | false | do not open GCN/image-encoder training without stronger physical alignment or a separate data card |
| phase97_pfhub_only_appendix_extension | PFHub-style synthetic benchmark only | synthetic_registered_by_definition | synthetic_appendix_only | false | keep as appendix/local mechanism evidence unless a registered target appears |

## Protocol

| Protocol | Component | Pass | Stop |
| --- | --- | --- | --- |
| P97-PROT-001 | physical_mapping | source, target, and coordinate systems are compatible without independent test-time rescaling | feature can only be interpreted as a synthetic basis with no target-data registration |
| P97-PROT-002 | baseline_contract | all baseline artifacts or commands are available before Phase 98 | candidate cannot be compared to the frozen floor or strongest baseline |
| P97-PROT-003 | leakage_control | train/validation-only selection and fixed split manifest | feature alignment or route selection depends on test performance |
| P97-PROT-004 | compute_governance | A100 flags remain false | any route tries to skip Phase 98 local baseline-first smoke |

## Next Action

resolve pad registration or add an external registered data card before Phase 98
