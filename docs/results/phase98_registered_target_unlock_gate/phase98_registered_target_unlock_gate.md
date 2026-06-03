# Phase 98 Registered Transfer Target Unlock Gate

## Gate Decision

Status: `registered_surrogate_unlocked_no_a100`.
Phase 99 local smoke allowed: `true`.
AM-Bench transfer unlocked: `false`.
A100 training allowed now: `false`.
A100-SXM4-80GB request now: `false`.

Phase 98 can unlock local smoke for a registered target or generated surrogate only. It is not an A100 training gate.

## Unlock Candidates

| Candidate | Dataset | Registration | Status | Phase 99 | Next action |
| --- | --- | --- | --- | --- | --- |
| phase98_generated_pfhub_registered_surrogate_v1 | phase96_pfhub_style_heat_source_v1 | registered_by_analytic_definition | phase99_local_smoke_ready_no_a100 | true | enter Phase 99 local baseline-first smoke on generated registered surrogate |
| phase98_unlock_phase97_ambench_mds2_2716_pad_thermography_xypt | mds2-2716 | pad thermography and pad XYPT exist, but current evidence has only independent-rescale diagnostics | blocked_registration_evidence_required | false | provide camera-to-galvo or equivalent coordinate registration before local smoke |
| phase98_unlock_phase97_external_public_registered_thermal_process_dataset | external_tbd | must_be_verified | blocked_source_manifest_data_card_required | false | provide public source manifest, registration story, split plan, and baseline plan |
| phase98_unlock_phase97_ambench_mds2_2716_single_track_scan_path | mds2-2716 | no single-track scan-path group or camera-pixel to galvo-mm mapping is available | blocked_registration_evidence_required | false | provide camera-to-galvo or equivalent coordinate registration before local smoke |
| phase98_unlock_phase97_current_ambench_spot_size_process_kernel | mds2-2716 | blocked_no_source_path_mapping | blocked_no_physical_mapping | false | do not inject heat-kernel/source-path features into broad spot_size without a registered source path or equivalent physical mapping |
| phase98_unlock_phase97_ambench_mds2_2718_exact_line_microstructure | mds2-2718 | not_registered_to_thermal_pixels_or_source_path | diagnostic_only | false | do not open GCN/image-encoder training without stronger physical alignment or a separate data card |
| phase98_unlock_phase97_pfhub_only_appendix_extension | pfhub_style_local | synthetic_registered_by_definition | handled_by_generated_surrogate_candidate | false | use generated registered surrogate row for local mechanism smoke |

## Protocol

| Protocol | Component | Pass | Stop |
| --- | --- | --- | --- |
| P98-PROT-001 | source_manifest | source is public or generated from committed code and records parameters | source cannot be regenerated or inspected |
| P98-PROT-002 | registration_story | coordinate mapping is exact, documented, or verified before local smoke | only independent-rescale or test-selected alignment is available |
| P98-PROT-003 | baseline_plan | mean/low-order, deterministic surrogate, and feature baselines are specified | candidate cannot be compared to a simpler baseline |
| P98-PROT-004 | compute_governance | A100 flags remain false | any route tries to open A100 before Phase 99 local smoke |

## Next Action

enter Phase 99 local baseline-first smoke on the generated registered surrogate
