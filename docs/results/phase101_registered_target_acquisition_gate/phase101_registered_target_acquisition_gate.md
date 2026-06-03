# Phase 101 Registered Target Acquisition Gate

## Gate Decision

Status: `blocked_no_real_registered_target`.
Phase 102 baseline smoke allowed: `false`.
AM-Bench transfer unlocked: `false`.
A100 training allowed now: `false`.
A100-SXM4-80GB request now: `false`.

Phase 101 requires a real registered AM-Bench or external target. The generated analytic surrogate is closed for transfer use.

## Target Rows

| Target | Dataset | Registration | Phase 102 | Status | Next action |
| --- | --- | --- | --- | --- | --- |
| phase101_ambench_mds2_2716_pad_thermography_xypt | mds2-2716 | pad thermography and pad XYPT exist, but current evidence has only independent-rescale diagnostics | false | blocked_registration_evidence_required | search for documented pad camera-to-galvo registration before fixed source-path features |
| phase101_external_public_registered_thermal_process_dataset | external_tbd | must_be_verified | false | blocked_source_manifest_data_card_required | create a public source manifest and registration data card before any model work |
| phase101_ambench_mds2_2716_single_track_scan_path | mds2-2716 | no single-track scan-path group or camera-pixel to galvo-mm mapping is available | false | blocked_registration_evidence_required | do not build source-path features for Line_* thermography; acquire aligned scan-path metadata or registration |
| phase101_p94_cand_exaca_sim | ExaCA cellular-automata solidification code | requires_generated_dataset_and_alignment_card | false | blocked_until_simulation_data_card | create an ExaCA data-card proposal before any generated-data model training |
| phase101_ambench_mds2_2718_exact_line_microstructure | mds2-2718 | not_registered_to_thermal_pixels_or_source_path | false | diagnostic_only_not_registered_transfer | do not open GCN/image-encoder training without stronger physical alignment or a separate data card |
| phase101_p94_cand_ext_thermal | external public registered thermal/process dataset | missing_data_card | false | blocked_no_external_data_card | provide or identify a public dataset with source manifest and registration story |
| phase101_generated_registered_surrogate_closed | phase98_generated_pfhub_registered_surrogate_v1 | registered_by_analytic_definition | false | closed_local_mechanism_not_transfer_target | do not treat analytic surrogate as AM-Bench/external transfer evidence |

## Manual Queue

| Queue | Target | Missing | Minimum evidence |
| --- | --- | --- | --- |
| P101-QUEUE-001 | phase101_ambench_mds2_2716_pad_thermography_xypt | coordinate_registration | camera-to-galvo mapping, scan-path alignment, or equivalent coordinate registration |
| P101-QUEUE-002 | phase101_external_public_registered_thermal_process_dataset | source_manifest_data_card | public source manifest, license/reproducibility note, split plan, and baseline plan |
| P101-QUEUE-003 | phase101_ambench_mds2_2716_single_track_scan_path | coordinate_registration | camera-to-galvo mapping, scan-path alignment, or equivalent coordinate registration |
| P101-QUEUE-004 | phase101_p94_cand_exaca_sim | source_manifest_data_card | public source manifest, license/reproducibility note, split plan, and baseline plan |
| P101-QUEUE-005 | phase101_ambench_mds2_2718_exact_line_microstructure | registered_transfer_evidence | proof that source/path features map physically to the target observations |
| P101-QUEUE-006 | phase101_p94_cand_ext_thermal | source_manifest_data_card | public source manifest, license/reproducibility note, split plan, and baseline plan |

## Protocol

| Protocol | Component | Requirement | Current status |
| --- | --- | --- | --- |
| P101-PROT-001 | real_transfer_target | A Phase 102 target must be a real AM-Bench or external registered target, not the generated analytic surrogate. | enforced_by_phase101 |
| P101-PROT-002 | source_manifest | Target needs a public source manifest or generated-data card with reproducibility and licensing notes. | missing_for_external_and_simulation_targets |
| P101-PROT-003 | registration | Source/path coordinates must map to target observations without test-label alignment. | missing_for_current_ambench_transfer_routes |
| P101-PROT-004 | compute_governance | Phase 101 may only open Phase 102 local baseline-first smoke, not A100 training. | a100_locked |

## Next Action

provide pad registration evidence or an external registered-target data card
