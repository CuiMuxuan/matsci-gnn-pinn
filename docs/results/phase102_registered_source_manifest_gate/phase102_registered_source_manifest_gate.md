# Phase 102 Registered Source-Manifest/Data-Card Gate

## Gate Decision

Status: `source_manifest_ready_phase103_intake_allowed`.
Preferred candidate: `P102-CAND-NIST-AMMT-3D-SCAN`.
Phase 103 intake allowed: `true`.
Large server download allowed for Phase 103: `true`.
Phase 104 baseline smoke allowed: `false`.
A100 training allowed now: `false`.
A100-SXM4-80GB request now: `false`.

Phase 102 only opens data intake/audit. It does not open baseline smoke or model training.

## Candidate Sources

| Candidate | Dataset | Manifest | Registration | Phase 103 | Status |
| --- | --- | --- | --- | --- | --- |
| P102-CAND-NIST-AMMT-3D-SCAN | nist_mds2_2044 | ready_official_file_manifest | claimed_by_official_dataset_description_pending_metadata_audit | true | source_manifest_ready_phase103_intake_allowed |
| P102-CAND-AMBENCH-PAD-XYPT | mds2-2716 | present | blocked_missing_camera_to_galvo_or_equivalent_mapping | false | blocked_registration_evidence_required |
| P102-CAND-EXACA-SIM-DATA-CARD | ExaCA cellular-automata solidification code | missing_generated_target_manifest | requires_generated_dataset_and_alignment_card | false | blocked_until_simulation_manifest |

## NIST AMMT File Manifest

| File | GiB | Required first | Scope | Status |
| --- | --- | --- | --- | --- |
| Metadata.zip | 0.002318 | true | minimal_registration_metadata_intake | manifested_not_downloaded |
| Build Command Data.zip | 6.909898 | false | long_running_server_download_after_metadata_pass | manifested_not_downloaded |
| In-situ Meas Data.zip | 8.540620 | false | long_running_server_download_after_metadata_pass | manifested_not_downloaded |
| Movies.zip | 0.650952 | false | optional_visual_context_after_registration_pass | manifested_not_downloaded |

## Registration Checks

| Check | Component | Status | Phase 103 check |
| --- | --- | --- | --- |
| P102-REG-001 | source_path_commands | manifested_pending_download | inspect Metadata.zip and Build Command Data schemas for command coordinates, units, and timestamps |
| P102-REG-002 | target_observations | manifested_pending_download | inspect In-situ Meas Data schemas after Metadata.zip confirms frame/timing references |
| P102-REG-003 | coordinate_transform | critical_pending_metadata_audit | locate transform files and verify coordinate frame, units, orientation, and invertibility |
| P102-REG-004 | trigger_timing | critical_pending_metadata_audit | verify command/monitoring clocks and trigger alignment |
| P102-REG-005 | split_safety | draft_pending_schema | derive train/validation/test split keys from build segments, commands, or time blocks |

## Phase 103 Queue

| Queue | Task | Location | Pass |
| --- | --- | --- | --- |
| P102-INTAKE-001 | download Metadata.zip and write checksum/content-length audit | A100 server /root/matsci-gnn-pinn/data/raw/nist_ammt/mds2_2044 | metadata contains coordinate transform/timing/schema files sufficient to plan joins |
| P102-INTAKE-002 | start resumable long-running server downloads for Build Command Data and In-situ Meas Data only after metadata inventory starts | A100 server /root/matsci-gnn-pinn/data/raw/nist_ammt/mds2_2044 | large packages are present and command/target schemas can be sampled |
| P102-INTAKE-003 | extract a tiny registered sample table | A100 server /root/matsci-gnn-pinn/data/raw/nist_ammt/mds2_2044 | source/path features map to target observations without independent rescaling |

## Protocol

| Protocol | Component | Current | Stop |
| --- | --- | --- | --- |
| P102-PROT-001 | no_training | enforced | any model training, baseline smoke, or A100 training is started in Phase 102 |
| P102-PROT-002 | public_provenance | satisfied_for_nist_ammt | private or non-reproducible source |
| P102-PROT-003 | registration_evidence | manifested_pending_phase103_metadata_audit | registration depends on independent rescaling or target-label fitting |
| P102-PROT-004 | large_downloads | allowed_by_user_for_server | large files are downloaded locally or without audit manifests |

## Next Action

enter Phase 103 minimal registered data intake/audit on the NIST AMMT metadata package
