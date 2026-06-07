# Phase 143 Paper Evidence Refresh

- Status: `phase143_paper_evidence_refresh_ready_first_paper_narrow_claims`
- First paper draft allowed now: `True`
- New external model claim ready: `False`
- Model training allowed: `False`
- A100 80GB request now: `False`

## Latest External Diagnostics

| Branch | Status | Final use | Boundary |
| --- | --- | --- | --- |
| matbench_glass | phase139_matbench_glass_focused_review_closed_split_sensitivity_or_shortcut | appendix_negative_external_diagnostic | nearest-neighbor identity and class-balance audits block glass model claims |
| matbench_mp_is_metal_large_source | phase140_matbench_mp_is_metal_real_gate_missing_source_acquisition_blocked | blocked_source_acquisition_diagnostic | large-source HTTPS acquisition blocked before any real gate artifact; not a compute or model result |
| matbench_expt_is_metal | phase142_matbench_expt_is_metal_focused_review_closed_split_sensitivity_or_shortcut | appendix_negative_external_diagnostic | nearest-neighbor identity and class-balance audits block experimental is-metal model claims |

## Claim Boundary

| Claim | Status | Allowed use | Guard |
| --- | --- | --- | --- |
| P143-CLAIM-001 | allowed_narrow_floor | main_text | claim only route-guarded fixed-sampling broad12/broad21 spot_size under broad_process_v1; Phase 138-142 external diagnostics do not expand the main claim |
| P143-CLAIM-002 | diagnostic_only | appendix_or_limitations | Phase 138-142 external branches are closed or blocked diagnostics: matbench_glass, matbench_mp_is_metal_large_source, matbench_expt_is_metal |
| P143-CLAIM-003 | blocked | explicit_exclusions | do not claim complete GNN-PINN, general process-condition modeling, density-invariant robustness, source-path/Green feature success, microstructure GNN success, or Matbench glass/is-metal model success |
| P143-CLAIM-004 | blocked_missing_venue_benchmark | planning | submission readiness still requires target venue and benchmark-paper comparison |
| P143-CLAIM-005 | preserved | quality_gate | Phase 143 cannot strengthen claims beyond the Phase 137/116 floor evidence |
| P143-CLAIM-006 | preserved | quality_gate | Phase 143 preserves the existing floor instead of opening new model training |

## Decisions

| Decision | Route | Outcome | Next action |
| --- | --- | --- | --- |
| P143-DECISION-001 | first_paper_claim_boundary | preserve_narrow_claims | continue first-paper polishing around the narrow floor or run another no-training baseline-first intake |
| P143-DECISION-002 | phase138_142_external_training | blocked | do not train on Phase 138-142 external diagnostics |
| P143-DECISION-003 | a100_sxm4_80gb_request | blocked | continue using A800 40GB for no-training reviews and small reproductions |
