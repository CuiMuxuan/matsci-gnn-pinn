# Phase 146 Paper Evidence Refresh

- Status: `phase146_paper_evidence_refresh_ready_first_paper_narrow_claims`
- First paper draft allowed now: `True`
- New external model claim ready: `False`
- Model training allowed: `False`
- A100 80GB request now: `False`

## Latest External Diagnostics

| Branch | Status | Final use | Boundary |
| --- | --- | --- | --- |
| mpea_mechanical | phase145_mpea_mechanical_focused_review_closed_split_sensitivity_or_shortcut | appendix_negative_external_diagnostic | MPEA hardness is blocked by split sensitivity, process/shortcut controls, and target-distribution imbalance |

## Claim Boundary

| Claim | Status | Allowed use | Guard |
| --- | --- | --- | --- |
| P146-CLAIM-001 | allowed_narrow_floor | main_text | claim only route-guarded fixed-sampling broad12/broad21 spot_size under broad_process_v1; Phase 144-145 MPEA diagnostics do not expand the main claim |
| P146-CLAIM-002 | diagnostic_only | appendix_or_limitations | Phase 144-145 branches are closed diagnostics: mpea_mechanical |
| P146-CLAIM-003 | blocked | explicit_exclusions | do not claim complete GNN-PINN, general process-condition modeling, density-invariant robustness, source-path/Green feature success, microstructure GNN success, Matbench glass/is-metal model success, or MPEA hardness model success |
| P146-CLAIM-004 | blocked_missing_venue_benchmark | planning | submission readiness still requires target venue and benchmark-paper comparison |
| P146-CLAIM-005 | preserved | quality_gate | Phase 146 preserves the Phase 143/116 floor evidence and does not strengthen claims |

## Decisions

| Decision | Route | Outcome | Next action |
| --- | --- | --- | --- |
| P146-DECISION-001 | first_paper_claim_boundary | preserve_narrow_claims | continue first-paper polishing around the narrow floor or run another no-training baseline-first intake |
| P146-DECISION-002 | phase144_145_mpea_training | blocked | do not train on Phase 144-145 MPEA diagnostics |
| P146-DECISION-003 | a100_sxm4_80gb_request | blocked | continue using A800 40GB for no-training reviews and small reproductions |
