# Phase 74 Manuscript v0 Claim-Audit Package

## Writing Gate

Mode: `evidence_locked_manuscript_v0`.
Status: `ready_for_internal_manuscript_review`.
Main claim locked: `true`.
Literature gaps open: `3`.
Trainable model opened now: `false`.

## Claim Audit

| Claim | Location | Audit status | Allowed | Strength |
| --- | --- | --- | --- | --- |
| C61-MAIN-001 | Results: Fixed-sampling spot-size transfer | supported_for_v0 | yes | moderate |
| C61-RESULT-001 | Results: Fixed-sampling spot-size transfer | supported_for_v0 | yes | moderate |
| C61-RESULT-002 | Results: Fixed-sampling spot-size transfer | supported_for_v0 | yes | moderate |
| C61-STRESS-001 | Results: Stress tests | supported_for_v0 | yes | moderate |
| C61-STRESS-002 | Results: Stress tests | supported_for_v0 | yes | cautious |
| C61-BOUNDARY-001 | Results: Boundary tests | supported_for_v0 | yes | strong |
| C61-BOUNDARY-002 | Results: Boundary tests | supported_for_v0 | yes | strong |
| C61-ROUTE-001 | Results: Route-guard boundaries | supported_for_v0 | yes | moderate |
| C61-APPX-001 | Appendix: Negative diagnostics | supported_for_v0 | yes | moderate |
| C61-GATE-001 | Discussion: Next-branch gate | supported_for_v0 | yes | moderate |
| C61-METHOD-001 | Methods: Claim governance | supported_for_v0 | yes | moderate |
| C74-LIT-LOCK | Introduction/Related Work | locked_out_of_v0_claims | no | none_for_unverified_literature |
| C74-MODEL-GATE | Discussion/Future Work | supported_for_v0 | yes | boundary |

## Table/Figure Inventory

| Artifact | Role | Status | Path |
| --- | --- | --- | --- |
| T1 | main_table | ready | docs/results/phase60_manuscript_evidence_package/phase60_main_spot_size_seed_positive_table.csv |
| T2 | route_guard_table | ready | docs/results/phase60_manuscript_evidence_package/phase60_route_guard_boundary_table.csv |
| T3 | stress_boundary_table | ready | docs/results/phase60_manuscript_evidence_package/phase60_stress_boundary_table.csv |
| T4 | appendix_negative_table | ready | docs/results/phase60_manuscript_evidence_package/phase60_appendix_negative_diagnostic_table.csv |
| T5 | next_branch_gate_table | ready | docs/results/phase60_manuscript_evidence_package/phase60_next_branch_gate_table.csv |
| M1 | results_v0_source | ready | docs/results/phase61_manuscript_draft_package/phase61_results_draft.md |
| M2 | methods_v0_source | ready | docs/results/phase61_manuscript_draft_package/phase61_methods_draft.md |
| M3 | captions | ready | docs/results/phase61_manuscript_draft_package/phase61_table_figure_captions.md |
| A1 | claim_crosswalk | ready | docs/results/phase61_manuscript_draft_package/phase61_claim_evidence_crosswalk.csv |
| A2 | candidate_signal_scorecard | ready | docs/results/phase68_validation_signal_scorecard/phase68_candidate_signal_scorecard.csv |
| A3 | candidate_a_gate | ready | docs/results/phase69_spot_size_signal_probe/phase69_candidate_a_gate.json |
| A4 | candidate_b_gate | ready | docs/results/phase70_route_policy_audit/phase70_candidate_b_gate.json |
| A5 | candidate_c_gate | ready | docs/results/phase71_data_registration_audit/phase71_candidate_c_gate.json |

## Boundary Register

| Boundary | Scope | Status | Main text treatment |
| --- | --- | --- | --- |
| C74-EXCL-001 | density-invariant robustness | excluded_from_main_claim | state as limitation or boundary only |
| C74-EXCL-002 | universal process-conditioning success | excluded_from_main_claim | state as limitation or boundary only |
| C74-EXCL-003 | laser_power, scan_speed, or full-process strong-baseline wins | excluded_from_main_claim | state as limitation or boundary only |
| C74-EXCL-004 | source-path or Green's-function broad12/broad21 success under current data registration | excluded_from_main_claim | state as limitation or boundary only |
| C74-GATE-A | Candidate A: bounded physical spot-size parameterization | paused_no_training_signal | future work only |
| C74-GATE-B | Candidate B: validation-auditable route policy | blocked_no_validation_visible_route_policy_signal | future work only |
| C74-GATE-C | Candidate C: data-aligned heat-kernel or Green's-function features | blocked_by_registration_data | data limitation and future registered-target work |
| C74-GATE-DENSITY | density-failure-driven model expansion | block_density_failure_driven_model_expansion | route boundary, not model signal |
| C74-GATE-TRAINABLE | all current trainable model branches | no_trainable_model_opened | do not imply a newer architecture has passed |
