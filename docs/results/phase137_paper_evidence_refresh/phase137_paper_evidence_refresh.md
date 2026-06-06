# Phase 137 Paper Evidence Refresh

## Gate Decision

Status: `phase137_paper_evidence_refresh_ready_first_paper_narrow_claims`.
First-paper draft allowed now: `true`.
Submission ready: `false`.
Model training allowed: `false`.
A100 training allowed now: `false`.
A100-SXM4-80GB request now: `false`.

Phase 137 refreshes claim boundaries from existing small artifacts only. It does not read raw data, run baselines, or train a model.

## External Diagnostics

| Branch | Terminal status | Blockers | Boundary |
| --- | --- | --- | --- |
| battery_failure_databank | phase119_battery_failure_candidate_sweep_closed_all_phase117_candidates | close Battery Failure Databank as diagnostic or open a new external baseline-first intake | do not use battery failure targets for model claims |
| matbench_steels | phase122_matbench_steels_low_capacity_mechanism_closed_no_guarded_gain | phase121_validation_guard_gain;phase121_test_reversal_guard | interpretable low-capacity alloy mechanism did not beat the focused-review guard |
| matbench_expt_gap | phase125_matbench_expt_gap_low_capacity_mechanism_closed_no_guarded_gain | phase124_validation_guard_gain;phase124_test_reversal_guard | low-capacity band-gap mechanism did not beat the focused-review guard |
| matbench_phonons | phase127_matbench_phonons_focused_review_closed_split_sensitivity_or_shortcut | original_split_shortcut_dominance;shortcut_dominant_split_count;target_distribution_imbalanced_split_count | shortcut and target-distribution audits block model claims |
| matbench_dielectric | phase128_matbench_dielectric_n_closed_no_stable_guarded_gap | composition, family, or dominant-element shortcut dominates the selected safe profile | baseline-first gate closed by no stable guarded gap |
| matbench_log_gvrh | phase134_matbench_log_gvrh_focused_review_closed_split_sensitivity_or_shortcut | target_distribution_imbalanced_split_count | target-distribution imbalance blocks mechanism or training |
| matbench_log_kvrh | phase131_matbench_log_kvrh_focused_review_closed_split_sensitivity_or_shortcut | target_distribution_imbalanced_split_count | target-distribution imbalance blocks mechanism or training |
| matbench_jdft2d | phase133_matbench_jdft2d_focused_review_closed_split_sensitivity_or_shortcut | split_sensitivity_pass_rate;shortcut_dominant_split_count;target_distribution_imbalanced_split_count | split sensitivity, shortcut dominance, and target imbalance block model claims |
| matbench_perovskites | phase136_matbench_perovskites_focused_review_closed_split_sensitivity_or_shortcut | target_distribution_imbalanced_split_count | target-distribution imbalance blocks mechanism or training |

## Claim Boundaries

| Claim | Status | Use | Wording guard |
| --- | --- | --- | --- |
| P137-CLAIM-001 | allowed_narrow_floor | main_text | claim only route-guarded fixed-sampling broad12/broad21 spot_size under broad_process_v1; do not generalize to full GNN-PINN or universal process conditioning |
| P137-CLAIM-002 | diagnostic_only | appendix_or_limitations | external branches are closed diagnostics: battery_failure_databank, matbench_steels, matbench_expt_gap, matbench_phonons, matbench_dielectric, matbench_log_gvrh, matbench_log_kvrh, matbench_jdft2d, matbench_perovskites |
| P137-CLAIM-003 | blocked | explicit_exclusions | do not claim complete GNN-PINN, general process-condition modeling, density-invariant robustness, successful source-path/Green features, or successful microstructure GNN |
| P137-CLAIM-004 | blocked_missing_venue_benchmark | planning | submission readiness still requires target venue and benchmark-paper comparison |
| P137-CLAIM-005 | preserved | quality_gate | Phase 137 cannot strengthen claims beyond Phase 116 floor evidence |

## Next Decisions

| Decision | Route | Result | Next action |
| --- | --- | --- | --- |
| P137-DECISION-001 | first_paper_draft | allowed_with_narrow_claims | draft or polish first paper around the narrow route-guarded floor; resolve venue/benchmark blockers before submission |
| P137-DECISION-002 | external_diagnostic_training | blocked | do not train on closed external diagnostics |
| P137-DECISION-003 | new_baseline_first_source | allowed_no_training_intake_only | open a new baseline-first source only after the active evidence package is closed |
| P137-DECISION-004 | a100_sxm4_80gb_request | blocked | continue using A800 40GB for no-training reviews and small reproductions |
