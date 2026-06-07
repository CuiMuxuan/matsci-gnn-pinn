# Phase 154 Route Coverage and Remaining-Scheme Audit

## Gate
- Status: `phase154_route_coverage_audit_ready_current_routes_verified_future_not_exhausted`
- Currently executable model routes verified: `true`
- All possible future schemes exhausted: `false`
- Future/preconditioned route rows: `3`
- First-paper draft allowed now: `true`
- First-paper submission ready: `false`
- Phase 154 model training allowed: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
The audit answers the user's route-coverage question: all currently opened and executable model/research routes have been verified or closed under the current data and gate conditions, but the future scheme space is not exhausted. Future work requires a new source, new registration, a new dense target/split, or venue/literature input.

## Route Coverage
| route_id | route_family | current_status | verification_scope | evidence_anchor | currently_executable_resolved | future_scheme_space_exhausted | model_training_allowed_now | a100_training_allowed_now | missing_precondition | next_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P154-ROUTE-001 | first_paper_main_floor | verified_positive_narrow_scope | fixed-sampling broad12/broad21 spot_size under broad_process_v1, seeds 7/1/2 | docs/results/phase153_first_paper_contribution_refinement/phase153_first_paper_contribution_refinement_gate.json | true | false | false | false | none for first-paper draft; venue/literature still needed for submission | write/refine first paper around the narrow floor |
| P154-ROUTE-002 | neural_operator_fno_dense_field | closed_diagnostic_no_operator_gap | Phase 149 readiness, Phase 150 dense inventory, Phase 151 fixed-grid baseline review | docs/results/phase152_paper_evidence_neural_operator_route_closure/phase152_neural_operator_route_closure_table.csv | true | false | false | false | new leakage-safe dense target/split that strong baselines do not solve | do not train FNO/neural operators from current dense candidates |
| P154-ROUTE-003 | registered_source_path_green_capl | closed_diagnostic_under_current_registration | source-path, Green proxy, registered Layer Camera, G-code, and path-contact audits | docs/results/phase153_first_paper_contribution_refinement/phase153_claim_phrasing_guard_table.csv | true | false | false | false | stronger camera-to-galvo registration or new source-target target with guarded gap | do not tune the closed scalar proxy/path-contact branch |
| P154-ROUTE-004 | microstructure_gnn_or_image_encoder | closed_diagnostic_alignment_or_stability_limit | real-micro, region, patch embedding, and manuscript claim-boundary packages | docs/results/phase116_paper_evidence_consolidation/phase116_manuscript_claim_status_table.csv | true | false | false | false | physically stronger microstructure/thermal alignment or new benchmark source | do not claim microstructure GNN success from current evidence |
| P154-ROUTE-005 | external_baseline_first_sources | sampled_sources_verified_or_closed_as_diagnostics | Battery, Matbench, MPEA, glass/is-metal/perovskite-style baseline-first intakes | docs/results/phase116_paper_evidence_consolidation/phase116_manuscript_claim_status_table.csv | true | false | false | false | fresh public source with leakage-safe splits and strong-baseline-visible gap | new baseline-first source intake is allowed only as a new gated branch |
| P154-ROUTE-006 | first_paper_submission | draft_allowed_submission_not_ready | Phase 153 contribution package plus open venue/literature gaps | docs/results/phase153_first_paper_contribution_refinement/phase153_first_paper_contribution_refinement_gate.json | false | false | false | false | target venue/author guide and benchmark literature verification | resolve writing evidence gaps before submission polish |
| P154-ROUTE-007 | large_gpu_training_or_80gb | blocked_no_measured_need | all active gates keep A100/80GB locks false | docs/results/phase153_first_paper_contribution_refinement/phase153_first_paper_contribution_refinement_gate.json | true | false | false | false | seed-positive branch with measured 40GB memory/runtime bottleneck | do not request A100-SXM4-80GB |

## Remaining Schemes
| remaining_id | scheme_or_need | status | why_not_done_now | required_precondition | can_start_without_new_input | recommended_next_gate |
| --- | --- | --- | --- | --- | --- | --- |
| P154-REMAIN-001 | target venue / author guide / benchmark papers | open_non_model_blocker | This requires user-selected venue or verified benchmark-paper set. | target venue, author guide, or 3-10 accepted benchmark papers | false | literature/venue verification and manuscript alignment package |
| P154-REMAIN-002 | fresh leakage-safe dense operator target | future_preconditioned_route | Current fixed-grid dense candidates either lack leakage-safe splits or are solved by strong baselines. | new dense target/split source with a strong-baseline-visible gap | false | no-training dense target intake and baseline-gap audit |
| P154-REMAIN-003 | stronger scan-path/camera registration | future_preconditioned_route | Current source-path/CAPL/path-contact diagnostics did not clear route guards. | defensible camera-pixel to galvo-mm registration or new registered target | false | registered target/source intake before any mechanism |
| P154-REMAIN-004 | fresh baseline-first source intake | allowed_if_opened_as_new_branch | Existing source branches are closed or diagnostic under current gates. | small public source or server-local source with manifest, leakage-safe split, and strong baselines | true | baseline-first source intake with all training locks false |
| P154-REMAIN-005 | large GPU / A100-SXM4-80GB training | blocked | No seed-positive branch has a measured 40GB bottleneck. | passed seed-positive gate plus measured A800/A100-40GB memory/runtime failure | false | none until a positive branch proves compute need |

## Decisions
| decision_id | question | answer | evidence_anchor | project_action |
| --- | --- | --- | --- | --- |
| P154-DECISION-001 | Are all currently executable model/research branches verified? | yes_under_current_data_and_gate_conditions | docs/results/phase154_route_coverage_and_remaining_scheme_audit/phase154_route_coverage_table.csv | do not reopen closed branches without a new baseline-first gate |
| P154-DECISION-002 | Are all possible future schemes exhausted? | no_future_preconditioned_routes_remain | docs/results/phase154_route_coverage_and_remaining_scheme_audit/phase154_remaining_scheme_table.csv | treat future schemes as requiring new data, registration, venue evidence, or fresh source intake |
| P154-DECISION-003 | Can the first paper proceed? | draft_yes_submission_not_ready | docs/results/phase154_route_coverage_and_remaining_scheme_audit/phase154_route_coverage_gate.json | continue writing around the narrow floor and resolve venue/literature blockers |
| P154-DECISION-004 | Should A100-SXM4-80GB be requested now? | no | docs/results/phase154_route_coverage_and_remaining_scheme_audit/phase154_route_coverage_table.csv | keep A800/A100-40GB as sufficient until a measured positive-branch bottleneck exists |
