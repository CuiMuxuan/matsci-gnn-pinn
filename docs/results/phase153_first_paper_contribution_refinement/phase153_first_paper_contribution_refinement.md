# Phase 153 First-Paper Contribution Refinement

## Gate
- Status: `phase153_first_paper_contribution_refinement_ready_narrow_claims`
- Contribution refinement ready: `true`
- First-paper draft allowed now: `true`
- First-paper submission ready: `false`
- New model claim ready: `false`
- Phase 153 model training allowed: `false`
- Operator training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Writing Boundary
The first paper should be framed as a narrow, evidence-controlled contribution: route-guarded process-conditioned Macro PINN evidence for fixed-sampling broad12/broad21 spot_size, plus a reproducible claim-governance protocol. Closed branches remain diagnostic appendices.

## Contribution Table
| contribution_id | contribution_title | writeable_claim | evidence_anchor | main_text_use | scope_guard | novelty_boundary | risk_if_overwritten |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P153-CONTRIB-001 | Route-guarded process-conditioned Macro PINN floor | A conservative route-guarded process-conditioned Macro PINN improves fixed-sampling broad12/broad21 spot_size transfer over strong baselines across three seeds. | docs/results/phase60_manuscript_evidence_package/phase60_main_spot_size_seed_positive_table.csv | primary experimental contribution | fixed-sampling broad12/broad21 spot_size only; seeds 7/1/2 | broad12 (Test RMSE: 136.384781987 vs strong 151.850577987; Hot q90 RMSE: 162.125337065 vs strong 252.554439588; Gradient q90 RMSE: 165.282182272 vs strong 233.119660009) | broad21 (Test RMSE: 146.002302637 vs strong 149.185412106; Hot q90 RMSE: 164.31388799 vs strong 251.976794482; Gradient q90 RMSE: 174.735838739 vs strong 231.072566056) | Overclaiming as universal process modeling would contradict route-guard diagnostics. |
| P153-CONTRIB-002 | Validation-only route governance and strong-baseline discipline | The study contributes an auditable route-governance protocol that uses validation-only route selection, strong non-neural baselines, seed checks, and explicit diagnostic closure before model promotion. | docs/results/phase88_fallback_manuscript_finalization/phase88_claim_lock_table.csv | methods and evaluation contribution | protocol contribution; not a claim that every route improves performance | Use as reproducibility/governance framing tied to recorded artifacts. | Removing the guard makes negative branches look like failed tuning rather than controlled evidence. |
| P153-CONTRIB-003 | Boundary-aware process-axis evidence map | The paper separates the spot_size positive branch from route-guard-only or fallback behavior on line, laser_power, scan_speed, and full process splits. | docs/results/phase60_manuscript_evidence_package/phase60_route_guard_boundary_table.csv | results boundary and limitations | do not claim all process axes beat strong baselines | Boundary map clarifies where process conditioning is useful versus guarded. | Axis-sensitive evidence would be flattened into an unsupported generalization claim. |
| P153-CONTRIB-004 | Closed-branch diagnostic ledger including neural operators | The manuscript can report a diagnostic ledger showing why NIST AMMT, CAPL/path-contact, microstructure, MPEA, and neural-operator routes were not promoted to main-text model claims. | docs/results/phase152_paper_evidence_neural_operator_route_closure/phase152_claim_boundary_refresh_table.csv | appendix and limitations contribution | diagnostic closure only; no success claim for closed branches | Contribution is transparent claim governance, not a new neural-operator architecture. | Closed diagnostics could be misrepresented as model innovations. |
| P153-CONTRIB-005 | Paper-ready evidence contract | The first paper has a machine-readable evidence contract connecting main claims, claim guards, artifact paths, and remaining submission blockers. | docs/results/phase116_paper_evidence_consolidation/phase116_positive_floor_table.csv | reproducibility and artifact availability framing | submission readiness still requires venue/literature alignment | Engineering reproducibility support; not an added experimental result. | Paper could drift from evidence-backed claims into unsupported framing. |

## Section Map
| section_id | section_name | section_purpose | primary_claim | evidence_anchor | must_include | must_not_include |
| --- | --- | --- | --- | --- | --- | --- |
| P153-SEC-001 | Introduction | Frame the problem as evidence-controlled process-conditioned thermal modeling. | The work targets reliable route-guarded process transfer, not universal AM digital twins. | docs/results/phase60_manuscript_evidence_package/phase60_main_spot_size_seed_positive_table.csv | fixed-sampling broad12/broad21 spot_size scope and strong-baseline requirement | complete GNN-PINN, universal process modeling, FNO success |
| P153-SEC-002 | Method | Describe Macro PINN, process-route selection, and validation-only governance. | broad_process_v1 is a conservative route guard with explicit fallback behavior. | docs/results/phase60_manuscript_evidence_package/phase60_route_guard_boundary_table.csv | route table, selection policy, train/validation/test separation | trainable mixture-of-experts success or unverified GNN/microstructure mechanism |
| P153-SEC-003 | Results | Lead with the seed-robust spot_size floor and separate boundary axes. | spot_size is the only current process-conditioned strong-baseline positive main result. | docs/results/phase60_manuscript_evidence_package/phase60_main_spot_size_seed_positive_table.csv | three metrics for broad12 and broad21; seeds 7/1/2 | density-invariant robustness or all-axis superiority |
| P153-SEC-004 | Limitations and Appendix | Record negative diagnostics and explain why newer routes were not promoted. | Closed branches are useful boundary evidence, not model success. | docs/results/phase152_paper_evidence_neural_operator_route_closure/phase152_claim_boundary_refresh_table.csv | neural-operator closure, CAPL/path-contact closure, MPEA and microstructure diagnostics | operator training, dense-field operator success, or 80GB necessity |
| P153-SEC-005 | Conclusion | Restate the narrow positive result and the route-governance contribution. | The contribution is a guarded, reproducible first step with explicit boundaries. | docs/results/phase88_fallback_manuscript_finalization/phase88_claim_lock_table.csv | venue/literature gaps remain before submission-ready claims | future-work diagnostics as completed contributions |

## Phrasing Guards
| guard_id | unsafe_or_overbroad_phrase | paper_safe_replacement | reason | evidence_anchor |
| --- | --- | --- | --- | --- |
| P153-PHRASE-001 | complete GNN-PINN framework | route-guarded process-conditioned Macro PINN prototype | The stable evidence is a Macro PINN route floor, not a complete GNN-PINN system. | docs/results/phase60_manuscript_evidence_package/phase60_main_spot_size_seed_positive_table.csv |
| P153-PHRASE-002 | general process-condition modeling | fixed-sampling spot_size transfer under broad_process_v1 | Only spot_size is main-text strong-baseline positive; other axes are boundary evidence. | docs/results/phase60_manuscript_evidence_package/phase60_main_spot_size_seed_positive_table.csv |
| P153-PHRASE-003 | density-invariant robustness | fixed-sampling seed-robust transfer with density stress reported as a limitation | Alternate-density stress remains a boundary, not a positive robustness result. | docs/results/phase88_fallback_manuscript_finalization/phase88_claim_lock_table.csv |
| P153-PHRASE-004 | neural-operator/FNO success | neural-operator route closed as diagnostic after fixed-grid baseline review | Phase 149-151 found no strong-baseline-visible operator modeling gap. | docs/results/phase152_paper_evidence_neural_operator_route_closure/phase152_claim_boundary_refresh_table.csv |
| P153-PHRASE-005 | source-path/Green/CAPL/path-contact success | source-path and path-contact routes are appendix diagnostics under current guards | Path/contact and source-kernel routes did not clear their baseline/registration gates. | docs/results/phase152_paper_evidence_neural_operator_route_closure/phase152_claim_boundary_refresh_table.csv |
| P153-PHRASE-006 | microstructure GNN success | microstructure routes remain diagnostic due unstable alignment/performance | Existing microstructure evidence is not stable enough for a model contribution claim. | docs/results/phase116_paper_evidence_consolidation/phase116_manuscript_claim_status_table.csv |
| P153-PHRASE-007 | A100-SXM4-80GB is required | A800/A100-40GB remains sufficient for the current gates; 80GB is unproven | No seed-positive branch has hit a measured 40GB memory/runtime bottleneck. | docs/results/phase152_paper_evidence_neural_operator_route_closure/phase152_claim_boundary_refresh_table.csv |

## Open Gaps
| gap_id | gap_type | status | why_it_matters | required_resolution | blocks_submission |
| --- | --- | --- | --- | --- | --- |
| P153-GAP-001 | target_venue | open | Final section ordering, citation density, and submission language depend on venue. | User supplies target venue/author guide or accepted benchmark papers. | true |
| P153-GAP-002 | literature_verification | open | Novelty statements need verified benchmark papers and citation proximity. | Run literature verification before final Introduction/Related Work claims. | true |
| P153-GAP-003 | neural_operator_branch | closed_diagnostic | Prevents FNO/operator claims from entering the manuscript. | continue first-paper refinement around the route-guarded spot_size floor; do not write neural-operator/FNO success; open only a fresh no-training source intake if continuing model discovery | false |
| P153-GAP-004 | additional_model_training | closed_for_first_paper | New training would blur the frozen first-paper floor unless a fresh gate passes. | Open only a fresh no-training baseline-first intake before any new training. | false |
