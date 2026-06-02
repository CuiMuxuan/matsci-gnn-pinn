# AM-Bench Phase 61 Manuscript Draft Package

## Purpose

Phase 61 turns the Phase 60 evidence package into manuscript-facing draft text, captions, and a claim-to-evidence crosswalk. It does not add training evidence.

## Writing Stage Gate

Mode: `section_draft`.
Active gate: `draft_ready` for Results, Methods, captions, and internal claim mapping; `needs_verification` for Introduction and Related Work literature claims.
Evidence status: Phase 60 result tables are writing-ready; literature gaps are listed below.

## Model-Expansion Boundary

Phase 60 gate decision: `block_density_failure_driven_model_expansion`.
Selected upper-bound variant: `blend:broad_process_v1->mean:alpha=1`.

## Package Outputs

| Artifact | Path |
| --- | --- |
| results_draft | docs/results/phase61_manuscript_draft_package/phase61_results_draft.md |
| methods_draft | docs/results/phase61_manuscript_draft_package/phase61_methods_draft.md |
| captions | docs/results/phase61_manuscript_draft_package/phase61_table_figure_captions.md |
| claim_evidence_crosswalk | docs/results/phase61_manuscript_draft_package/phase61_claim_evidence_crosswalk.csv |
| literature_gap_register | docs/results/phase61_manuscript_draft_package/phase61_literature_gap_register.csv |
| package_markdown | docs/results/phase61_manuscript_draft_package/phase61_manuscript_draft_package.md |
| manifest | docs/results/phase61_manuscript_draft_package/phase61_manuscript_draft_package_manifest.json |

## Claim Anchor Overview

| Anchor | Location | Strength | Support | Open risk |
| --- | --- | --- | --- | --- |
| C61-MAIN-001 | Results: Fixed-sampling spot-size transfer | moderate | docs/results/phase60_manuscript_evidence_package/phase60_main_spot_size_seed_positive_table.csv; docs/results/phase60_manuscript_evidence_package/phase60_manuscript_evidence_package_manifest.json | Claim must remain limited to fixed-sampling broad12/broad21 spot_size. |
| C61-RESULT-001 | Results: Fixed-sampling spot-size transfer | moderate | docs/results/phase60_manuscript_evidence_package/phase60_main_spot_size_seed_positive_table.csv; rows dataset=broad12 | Do not generalize beyond broad12 spot_size. |
| C61-RESULT-002 | Results: Fixed-sampling spot-size transfer | moderate | docs/results/phase60_manuscript_evidence_package/phase60_main_spot_size_seed_positive_table.csv; rows dataset=broad21 | Do not generalize beyond broad21 spot_size. |
| C61-STRESS-001 | Results: Stress tests | moderate | docs/results/phase60_manuscript_evidence_package/phase60_stress_boundary_table.csv; scenario=stronger_baseline_stress; rows=6 | Only applies to the implemented reproducible stress baseline family. |
| C61-STRESS-002 | Results: Stress tests | cautious | docs/results/phase60_manuscript_evidence_package/phase60_stress_boundary_table.csv; scenario=auxiliary_process_panel; rows=3 | Single seed auxiliary panel; do not treat as seed-robust external validation. |
| C61-BOUNDARY-001 | Results: Boundary tests | strong | docs/results/phase60_manuscript_evidence_package/phase60_stress_boundary_table.csv; scenario=alternate_density_stress; boundary rows=3 | Boundary must not be softened into robustness. |
| C61-BOUNDARY-002 | Results: Boundary tests | strong | docs/results/phase60_manuscript_evidence_package/phase60_stress_boundary_table.csv; scenario=residual_upper_bound_gate; docs/results/phase60_manuscript_evidence_package/phase60_manuscript_evidence_package_manifest.json | Only blocks density-failure-driven expansion, not all future model work. |
| C61-ROUTE-001 | Results: Route-guard boundaries | moderate | docs/results/phase60_manuscript_evidence_package/phase60_route_guard_boundary_table.csv; route_guard_only=6; no_process_fallback=2 | Must not claim strong-baseline wins for laser_power, scan_speed, or full process. |
| C61-APPX-001 | Appendix: Negative diagnostics | moderate | docs/results/phase60_manuscript_evidence_package/phase60_appendix_negative_diagnostic_table.csv; rows=14 | Appendix wording should not bury limitations that affect main claim scope. |
| C61-GATE-001 | Discussion: Next-branch gate | moderate | docs/results/phase60_manuscript_evidence_package/phase60_next_branch_gate_table.csv; rows=4 | A future validation-visible signal can reopen a candidate branch. |
| C61-METHOD-001 | Methods: Claim governance | moderate | docs/results/phase60_manuscript_evidence_package/phase60_manuscript_evidence_package_manifest.json | Methods should not imply all historical branches used the Phase 60 gate before it existed. |

## Literature And Target-Style Gaps

| Gap | Location | Claim needing support | Evidence needed | Blocks current draft |
| --- | --- | --- | --- | --- |
| LIT_GAP-61-001 | Introduction or Dataset paragraph | AM-Bench is a public additive-manufacturing benchmark suitable for thermal/process generalization studies. | verified dataset citation or official NIST AM-Bench source | no |
| LIT_GAP-61-002 | Related work | Physics-informed neural networks and process-conditioned neural models have known tradeoffs under sparse or heterogeneous thermal data. | verified literature review and representative primary papers | no |
| LIT_GAP-61-003 | Target-venue adaptation | Final section order, citation density, and caption style match the target journal or conference. | target venue author guide or 3-10 benchmark manuscripts | no |

## Next Step

Use the draft sections as the Phase 61 manuscript base. Before Introduction, Related Work, or final polishing, resolve the listed literature and target-style gaps.
