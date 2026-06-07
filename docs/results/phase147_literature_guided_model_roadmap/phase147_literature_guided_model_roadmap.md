# Phase 147 Literature-Guided Model Roadmap

- Status: `phase147_literature_guided_model_roadmap_ready_phase148_no_training_design`
- Recommended Phase 148 route: `capl_path_contact_graph_audit`
- Phase 148 no-training design allowed: `True`
- Model mechanism allowed now: `false`
- Model training allowed now: `false`
- A100 training allowed now: `false`

## Route Audit

| route_id | recommended_use | next_phase_action | prior_project_evidence |
| --- | --- | --- | --- |
| thermal_pinn | background_and_method_positioning | do not open standalone thermal-PINN training from this route | docs/results/phase146_paper_evidence_refresh/phase146_paper_evidence_refresh_gate.json |
| geometry_agnostic_gnn | architecture_inspiration_only | allow only no-training topology-feature gate before any GNN mechanism | docs/results/phase111_nist_ammt_registered_target_closure_package/phase111_nist_ammt_registered_target_closure_gate.json |
| physics_hardcoded_gnn | phase148_design_target_if_no_training_gate_passes | defer implementation until a no-training topology gate beats controls | docs/results/phase115_nist_ammt_diagnostic_closure_package/phase115_nist_ammt_diagnostic_closure_gate.json |
| capl_path_history | only_if_finer_than_phase114 | open no-training path-contact graph audit only with shuffled/shortcut controls | docs/results/phase114_nist_ammt_gcode_strategy_source_gate/phase114_nist_ammt_gcode_strategy_source_gate.json |
| meltpoolgan | negative_control_and_related_work | do not train on current Phase 112 melt-pool targets | docs/results/phase113_nist_ammt_melt_pool_focused_review/phase113_nist_ammt_melt_pool_focused_review_gate.json |
| neural_operator | future_large_data_candidate | do not request 80GB or train neural operators without a seed-positive dense gate | docs/results/phase146_paper_evidence_refresh/phase146_paper_evidence_refresh_gate.json |
| microstructure_gnn | limitations_and_future_work | do not claim microstructure GNN success in the first paper | docs/results/phase146_paper_evidence_refresh/phase146_paper_evidence_refresh_gate.json |

## Decisions

| decision_id | candidate_route | decision | rationale | phase148_allowed | model_training_allowed | a100_training_allowed_now | a100_80gb_request_now | evidence_anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P147-DECISION-001 | capl_path_contact_graph | allow_no_training_design | External CAPL/MeltpoolGAN routes motivate ordered path-contact/reheat graphs, but Phase 114 already closed simple G-code strategy summaries; the next branch must be finer-grained and baseline-first. | true | false | false | false | docs/results/phase114_nist_ammt_gcode_strategy_source_gate/phase114_nist_ammt_gcode_strategy_source_gate.json |
| P147-DECISION-002 | physics_hardcoded_graph_residual | defer_until_no_training_gap | MAM-PhyGNN-style inductive bias is promising, but mechanism design must wait until a topology/source gate beats strong baselines and shuffled controls. | false | false | false | false | docs/results/phase115_nist_ammt_diagnostic_closure_package/phase115_nist_ammt_diagnostic_closure_gate.json |
| P147-DECISION-003 | neural_operator_or_microstructure_gnn | keep_as_future_work | Neural operators and microstructure GNNs are externally credible, but current project evidence lacks a stable dense/operator target or physically registered microstructure gap. | false | false | false | false | docs/results/phase146_paper_evidence_refresh/phase146_paper_evidence_refresh_gate.json |

## Boundaries

| boundary_id | claim_or_route | status | allowed_wording | blocked_wording | evidence_anchor |
| --- | --- | --- | --- | --- | --- |
| P147-BOUNDARY-001 | first_paper_main_claim | unchanged | route-guarded fixed-sampling broad12/broad21 spot_size under broad_process_v1 | complete GNN-PINN or general process-condition modeling | docs/results/phase146_paper_evidence_refresh/phase146_paper_evidence_refresh_gate.json |
| P147-BOUNDARY-002 | capl_path_contact_graph | no_training_design_only | literature-motivated no-training path-contact graph audit | CAPL/G-code success, source-path/Green success, or graph model success | docs/results/phase114_nist_ammt_gcode_strategy_source_gate/phase114_nist_ammt_gcode_strategy_source_gate.json |
| P147-BOUNDARY-003 | compute | a800_sufficient | continue with small artifact gates and A800 reproduction | request A100-SXM4-80GB before a seed-positive 40GB blocker | docs/results/phase115_nist_ammt_diagnostic_closure_package/phase115_nist_ammt_diagnostic_closure_gate.json |

Next action: implement Phase 148 no-training path-contact graph audit only; compare against Phase 114 G-code strategy, scalar source, shuffled sequence, layer/time, and camera controls
