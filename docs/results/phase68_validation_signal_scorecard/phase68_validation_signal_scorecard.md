# Phase 68 Validation-Visible Signal Mining Scorecard

## Purpose

Phase 68 translates the Phase 59-61 evidence boundary into explicit next-step decisions for model innovation. It does not add training evidence.

## Current Decision

Trainable model branches opened by current evidence: `0`.
No Candidate A/B/C training should start from the Phase 58/59 density failure alone. New model work must first pass a train/validation-visible signal probe.

## Candidate Scorecard

| ID | Candidate | Status | Decision | First action | 80GB trigger |
| --- | --- | --- | --- | --- | --- |
| A | bounded physical spot-size parameterization | paused_no_training_signal | do_not_train_from_density_failure | build a no-training signal probe for spot_size physics features from train/validation artifacts before changing the model | request A100-SXM4-80GB only if the reopened model requires large multi-panel or multi-seed training that exceeds measured 40GB memory |
| B | validation-auditable route policy | blocked_by_phase59_validation_gate | policy_audit_only | run a non-trainable validation-only policy audit among existing comparable routes before any trainable policy | request A100-SXM4-80GB only for a later high-capacity policy after low-capacity policy evidence passes on 40GB |
| C | data-aligned heat-kernel or Green's-function features | blocked_by_registration_data | data_audit_before_training | inventory or add aligned scan-path/pad-thermography data and pass coordinate/unit/coverage checks before model changes | request A100-SXM4-80GB only if aligned dense pad or multi-target feature training exceeds measured 40GB memory |
| D | larger model architecture branch: Bayesian PINN, attention, GCN/CNN, or meta-learning | deferred_requires_local_identifiability_gate | synthetic_or_local_gate_first | define a small identifiability or region-preserving local gate for the architecture and compare against deterministic controls | request A100-SXM4-80GB before launching learned image encoders, large GCN/CNN backbones, large ensembles, or multi-dataset training expected to exceed 40GB |
| E | external robustness or second dataset branch | open_for_data_planning_only | prepare_data_card_and_split_manifest | prepare a data card, download/verification route, split manifest, baseline table, and local feasibility gate | request A100-SXM4-80GB if the chosen external branch needs dense multi-process or image-backbone training beyond measured 40GB memory |

## Action Queue

| Priority | Action | Type | Description | A100 | 80GB | Exit gate |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | P68-AUDIT | package_closeout | Commit and sync the Phase 68 scorecard after local/server verification. | no | no | local/GitHub/server same commit and manifest counts match |
| 2 | P68-SPOT-SIGNAL | non_training_signal_probe | Mine train/validation artifacts for a bounded spot_size physics signal before Candidate A training. | no | no | signal appears on both broad12 and broad21 without relying on test labels |
| 3 | P68-ROUTE-POLICY | non_training_policy_audit | Audit whether validation-only route choice among existing comparable routes can improve boundary axes without weakening spot_size. | no | no | validation policy preserves broad12/broad21 spot_size and improves at least one boundary axis |
| 4 | P68-DATA-REGISTRATION | data_audit | Search local/server data for aligned scan-path or pad-thermography targets before reopening heat-kernel or Green's-function features. | no | no | coordinate units, coverage, and target/source registration are compatible |
| 5 | P68-80GB-TRIGGER | resource_gate | Request a new A100-SXM4-80GB server only after a planned training branch has measured or clearly projected 40GB memory overflow. | yes | conditional | current A100-SXM4-40GB cannot run the validated branch safely |

## Interpretation

The immediate path is manuscript-first plus non-training signal mining. Candidate A/B/C can be reopened, but only after a validation-visible signal appears on both broad12 and broad21 or after new registered data removes a data blocker. Larger model architectures and new datasets are allowed, but they must start from small identifiability, data-card, split-manifest, and baseline gates. A100-SXM4-80GB should be requested only when a validated branch demonstrably cannot run on the current 40GB server.
