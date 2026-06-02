# Phase 94 External Registered-Target Candidate Gate

## Gate Decision

Status: `opened_local_design_gate_no_a100`.
Preferred next candidate: `P94-CAND-PFHUB-PINN`.
Phase 95 local gate allowed: `true`.
A100 training allowed now: `false`.
A100-SXM4-80GB request now: `false`.

Phase 94 only opens local/no-training design work. It does not open A100 training or submission-ready claims.

## Candidate Triage

| Candidate | Source | Status | Allowed next gate | Priority |
| --- | --- | --- | --- | --- |
| P94-CAND-AMBNCH-PAD-REG | NIST AM-Bench / mds2-2716 pad thermography plus XYPT | blocked_until_pad_registration_evidence | data_registration_evidence_update_only | 1 |
| P94-CAND-PFHUB-PINN | PFHub phase-field benchmark problems | open_for_local_design_gate | phase95_local_synthetic_benchmark_design | 2 |
| P94-CAND-EXACA-SIM | ExaCA cellular-automata solidification code | blocked_until_simulation_data_card | simulation_data_card_only | 3 |
| P94-CAND-EXT-THERMAL | external public registered thermal/process dataset | blocked_no_external_data_card | source_manifest_and_data_card_required | 5 |
| P94-CAND-MANUSCRIPT | target venue or accepted benchmark papers | blocked_missing_target_benchmarks | benchmark_review_only | 0 |

## Design Queue

| Queue | Candidate | Task | Compute |
| --- | --- | --- | --- |
| P94-DESIGN-001 | P94-CAND-PFHUB-PINN | write candidate_design.json and local/no-training benchmark protocol | local CPU/GPU smoke only; no A100 training |

## Next Action

enter Phase 95 local/no-training design gate for the highest-priority open candidate
