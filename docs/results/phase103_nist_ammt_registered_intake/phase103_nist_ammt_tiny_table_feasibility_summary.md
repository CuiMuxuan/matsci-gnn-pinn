# Phase 103 Tiny Registered-Table Feasibility Gate

- Status: `tiny_registered_table_construction_allowed_training_locked`
- Tiny-table construction allowed: `True`
- Phase 104 baseline smoke allowed: `false`
- A100 training allowed now: `false`
- Next action: manually construct a tiny registered source/path-to-target sample table and split manifest

| Role | Required | Scout hits | Sample rows | Text rows | Status |
|---|---:|---:|---:|---:|---|
| coordinate_transform | True | 2 | 2 | 1 | ready_for_manual_join_review |
| trigger_timing | True | 0 | 0 | 0 | ready_for_manual_join_review_layer_join |
| source_command_path | True | 201 | 8 | 8 | ready_for_manual_join_review |
| target_observation | True | 201 | 8 | 0 | ready_for_manual_join_review_binary_schema |
| split_key | False | 402 | 16 | 8 | ready_for_manual_join_review |

This package is a no-training feasibility gate. It does not create the tiny registered table, leakage-safe split manifest, baselines, model mechanisms, or any A100 training claim.
