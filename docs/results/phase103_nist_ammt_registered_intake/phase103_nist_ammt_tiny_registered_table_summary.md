# Phase 103 Tiny Registered Source-Target Table

- Status: `tiny_registered_table_ready_manual_baseline_pending`
- Tiny registered table ready: `True`
- Leakage-safe split manifest ready: `True`
- Row count: `24`
- Split group count: `12`
- Split counts: `train=14, val=4, test=6`
- Phase 104 baseline smoke allowed: `false`
- A100 training allowed now: `false`
- Next action: validate the tiny layer-member table and split manifest before any separate Phase 104 baseline-smoke decision

This package is no-training evidence. It joins source command layer members to layer-camera target members by audited integer layer offsets and writes a split manifest grouped by `source_layer_index`; it does not read raw ZIP payloads, run baselines, train models, or open an A100 training gate.
