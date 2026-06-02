# AM-Bench Phase 58 Clean Repro and Stress-Test Plan

## Purpose

Phase 58 starts the post-governance validation path. The goal is not to add a
new model first, but to prove that the current Phase 55/56/57 paper-facing
package can be rebuilt from a clean GitHub checkout, then stress-test the
frozen `spot_size` claim before any new architecture branch is allowed to
seed-expand.

## Clean-Checkout Repro Result

The A100 server rebuilt the Phase 55 seed summary, Phase 56 manuscript package,
and Phase 57 claim governance package from a fresh checkout at commit `35f11b1`.
The clean checkout used the main server repository only as a read-only source of
large `outputs/` artifacts, because training outputs are intentionally not
tracked by GitHub.

Repro command:

```bash
cd /root/matsci-gnn-pinn
bash scripts/server/run_phase58_clean_repro_package_a100.sh
```

Repro manifest:

```text
docs/results/phase58_clean_repro/phase58_clean_repro_manifest.json
```

Key verification:

| Check | Result |
|---|---|
| Clean checkout commit | `35f11b1` |
| Phase 55 transfer gate | `seed_robust_transfer_positive` |
| Aggregate transfer positive | `true` |
| Paired seed transfer positive | `true` |
| Claim ledger `paper_positive_seed_robust` rows | `2` |
| Claim ledger `route_guard_only` rows | `6` |
| Claim ledger `route_guard_no_process_positive` rows | `2` |
| Claim ledger `diagnostic_negative` rows | `11` |
| Claim ledger `blocked_by_data` rows | `1` |
| A100 status after rebuild | idle, `14 MiB / 40960 MiB`, `0%` |

This confirms that the current main claim package is reproducible from GitHub
code plus server-side artifacts. It does not by itself prove robustness against
stronger baselines or alternate data panels.

## Stress-Test Matrix

The next Phase 58 work should run only stress tests that can challenge the
frozen `spot_size` claim without changing the claim wording midstream.

| Track | Test | Go/no-go rule |
|---|---|---|
| Baseline strength | Add low-risk scikit-learn baselines already available in the server env, such as histogram gradient boosting or random forest if dependency checks pass. | Keep the main claim only if `broad_process_v1` remains better on RMSE, hot q90 RMSE, and gradient q90 RMSE for broad12 and broad21. |
| Sampling density | Rebuild broad12/broad21 `spot_size` summaries using a denser but fixed process-balanced sample if existing artifacts permit it. | Preserve the Phase 57 frozen-floor relationship or mark as density-sensitive. |
| Independent process panel | Build one additional process-balanced subset that is not exactly broad12 or broad21 if data coverage permits. | Treat as external robustness, not replacement evidence. |
| Report drift | Regenerate Phase 55/56/57 package after every stress-test change. | Any claim package drift must be explained by changed inputs, not script nondeterminism. |

## Current Decision

Proceed to Phase 58 stress testing before implementing a new model branch. A new
architecture may be attempted only after the stress-test result identifies a
specific, train/validation-visible gap that is not already covered by
`broad_process_v1`.
