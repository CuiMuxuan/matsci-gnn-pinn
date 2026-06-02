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

## Stronger-Baseline Stress Result

The A100 server ran the stronger-baseline stress at commit `9d177ae` after
fixing the artifact-index parser to skip trailing blank TSV rows. The runner
generated random forest and histogram gradient boosting baselines for broad12
and broad21 `spot_size`, each with coordinate-only and coordinate+process
features.

Result package:

```text
docs/results/phase58_stronger_baseline_stress/
```

Stress gate:

```text
claim_survives_stronger_baselines
```

Key comparison:

| Dataset | Metric | Frozen `broad_process_v1` | Best baseline after stress | Pass |
|---|---|---:|---:|---|
| broad12 | RMSE | 136.384782 | 151.850578 | yes |
| broad12 | hot q90 RMSE | 162.125337 | 252.554440 | yes |
| broad12 | gradient q90 RMSE | 165.282182 | 233.119660 | yes |
| broad21 | RMSE | 146.002303 | 149.185412 | yes |
| broad21 | hot q90 RMSE | 164.313888 | 251.976794 | yes |
| broad21 | gradient q90 RMSE | 174.735839 | 231.072566 | yes |

No stress baselines were missing. The strongest baseline after adding random
forest and histogram gradient boosting remains the prior mean baseline on all
six required dataset/metric checks.

## Sampling and Panel Stress Runner

`scripts/server/run_phase58_sampling_panel_stress_a100.sh` is the next Phase 58
runner. It uses isolated profile tags (`phase58_density_profile` and
`phase58_panel_profile`) so alternate sampling or auxiliary-panel results cannot
overwrite the frozen Phase 55 `broad_process_profile` artifacts.

The density branch reruns broad12/broad21 `spot_size` with a denser fixed sample.
The panel branch runs an auxiliary broad15 process-balanced `spot_size` panel.
These results are stress evidence only; they do not replace the Phase 55 frozen
floor unless they pass the same full seed and baseline gates in a later phase.

## Sampling and Panel Stress Result

Result package:

```text
docs/results/phase58_sampling_panel_stress/
```

The alternate-density branch is mixed. The broad12 `spot_size` branch remains
positive against the strongest baseline and no-process Macro PINN, but broad21
does not beat the mean baseline under the denser sample. This means the frozen
Phase 55 claim survives as a fixed-sampling, seed-robust result, but it should
not be described as density-invariant.

| Dataset | Metric | `broad_process_v1` | Best strong baseline | No-process | Gate |
|---|---|---:|---:|---:|---|
| broad12 density | RMSE | 139.085217 | 140.201362 | 260.807865 | pass |
| broad12 density | hot q90 RMSE | 235.696768 | 253.374499 | 478.337515 | pass |
| broad12 density | gradient q90 RMSE | 221.604222 | 234.028899 | 434.594412 | pass |
| broad21 density | RMSE | 153.259455 | 139.725646 | 226.518789 | fail vs strong |
| broad21 density | hot q90 RMSE | 270.628922 | 253.129723 | 421.639304 | fail vs strong |
| broad21 density | gradient q90 RMSE | 250.519935 | 231.780894 | 374.684937 | fail vs strong |

The auxiliary broad15 process-balanced panel is positive at seed 7:

| Dataset | Metric | `broad_process_v1` | Best strong baseline | No-process | Gate |
|---|---|---:|---:|---:|---|
| broad15 panel | RMSE | 138.855456 | 151.850578 | 206.100512 | pass |
| broad15 panel | hot q90 RMSE | 158.622677 | 252.554440 | 363.828210 | pass |
| broad15 panel | gradient q90 RMSE | 165.869192 | 233.732337 | 330.201687 | pass |

The next model-development step should therefore start from Phase 59 residual
anatomy of the broad21 density failure. A new model branch is justified only if
the failure is visible from train/validation residual structure and repeats
under the frozen broad12/broad21 evidence contract.

## Current Decision

Close Phase 58 as a bounded stress-test phase. The stronger-baseline stress and
auxiliary broad15 panel support keeping the current `spot_size` paper-facing
floor intact. The alternate-density broad21 failure must be treated as a claim
boundary and the first Phase 59 diagnostic target, not hidden or folded into a
stronger manuscript claim.
