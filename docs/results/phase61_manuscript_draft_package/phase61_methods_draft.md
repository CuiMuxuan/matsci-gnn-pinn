# Phase 61 Methods Draft

## Evidence-Governed Experiment Selection

The manuscript should describe experiments as a staged evidence-governed workflow rather than as an unconstrained architecture sweep. [C61-METHOD-001]
Route selection, seed expansion, and future branch promotion are described under a frozen-floor rule that requires preserving the broad12/broad21 `spot_size` floor before seed expansion. [C61-METHOD-001]
The fixed floor for later branches is the `broad_process_v1` route on broad12 and broad21 `spot_size`, evaluated with seeds 7, 1, and 2. [C61-MAIN-001]

## Route-Guarded Macro PINN Evaluation

The route guard separates process-conditioned and no-process fallback behavior by split axis, so axes that do not pass the strong-baseline gate are not promoted as process-conditioning wins. [C61-ROUTE-001]
The main promoted branch is the `spot_size` route, while laser power, scan speed, and full process remain route-boundary cases. [C61-ROUTE-001]

## Metric Protocol

All main comparisons should report test root mean squared error (RMSE), hot q90 RMSE, and gradient q90 RMSE. [C61-RESULT-001; C61-RESULT-002]
The hot q90 and gradient q90 metrics are retained in the main text because Phase 35-59 diagnostics repeatedly showed that global RMSE alone can hide region-specific tradeoffs. [C61-APPX-001]

## Stress And Residual-Gate Protocol

Stress testing uses stronger tabular baselines, an auxiliary process-balanced panel, and alternate sampling density as separate checks instead of merging them into the main fixed-sampling claim. [C61-STRESS-001; C61-STRESS-002; C61-BOUNDARY-001]
The residual upper-bound probe is reported as a no-test-leakage gate: it uses train/validation evidence to decide whether the density failure supports a new model branch and reports that the winning validation-visible correction is fallback to the mean reference. [C61-BOUNDARY-002]

## Reproducibility Artifacts

The manuscript methods should point to the generated package tables, scripts, and manifests rather than only to prose descriptions. [C61-METHOD-001]
The Phase 60 and Phase 61 packages provide the source tables, draft text, captions, and claim anchors needed to audit each manuscript claim. [C61-METHOD-001]

## User-Review Items

- Confirm whether training hyperparameters should be summarized in this manuscript section or delegated to supplementary material.
- Add external dataset and PINN literature citations before finalizing any broad field-motivation claims.
