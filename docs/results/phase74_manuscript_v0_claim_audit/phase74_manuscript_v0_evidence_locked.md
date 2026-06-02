# Phase 74 Manuscript v0 Evidence-Locked Draft

## Scope Lock

Main claim: `fixed-sampling broad12/broad21 spot_size under broad_process_v1, seeds 7/1/2`.

This v0 draft is evidence-locked to internal result claims. It does not finalize Introduction, Related Work, or target-venue style claims while the Phase 61 literature gaps remain open.

## Results Draft

# Phase 61 Results Draft

## Writing Gate

Mode: section_draft
Active gate: draft_ready for result claims supported by Phase 60; needs_verification for literature-framed introduction claims.
Evidence status: internal result tables are writing-ready; external literature support is listed as LIT_GAP in the package manifest.

## Fixed-Sampling Spot-Size Transfer

The main performance result is the fixed-sampling `spot_size` transfer setting, where `broad_process_v1` is evaluated on broad12 and broad21 with seeds 7, 1, and 2. [C61-MAIN-001]
On broad12, the route achieved 136.384782 test RMSE, 162.125337 hot q90 RMSE, and 165.282182 gradient q90 RMSE, all below the corresponding best strong baselines. [C61-RESULT-001]
On broad21, the route achieved 146.002303 test RMSE, 164.313888 hot q90 RMSE, and 174.735839 gradient q90 RMSE, again below the corresponding best strong baselines. [C61-RESULT-002]
These results support a bounded main claim: the route-guarded process-conditioned Macro physics-informed neural network (Macro PINN) has a stable `spot_size` branch under the fixed broad12/broad21 sampling and seed protocol. [C61-MAIN-001]

## Stress Tests And Claim Boundaries

The stronger-baseline stress rows preserve the fixed-sampling floor: all six broad12/broad21 metric checks remain in the `pass` state. [C61-STRESS-001]
The auxiliary broad15 process panel is consistent with the fixed-sampling result, with all three reported metrics marked as `pass` at seed 7. [C61-STRESS-002]
The alternate-density broad21 rows are marked as `boundary` for test RMSE, hot q90 RMSE, and gradient q90 RMSE, so the manuscript must not claim density-invariant robustness. [C61-BOUNDARY-001]
The residual upper-bound gate selected `blend:broad_process_v1->mean:alpha=1` and records `block_density_failure_driven_model_expansion`, indicating that the density failure should be written as a route boundary rather than as a new model signal. [C61-BOUNDARY-002]
The combined interpretation is therefore not that the route guard is universally robust, but that the fixed-sampling `spot_size` floor is strong enough for the main result while the density-sensitive broad21 case must be discussed as a boundary. [C61-MAIN-001; C61-BOUNDARY-001; C61-BOUNDARY-002]

## Route-Guard Boundary Axes

The route-guard table keeps laser power, scan speed, and full process in boundary positions, while line positives are explicitly no-process fallback evidence. [C61-ROUTE-001]
This wording keeps the route-guard contribution separate from unsupported claims of universal process-conditioning superiority. [C61-ROUTE-001]

## Appendix And Negative-Evidence Discipline

The appendix table contains fourteen diagnostic or boundary rows, including the Phase 58 density stress and Phase 59 residual upper-bound gate. [C61-APPX-001]
The next-branch table keeps Candidate A paused, Candidate B blocked by the Phase 59 density gate, and Candidate C blocked by registration data. [C61-GATE-001]
The appendix should be used to show that negative branches were pruned by explicit gates rather than omitted after informal inspection. [C61-APPX-001; C61-GATE-001]

## User-Review Items

- Confirm whether this section should remain as a compact Results subsection or be split into Results and Discussion.
- Provide target venue examples before final style alignment.
- Resolve LIT_GAP items before drafting Introduction or Related Work claims.

## Methods Draft

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

## Model-Expansion Boundaries

The following boundaries must remain limitations, appendix evidence, or future work until a later gate reopens them:

- `C74-EXCL-001`: density-invariant robustness -> excluded_from_main_claim
- `C74-EXCL-002`: universal process-conditioning success -> excluded_from_main_claim
- `C74-EXCL-003`: laser_power, scan_speed, or full-process strong-baseline wins -> excluded_from_main_claim
- `C74-EXCL-004`: source-path or Green's-function broad12/broad21 success under current data registration -> excluded_from_main_claim
- `C74-GATE-A`: Candidate A: bounded physical spot-size parameterization -> paused_no_training_signal
- `C74-GATE-B`: Candidate B: validation-auditable route policy -> blocked_no_validation_visible_route_policy_signal
- `C74-GATE-C`: Candidate C: data-aligned heat-kernel or Green's-function features -> blocked_by_registration_data
- `C74-GATE-DENSITY`: density-failure-driven model expansion -> block_density_failure_driven_model_expansion
- `C74-GATE-TRAINABLE`: all current trainable model branches -> no_trainable_model_opened

## Next Writing Action

Use this v0 as the internal manuscript base. The next writing step is to resolve literature and target-venue gaps, then polish sections without changing the evidence boundary.
