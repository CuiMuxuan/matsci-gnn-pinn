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
