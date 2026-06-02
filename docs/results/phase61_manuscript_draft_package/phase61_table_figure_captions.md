# Phase 61 Table And Figure Caption Drafts

## Main Table

**Table 1. Fixed-sampling `spot_size` transfer under the route-guarded Macro PINN.** The table reports three-seed means and standard deviations for broad12 and broad21, comparing `broad_process_v1` with no-process Macro PINN and the best strong baseline on test RMSE, hot q90 RMSE, and gradient q90 RMSE. This table supports the main fixed-sampling `spot_size` claim only. [C61-MAIN-001; C61-RESULT-001; C61-RESULT-002]

## Route-Guard Table

**Table 2. Route-guard boundary classification across process axes.** The table separates the seed-robust `spot_size` process-conditioned branch from no-process fallback and route-guard-only axes, preventing laser power, scan speed, full process, or line fallback results from being written as universal process-conditioning wins. [C61-ROUTE-001]

## Stress And Boundary Table

**Table 3. Stress tests and residual-boundary checks for the fixed `spot_size` floor.** Stronger-baseline stress and the auxiliary broad15 panel support the fixed-sampling floor, while alternate-density broad21 and the Phase 59 residual upper-bound probe define the density-sensitive boundary. [C61-STRESS-001; C61-STRESS-002; C61-BOUNDARY-001; C61-BOUNDARY-002]

## Appendix Table

**Table S1. Negative diagnostic and route-boundary ledger.** This supplementary table records model branches and data-alignment paths that did not pass the manuscript gate, including Phases 33-53 and the Phase 58/59 density-sensitive diagnostics. [C61-APPX-001]

## Next-Branch Gate Table

**Table S2. Gates for future model branches.** Candidate A is paused pending a new validation-visible `spot_size` signal, Candidate B is blocked by the Phase 59 density gate, and Candidate C remains blocked by scan-path registration data. [C61-GATE-001]

## Figure Caption From Existing Phase 56 Asset

**Figure 1. Seed-stable `spot_size` transfer across broad12 and broad21.** Bars summarize the three required error metrics for `broad_process_v1`, no-process Macro PINN, and the strongest classical baseline under the fixed `spot_size` split. The figure should be paired with Table 1 and should not be reused to imply density-invariant or universal process-axis robustness. [C61-MAIN-001; C61-BOUNDARY-001]
