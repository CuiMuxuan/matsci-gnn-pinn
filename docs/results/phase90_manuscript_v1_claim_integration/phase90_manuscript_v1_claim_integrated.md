# Phase 90 Manuscript v1 Claim-Integrated Draft

## Scope And Venue Gate

Main claim: `fixed-sampling broad12/broad21 spot_size under broad_process_v1, seeds 7/1/2`.

This manuscript v1 integrates writing-ready literature evidence from Phase 89 with the Phase 61/74 experimental claim package. It remains provisional for venue-specific section order, citation density, and caption style because the target venue has not been provided.

Excluded from the main claim:

- density-invariant robustness
- universal process-conditioning success
- laser_power, scan_speed, or full-process strong-baseline wins
- source-path or Green's-function broad12/broad21 success under current data registration

## Introduction And Dataset Context

AM-Bench can be described as a public NIST-led additive-manufacturing benchmark ecosystem, with the project experiments tied to the mds2-2716 thermography/scan-strategy record and diagnostic microscopy tied to mds2-2718. [P89-EVD-AMBench-001;P89-EVD-AMBench-002;P89-EVD-MDS2-2716;P89-EVD-MDS2-2718]

Writing support:

- [P89-EVD-AMBench-001] Additive Manufacturing Benchmark Test Series (AM-Bench) (National Institute of Standards and Technology, 2026 access). AM-Bench is an appropriate public benchmark ecosystem for additive-manufacturing model validation and generalization claims.
- [P89-EVD-AMBench-002] AM Bench Data Management Systems (National Institute of Standards and Technology, 2023 page status; 2026 access). AM-Bench data and metadata are publicly discoverable through NIST data-management systems, including the Public Data Repository.
- [P89-EVD-MDS2-2716] AM Bench 2022 Measurement Results Data: In-situ Thermography and Scan Strategy for Laser-scanned Single Tracks and Pads on Bare In718 (National Institute of Standards and Technology, 2024 metadata update, DOI 10.18434/mds2-2716). The project experiments are based on a public NIST PDR thermography/scan-strategy dataset for AMB2022-03 single tracks and pads.
- [P89-EVD-MDS2-2718] AM Bench 2022 Measurement Results Data: Optical Microscopy of Laser-scanned Single Tracks and Pads (AMB2022-03) (National Institute of Standards and Technology, 2022 metadata update, DOI 10.18434/mds2-2718). Optical microscopy is a public auxiliary AM-Bench source used here only as diagnostic microstructure evidence.

Guard: Do not imply that all AM-Bench modalities were used or that registration blockers are solved.

## Related Work And Method Context

PINNs are established for forward and inverse PDE problems, but physics-informed ML has known optimization and data-integration limitations that motivate gated validation rather than architecture-only claims. [P89-EVD-PINN-001;P89-EVD-PINN-002;P89-EVD-PINN-003]

Feature-wise modulation is a recognized conditioning mechanism; the paper may describe the `spot_size` branch as a process-conditioned Macro PINN route guarded by the project's empirical route-selection evidence. [P89-EVD-FILM-001]

Writing support:

- [P89-EVD-PINN-001] Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations (Raissi, Perdikaris, and Karniadakis, 2019, DOI 10.1016/j.jcp.2018.10.045). PINNs support forward and inverse PDE problem settings and are a suitable conceptual baseline for physics-informed thermal modeling.
- [P89-EVD-PINN-002] Physics-informed machine learning (Karniadakis, Kevrekidis, Lu, Perdikaris, Wang, and Yang, 2021, DOI 10.1038/s42254-021-00314-5). Physics-informed ML is relevant to forward/inverse problems, high-dimensional settings, and data-physics integration, while limitations should be stated explicitly.
- [P89-EVD-PINN-003] Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks (Wang, Teng, and Perdikaris, 2021, DOI 10.1137/20M1318043). PINN optimization can suffer from gradient-flow pathologies, supporting careful wording around sparse/heterogeneous thermal data tradeoffs.
- [P89-EVD-FILM-001] FiLM: Visual Reasoning with a General Conditioning Layer (Perez, Strub, de Vries, Dumoulin, and Courville, 2018, DOI 10.1609/aaai.v32i1.11671). Feature-wise affine modulation is a recognized conditioning mechanism; this supports the process-conditioning mechanism description, not the AM-Bench result itself.

Guards: Tie limitations to representative literature and project diagnostics; avoid claiming the literature proves this exact failure mode. Do not claim FiLM literature alone validates AM-Bench performance; performance must cite Phase 55/60/74 artifacts.

## Methods

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


Phase 90 integration note: the conditioning mechanism may be described with FiLM-style feature-wise modulation support, but performance claims must remain tied to Phase 55/60/74 artifacts rather than to the conditioning literature alone. [P89-HANDOFF-CONDITIONING; C61-METHOD-001]

## Results

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

## Tables And Figures

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

## Limitations And Appendix Boundaries

The following boundaries remain limitations, appendix evidence, or future-work gates rather than main claims:

- `C74-EXCL-001`: density-invariant robustness -> excluded_from_main_claim
- `C74-EXCL-002`: universal process-conditioning success -> excluded_from_main_claim
- `C74-EXCL-003`: laser_power, scan_speed, or full-process strong-baseline wins -> excluded_from_main_claim
- `C74-EXCL-004`: source-path or Green's-function broad12/broad21 success under current data registration -> excluded_from_main_claim
- `C74-GATE-A`: Candidate A: bounded physical spot-size parameterization -> paused_no_training_signal
- `C74-GATE-B`: Candidate B: validation-auditable route policy -> blocked_no_validation_visible_route_policy_signal
- `C74-GATE-C`: Candidate C: data-aligned heat-kernel or Green's-function features -> blocked_by_registration_data
- `C74-GATE-DENSITY`: density-failure-driven model expansion -> block_density_failure_driven_model_expansion
- `C74-GATE-TRAINABLE`: all current trainable model branches -> no_trainable_model_opened

Phase 75 Bayesian inverse closure, Phase 79/80 bounded spot-size parameterization, and Phase 81 registered-target intake remain diagnostic or future-work evidence unless a future branch passes the local/no-training and A100 gates.

## Remaining Venue Dependency

No final venue-specific style claim is allowed yet.

Unresolved dependency: target venue, author guide, or 3-10 accepted benchmark papers.

Guard: Keep final section order, citation density, and caption style provisional.
