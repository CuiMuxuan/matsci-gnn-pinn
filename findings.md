# Findings: MatSci GNN-PINN

## Evidence Discipline

| State | Meaning |
|---|---|
| Candidate | Search result found; identity or relevance is not yet verified. |
| Verified | Title, authors, year, venue/preprint source, and technical relevance checked against a stable source. |
| Unresolved | Evidence is incomplete, conflicting, inaccessible, or only superficially similar. |
| Rejected | Verified as outside the proposal's actual task or method boundary. |

## Initial Project Context

- Repository: `C:\code\matsci-gnn-pinn`
- Goal: assess originality and create a publication-quality research design for a materials-science GNN-PINN project.
- Primary supplied material: `C:\Users\cjh02\.codex\attachments\f79079be-6017-4610-8b06-fd6021525a61\pasted-text.txt`
- Current innovation claims: pending extraction from the handoff document and repository.

## Literature Evidence Register

| ID | Claim / search theme | Source | State | Relevance | Notes |
|---|---|---|---|---|---|
| LIT-001 | Pending claim decomposition | Pending | Unresolved | Pending | Do not infer novelty before proposal extraction. |

## Initial Observations

- A novelty assessment will distinguish exact method-and-task overlap from partial overlap in architecture, physics constraints, training protocol, or application domain.
- No publication-quality claim will be made from repository names, abstracts, or search-result snippets alone.

## Handoff Extraction: Preliminary Claim Boundary

- **Verified project-status evidence:** the handoff records a narrow positive result for `broad_process_v1` / `spot_size` under strong-baseline review: a lightweight process-conditioned Macro PINN / FiLM route in a specific process-axis setting.
- **Explicitly unsupported claim:** the record says this must not be represented as a complete GNN-PINN success, a general process-conditioned modeling success, or density-invariant robustness.
- **Second-paper status:** synthetic inverse-heat hidden-source/closure evidence exists only as a mechanism prototype. It is not yet sufficient for an AM-Bench hidden-closure paper or a complete second main-paper result.
- **Closed / negative routes:** early microstructure GNN, source-path graph/Green features, dense neural operator/CNN-operator targets, failure-informed/adaptive sampling, a low-capacity hidden-closure head, and uncertainty-guided acquisition did not provide a publication-ready positive result in the documented gates.
- **Execution constraint:** the remote server has been released. No remote synchronization, SSH, or training escalation is authorized by the handoff; current work must begin with local evidence refresh and design gates.

## Implication for Novelty Search

The literature search will test two separate hypotheses rather than treating the repository title as a single innovation:

1. A narrowly scoped, process-conditioned Macro PINN / FiLM surrogate for additive-manufacturing thermal/process response under leakage-safe strong baselines.
2. A physics-constrained inverse-heat hidden-source/closure formulation, first validated synthetically and only later transferable to AM-Bench or NIST AMMT observations.

## Repository Baseline: Verified Local Evidence

- **Core implementation:** `src/gnnpinn/train/macro_pinn.py` is the primary Macro PINN training surface. The repository exposes `concat` and `film` input conditioning and scripts for condition-specific experiments.
- **Physical/application setting:** the documented Phase 23 pipeline builds calibrated multi-line `mds2-2716` temperature field data from `ThermalData/*/Signal`, conditioned on `laser_power_W`, `scan_speed_mm_s`, and `spot_size_um`.
- **Generalization design:** the repository evaluates line, laser-power, scan-speed, spot-size, and full-process grouped holdouts to distinguish line memorization from process-axis generalization.
- **Quantified narrow result:** the README records an early process-conditioned result improving held-out-line test RMSE from `175.127058` to `157.793227` and hot-region q90 RMSE from `351.525048` to `316.794319`; it also states that the early run did not beat the train-mean baseline globally.
- **Conditioning result boundary:** globally standardized process features are material. `scan_speed` favored concat, while `spot_size` favored FiLM and beat the train-mean baseline in a focused three-seed check. Concat+FiLM stacking and trainable routed conditioning were not universal improvements.
- **Inverse formulation:** Phase 164 and Phase 169 encode synthetic inverse-heat cases and a hidden-source/closure identifiability gate. The code includes source position/width/amplitude and closure coefficient parameters, but the documented evidence is synthetic.
- **Graph evidence:** Phase 148 provides a reproducible path/contact graph audit of AMMT source paths, including order-sensitive reheat and contact statistics. It is evidence for a possible physical descriptor, not proof that a GNN route improves the target task.

## Searchable Claim Matrix

| Claim ID | Exact search core | Prior-art risk before search | Required differentiator |
|---|---|---|---|
| C1 | `FiLM physics-informed neural network additive manufacturing thermal` | High | Controlled process-axis generalization and physical validation, not FiLM itself. |
| C2 | `PINN additive manufacturing temperature prediction process parameters holdout` | High | Leak-safe unseen-process/scan strategy evaluation across more than one build or alloy. |
| C3 | `inverse heat PINN hidden source closure identifiability` | High | Identifiability theorem/diagnostic plus AM observational transfer and uncertainty calibration. |
| C4 | `graph neural network laser powder bed fusion thermal history` | High | A graph encoding derived from causal scan-path/reheat physics that beats non-graph controls on external data. |

## Literature Evidence Register

| ID | Claim / search theme | Source | State | Relevance | Notes |
|---|---|---|---|---|---|
| LIT-001 | Pending claim decomposition | Local handoff + repository | Verified | High | Claim map C1-C4 completed. |
| LIT-002 | C1/C2: process-conditioned PINN AM thermal surrogates | Search queue | Candidate | Pending | Search formal papers and preprints separately. |
| LIT-003 | C3: inverse heat hidden-source/closure PINNs | Search queue | Candidate | Pending | Require identifiability and real-data transfer checks. |
| LIT-004 | C4: graph/scan-path AM thermal modeling | Search queue | Candidate | Pending | Comparator and negative-result context only unless a new route is proposed. |
| LIT-005 | `Machine learning for metal additive manufacturing: predicting temperature and melt pool fluid dynamics using physics-informed neural networks` | Exa discovery, DOI `10.1007/s00466-020-01952-9` | Candidate | High | Peer-reviewed 2021 prior art reported by search; identity and task details pending Crossref/publisher verification. |
| LIT-006 | `Single-track thermal analysis of laser powder bed fusion process: Parametric solution through physics-informed neural networks` | Exa discovery, DOI `10.1016/j.cma.2023.116019` | Candidate | High | 2023 LPBF parametric heat-equation PINN; process parameter and material-property conditioning reported by discovery result. |
| LIT-007 | `Physics-Informed Surrogates for Temperature Prediction of Multi-Tracks in Laser Powder Bed Fusion` | arXiv `2502.01820` | Candidate | High | 2025 preprint reporting DeepONet/PINN, multi-track temperature fields, tool paths, and laser parameters. |
| LIT-008 | `Physics-Informed Machine Learning Regulated by Finite Element Analysis for Simulation Acceleration of Laser Powder Bed Fusion` | arXiv `2506.20537` | Candidate | Medium-High | 2025 FEA-regulated PINN preprint; claims new-process-parameter generalization. |
| LIT-009 | `Real-Time 2D Temperature Field Prediction in Metal Additive Manufacturing Using Physics-Informed Neural Networks` | arXiv `2401.02403` | Candidate | Medium | 2024 AM PINN preprint; verify title/authors/scope. |

## Early Prior-Art Verdict (Not Yet Final)

- **C1/C2 architectural core is not novel enough on its own.** Published work already covers parametric LPBF thermal PINNs, and a 2025 multi-track preprint covers PINN/operator surrogate learning with tool paths and laser parameters.
- **Potential differentiation must be more specific than FiLM or process conditioning:** e.g., experimentally grounded cross-build extrapolation, causal scan-history representations with falsification controls, calibrated uncertainty, and a demonstrably identifiable hidden physics term.
- **C3/C4 remain open questions, not positive novelty claims.** They require focused discovery and identity verification before a defensible comparison can be made.

## Second Discovery Pass: Risk Expansion

- **FiLM/conditioning is established methodology:** the search surfaced feature-conditioned physics-informed operator/PDE methods, including FiLM time conditioning and parameter-encoder variants. FiLM can be an ablation, not a paper-level contribution.
- **C3 source inversion is not a blank space:** candidates include `Temperature field inversion of heat-source systems via physics-informed neural networks` (2022), `Heat source field inversion and detection based on physics-informed deep learning` (2025), and work on static/moving heat-source inverse conduction and theoretical convergence. A new source/closure method needs an identifiable object and a real AM validation problem.
- **C4 has direct prior art:** `Temperature Distribution Prediction in Laser Powder Bed Fusion using Transferable and Scalable Graph Neural Networks` (arXiv:2407.13838) reports GNN thermal prediction under user-defined scan strategies, and graph-theory LPBF work already reports in-situ infrared thermography validation. A generic scan-path graph GNN would be a direct overlap.
- **A promising gap is not yet demonstrated:** the available sources do not by themselves prove that no work jointly establishes causal scan-history descriptors, identifiable latent heat-source discrepancy, calibrated posterior uncertainty, and external AM observations under strict cross-build/extrapolation splits. This must be tested through identity-verified full sources and a deliberately designed study.

## Verification Worklist

`literature_candidates.csv` now contains candidate records for cross-source identity verification. No item in this worklist is writing-ready yet.

## Identity Verification: Completed Priority Records

- **LIT-005, LIT-006:** title, authors, year, venue, DOI, and article type agree between Crossref and OpenAlex. Both are verified journal prior art for the high-level AM-PINN claim.
- **LIT-007 through LIT-010:** title, author list, date, and identifier were checked against the official arXiv Atom metadata. They are verified preprints, but not peer-reviewed articles unless a later journal version is located.
- **LIT-011:** the journal article `10.1115/1.4047619` is verified through Crossref and OpenAlex. `10.1115/msec2020-8433` is its closely related conference proceeding and must not be double-counted.
- **LIT-012 and LIT-013:** title, authors, year, venue, DOI, and article type agree between Crossref and OpenAlex. They show that generic physics-informed heat-source field inversion/detection is established.
- **LIT-014:** title, authors, venue, and DOI agree; Crossref lists 2026 while OpenAlex lists 2025. It remains unresolved for bibliographic-year use, but its existence reinforces the active inverse-source theory landscape.

`literature_evidence_register.csv` is the authoritative local register for the records above. All items have `writing_ready = no` for manuscript drafting because full-text evidence locations have not yet been extracted.

## Focused AM Collision Check

- **AM-Bench heat-source calibration:** `Physics guided heat source for quantitative prediction of IN718 laser additive manufacturing processes` (npj Computational Materials, 2024, DOI `10.1038/s41524-024-01198-6`) uses AM-Bench Challenge 3 context, a physics-guided volumetric heat source, surrogate-assisted calibration, and quantitative process-condition predictions. This directly blocks presenting AM-Bench heat-source calibration as new.
- **Inverse heat source in LPBF:** `Volumetric heat source calibration for laser powder bed fusion` applies inverse heat conduction to fit a double-ellipsoid source across laser powers and scan speeds. A generic inverse-source parameterization is therefore not a contribution.
- **Uncertainty calibration in LPBF:** Bayesian model-updating and data-informed UQ work already calibrate thermal-model parameters against empirical melt-pool data. Bayesian language alone is not new.
- **Toolpath/geometry generalization:** work published in 2026 reports a U-Net with geometry and time/gradient representations for toolpath generalization. Toolpath OOD evaluation is essential evidence, not an exclusive novelty claim.
- **In-situ PINN data assimilation:** an existing 2D temperature-field PINN reports integration of in-situ measurements; real-sensor supervision alone is not enough.

## Strategic Consequence

The only credible route to a Q1-class contribution is a **joint, testable claim** that prior work has not established: a physically interpretable, identifiable discrepancy mechanism whose transportability is demonstrated under controlled scan-history interventions and independently calibrated, multi-build AM observations. This is a hypothesis to verify, not yet a novelty claim.

## Local Data Capacity Check

- `data/` contains 52 files totaling roughly 840.5 MB. The dominant local source is AM-Bench 2022 single-track material: a 524.5 MB thermography HDF5 file plus multiple cross-section TIFF assets.
- `outputs/` has 99 relatively compact artifacts (about 5.7 MB), while `docs/results/` contains extensive gate documentation and within-source ablation records.
- The local evidence supports process-axis holdouts and reproducibility checks, but no visible independent-build or cross-machine validation asset was found in the inventory.
- **Allowed claim strength today:** descriptive/pilot evidence only for any new method. A transferability, calibrated-UQ, or causal-closure claim must wait for an independent observation set or a controlled new experiment.

## Public Data Audit: Initial Confirmed Lead

- NIST states that AM-Bench 2022 PBF-LB benchmarks cover IN718 3D builds, different laser scan patterns, in-situ measurements, microstructure, residual strain/deflection, and mechanical behavior across associated data publications.
- The official AMB2022-03 publication page explicitly identifies public in-situ thermography and scan-strategy data for laser-scanned single tracks and pads on bare IN718; its companion optical-microscopy dataset is DOI `10.18434/mds2-2718`.
- AMB2022-01 and AMB2022-02 are high-value candidates because they describe 3D builds and different scan patterns. They may enable a physically independent evaluation set, but exact file-level joins, build identifiers, and accessible thermal measurements must be verified before this is treated as an available cross-build benchmark.
- Jina Reader did not return readable bodies for the dynamic NIST pages. This is an access-layer limitation; it does not negate the NIST page's explicit public-data statement. File-level feasibility will be checked through direct NIST responses and the repository's locally downloaded provenance.
- Direct NIST HTTP requests resolve both the human-facing AMB2022-03 publication page and `https://data.nist.gov/od/id/mds2-2716` to a machine-readable JSON record for DOI `10.18434/mds2-2716`; it contains thermography and scan-related metadata. `mds2-2718` similarly resolves to JSON for the companion optical-microscopy record. This makes file-level provenance auditing feasible without a login.
- The official AM-Bench direct-link page exposes a larger family of NIST DOI records, including IDs associated with AMB2022-01, AMB2022-02, and AMB2022-03. The next check is to map DOI-to-challenge and determine whether 01/02 supply independent builds and compatible observation targets.
- File-level inspection confirms that `mds2-2716` has `accessLevel: public` and a NIST Open License. Its components include the staring-camera thermography HDF5, a pad XYPT scan-strategy HDF5, sample-surface images, checksums, and a README. `mds2-2718` has downloadable single-track cross-section images and supporting measurement artifacts.
- Multiple bare-plate identifiers occur in the publication metadata, but this is not yet evidence of independent 3D builds. Treat them as possible replication/provenance candidates until the README and AMB2022-01/02 records establish the physical-build relationship.
- The official AM-Bench cross-reference maps AMB2022-02 to IN718 3D builds made with different laser scan patterns. Its linked public records include residual elastic strain/residual stress/part deflection (`mds2-2692`), serial sectioning/X-ray CT (`mds2-2711`), and the custom scan-strategy modeling-challenge dataset (`mds2-2617`). This is a candidate external/downstream validation family, not yet a confirmed thermal-field benchmark.
- Direct NIST API verification corrects the record mapping and confirms all are public: `mds2-2617` is the AMB2022-02 custom-laser-scan-strategies 3D-build challenge (21 components); `mds2-2692` is 3D-build microstructure (366 components); `mds2-2711` is residual elastic strain/residual stress/part deflection (27 components); and `mds2-2767` is serial sectioning/X-ray CT (8479 components).
- The local handoff's `mds2-2044` label returns HTTP 404 through the current NIST API. It is not yet eligible as a public-source dependency; its existing local ZIPs require provenance tracing before use.
- Component-level inspection confirms that `mds2-2617` publishes a README, challenge templates, and three scan-strategy trajectory files: `AMB2022-02-AMMT-XYPT-V6.h5`, `V7.h5`, and `V8.h5`. These encode the custom-scan-policy intervention required by the redesigned study.
- `mds2-2692`, `mds2-2711`, and `mds2-2767` form an AMB2022-01 3D-build physical-response chain: multi-modal microstructure, diffraction/contour-method strain-stress and deflection, plus spatially registered EBSD/XRCT data. They are useful downstream/physical targets, but not proof that 3D in-situ temperature fields are available.
- The `mds2-2617` metadata points to `mds2-2607` for nominal 3D build geometry, material data, and build information. That record is the next high-priority check for thermal-observation availability and identifier joins.

## Public Data Audit: Confirmed Usable Chain

- `mds2-2607` is public under the NIST Open License and is the AMB2022-01 nominal 3D-build record. It contains B6/B7/B8 in-situ thermocouple time-series CSVs and `AMB2022-01-AMMT-XYPT_v1.h5` with layer-wise commanded position, power, time, and instrument-trigger information.
- The AMB2022-01 README states that XYPT trigger indices synchronize scan-path arrays with camera-frame capture, although coaxial melt-pool camera data were not released for those challenges. Thermocouples are therefore the confirmed 3D thermal observation, not a full 3D temperature video.
- `mds2-2617` provides the controlled scan-policy family V6/V7/V8. Its README identifies synchronized laser/galvo commands, layer groups, and solid-cooling-rate/time-above-melting challenge templates.
- `mds2-2716` supplies the high-resolution local calibration domain: raw pre-temperature-calibration thermographic HDF5 data for single tracks and pads plus pad XYPT commands. The companion `mds2-2718` provides cross-section targets.
- **Feasible public protocol:** calibrate local measurement/operator components on AMB2022-03; train/calibrate the causal kernel on a subset of AMB2022-01 B6/B7/B8 thermocouple trajectories and XYPT histories; hold out a full build; use AMB2022-02 V6/V7/V8 scan policies as intervention/counterfactual evaluation; attach public AMB2022-01 microstructure/strain/deflection/CT outcomes as downstream physical consistency targets where sample joins can be verified.
- **Boundary:** AMB2022-02 must not be claimed as having released 3D thermography unless a file-level search finds it. Its confirmed assets are scan commands and challenge templates.

## Simulation Capacity And Proper Role

- CodeGraph confirms that the repository already includes a synthetic hidden-source/closure identifiability gate (`build_phase169_hidden_source_closure_identifiability_gate.py`). Its cases vary source position, width, amplitude, closure coefficient, ambient condition, and observation noise; it also includes an explicit no-closure source control.
- Existing moving-source inversion and XYPT tooling provide a second layer of support: the Phase 50 probe fits moving-source parameters, Phase 52 reads HDF5 XYPT paths, and Phase 148 derives ordered exposure/reheat/contact statistics and compares them with shuffled-path controls.
- Therefore, simulation data can be generated now without new external software for: known-truth parameter recovery, sensor-noise/sparsity stress tests, counterfactual scan-order experiments, history-shuffle negative controls, and pretraining/initialization of constrained components.
- Simulation data must not replace real AM observations in the central claim. It can establish identifiability and reduce sample demand, while AMB2022-03/01 data remain the calibration and independent-real-evaluation evidence.

## Phase 181 Registration Resolution

- The GPU server now holds a checksum-verified subset of NIST `mds2-2715`: README,
  all processing scripts, the Photron registration arrays, and B6/B7/B8 TAM and SCR
  HDF5 targets (19 files, 952,578,673 bytes).
- All six targets pass SHA-256 and HDF5 schema checks. Each is a `312 x 304 x 640`
  field, carries a direct `ScanStrategy_ref` to `mds2-2607`, and provides a rigid-2D
  millimeter registration grid and 40 um layer spacing.
- The linked XYPT source has 312 contiguous layers, a 100 kHz digital rate, and
  `Trigger_bit2=StaringCamera`; command X/Y coordinates are in millimeters.
- This resolves a narrower but rigorous route: per-layer, trigger-aware, physical-space
  TAM/SCR supervision. It does **not** resolve a global wall-clock schedule for raw
  frame-level causal histories, so Phase 180 remains a hard scope boundary.
- The resulting publication-safe task is a registered layer-space surrogate with
  build-level replication and scan-history controls, not a claim that the current
  public subset supports arbitrary absolute-time movie reconstruction.

## Phase 182 SCR Unit Resolution

- The B6 and B8 SCR HDF5 datasets declare `units=s`, while B7 declares `C/s`.
- The discrepancy is an HDF5 packaging metadata defect, not an unresolved target
  semantic: NIST `CR_v1.m` computes `DT./Dt` and labels the result `[oC/s]`; the
  HDF5 creation script writes the incorrect `s` attribute for generated SCR files.
- Phase 182 records each source declaration and a metadata-correction flag, then
  exposes the model target as `target_scr_C_per_s`. This correction is traceable in
  the output manifest and does not overwrite the source files.

## Phase 184-186 Mechanism Checks

- B6-only ridge baselines show that scan-history features improve both B7 validation
  and B8 held-build RMSE relative to coordinate/layer-only features: B8 TAM improves
  from `3.863e-4` to `3.124e-4` s and B8 SCR improves from `1.326e6` to `1.164e6` C/s.
- A causal heat-kernel descriptor constructed only from XYPT-derived local command time,
  source energy, and registered physical coordinates produces further positive B7/B8
  gains over full ridge for both targets.
- A fixed-seed within-layer shuffle of scan-history columns worsens validation and test
  RMSE for both targets. The descriptor therefore carries spatially aligned information
  beyond its marginal distribution and the coordinate field.
- These are still low-capacity empirical checks, not a publication claim. They justify a
  bounded candidate-model comparison with the fixed controls, not unconstrained GPU
  hyperparameter search.

## AMB2022-03 Raw Thermal Descriptor Boundary

- Phase 195 has passed on the independently checksum-verified AMB2022-03 thermography HDF5. The observed raw Signal schema is uint16, 12-bit, 700 x 640 x 304, with a threshold of 100 digital levels and the expected units.
- The frozen Phase 196 feature set consists only of raw-signal mean, population standard deviation, maximum, exact linear P99, above-threshold fraction, active-frame fraction, and mean/standard deviation of per-frame maxima.
- Extraction must stream one `Line_*` dataset in fixed frame chunks and use a 12-bit histogram for exact P99; it must not read cross-section labels or the ThermalCal conversion metadata. This preserves a clean separation between label-free feature construction and a later audited target join.

## AMB2022-03 Calibration Table

- Phase 197 joins the Phase 196 descriptor table to the Phase 193 per-condition cross-section summary only after feature extraction is complete. The resulting table has 21 rows, eight raw descriptors, three process fields, and depth/width means plus duplicate-section standard deviations.
- The seven leave-one-process-setting-out folds each hold out all three conditions at one exact `(power, speed, spot size)` setting. This is substantially stricter than random row splitting, but it remains local AMB2022-03 calibration evidence rather than external 3D-build validation.
- The next empirical comparison must use a fixed low-capacity baseline family, train preprocessing only within the fold, report every preregistered variant rather than selecting after test results, and distinguish prediction error from the two-section measurement spread.

## AMB2022-03 Fixed Baseline Result

- Under the frozen seven leave-one-setting-out folds, `process_ridge_control` is materially better than the primary process-plus-global-thermal ridge for depth (pooled RMSE 19.64 um versus 44.33 um). The raw-thermal-only ridge is substantially worse (119.11 um), and the shuffled-thermal negative control is similar to process-only (20.47 um).
- For width, the process-plus-global-thermal ridge is again not better than process-only (15.47 um versus 14.16 um), while the shuffled-thermal negative control is lower (11.74 um). This pattern specifically fails to support a stable signal from the eight global descriptors; it must not be used to choose a different model after holdout evaluation.
- All pooled median repeatability-normalized absolute residuals exceed one, including process-only depth 8.31 and width 7.36. The grouped prediction error is therefore much larger than the observed duplicate-section spread. The duplicate standard deviation is a measurement-repeatability reference, not a predictive interval.
- The defensible conclusion is negative and bounded: these eight global raw-signal summaries do not demonstrate robust additive geometry-prediction value under the frozen grouped protocol. This rules out capacity escalation on this branch rather than proving that the full thermography data are irrelevant.

## ThermalCal Evidence Boundary

- The AMB2022-03 HDF5 exposes `Coeff_a=0.9655`, `Coeff_b=197.2`, `Coeff_c=4.392e7`, `R-square=0.9988`, and calibration RMSE 4.923 as `ThermalCal` group attributes. It has no `Coeff_e` or emissivity attribute in either the calibration group or the first raw Signal attributes.
- The official report, Brandon Lane, *Thermal Calibration of Commercial Melt Pool Monitoring Sensors on a Laser Powder Bed Fusion System* (NIST AMS 100-35, DOI 10.6028/NIST.AMS.100-35), gives Eq. (6) as `T = c2 / (A ln(C/S + 1)) - B/A`, with `c2 = 14388 um K`, and explains that the calibration does not provide absolute real melt-pool temperature.
- The HDF5 model string contains an extra undefined `e` symbol (`c*e/x`) and is not an unambiguous transcription of the official equation. No converted-temperature descriptor may be constructed until the original AMB2022-03 calibration implementation or an official definition of that symbol is obtained.

## B8 Physical-Target Feasibility

- NIST's verified `mds2-2692` description document lists four measured cross sections: B7/P1 L7-L9 (as-built, XZ), B6/P2 L7-L9 (heat-treated, XZ), B8/P3 L9 (as-built, XY), and B8/P3 L10-W3 (as-built, XY). The B6 heat-treated observation is not comparable as an as-built thermal target.
- A public B8/P3/L9 IPF-Z TIFF component was selected from the official file list rather than downloading the 31.9 GB record. It is 30,826,644 bytes and its SHA-256 matches the NIST manifest: `842d6f93e4b7af4fd6b6f333ec57256bc4446e4ef0c8d68d413f9d8007a37495`.
- Metadata-only inspection shows a 5516 x 2905, 4-channel raster with 96 dpi X/Y tags. It lacks image orientation, a physical specimen origin, ModelPixelScale, ModelTiepoint, and GeoKey tags. The dpi is raster display metadata, not evidence of a specimen-to-build scale or origin.
- The official AMB2022-01 result description confirms B8/P3/L9 is an as-built XY cross section and refers to L9 midplane cuts, but does not provide a numerical section elevation, build-layer index, or registered XYPT transform. Therefore this record is a promising future physical-consistency resource, not a current evaluation target.

## Phase 207 Official Evidence Acquisition Scope

- Active evidence gaps are deliberately narrow: (1) an official AMB2022-03 implementation or documentation that defines the `e` term in the stored `ThermalCal` formula, and (2) an official B8/P3/L9 artifact giving section elevation/layer association, physical specimen-plane origin/orientation, and a reproducible transform to nominal B8/P3 XYPT coordinates.
- Search and download are permitted only for lawful, official, integrity-checkable material. Neither a secondary citation nor a visually inferred TIFF registration can satisfy either gap.
- First exact-keyword Exa batch (`RegressionF_ArrayAvg`/`ThermalCal` and AMB2022-03 coefficient names) returned generic same-name calibration material rather than a NIST implementation or registration artifact. No hit was accepted, downloaded, or used as evidence; subsequent acquisition switches to official record manifests, attached source files, and repository code search.
- Remote acquisition succeeded through Python's standard library after the minimal server image was found to lack `curl`. Official NIST metadata artifacts are now stored under `/root/matsci-gnn-pinn-data/evidence/phase207/`: `mds2-2716_metadata.json` (15,433 bytes; SHA-256 `1a340657648a217bd537886d9e845dab7fdcfbd9940657faa80c791f6619fa80`) and `mds2-2692_metadata.json` (465,982 bytes; SHA-256 `653e503e232513101ed313a0edfede0cec6aefee019092a61249b9610593238d`).
- The `mds2-2716` manifest has 11 components and exposes raw thermography, sample photos, checksum components, and `2716_README`; it does not presently expose a named calibration script or source-code component. This is evidence of a missing public artifact, not evidence that an undocumented temperature conversion is valid.
- The `mds2-2692` manifest has 366 components. Its complete paths must be screened for small description, coordinate, registration, or sample-metadata artifacts before any additional binary is downloaded.
- Path-level screening confirms that the only `mds2-2716` calibration-related component is `2716_README.txt` (12,573 bytes; manifest SHA-256 `ba44076ed51b69c0e4ca80ff0e2568eed2dc6459e85c9ad83b85860bee5760f2`). There is no public NIST component named as a calibration script, regression implementation, or source code.
- B8/P3/L9 entries in `mds2-2692` are EBSD TIFF/CSV/CTF/H5OINA-style measurement products and their checksums. The manifest has no component whose path declares a spatial registration, coordinate transform, sample-plane origin, or nominal XYPT link. The small official `2692_README.txt` and the 1.43 MB description PDF are the only remaining text-level sources worth inspecting before declaring the registration artifact unavailable.
- GitHub's exact global code search for `RegressionF_ArrayAvg` returned zero results. This does not prove the code never existed, but it rules out a public indexed GitHub implementation under that exact identifier.
- The checksum-verified `2716_README.txt` resolves one semantic ambiguity: `Cal_Method = RegressionF_ArrayAvg` means a regression function compiled by averaging all pixels in the array. It also identifies `Coeff_a`, `Coeff_b`, and `Coeff_c` as coefficients in `Model`, and calls `Model` only a functional-format string. It supplies no `Coeff_e`, emissivity field, Sakuma-Hattori definition, or declaration that the formula symbol `e` is Euler's constant. The temperature-conversion gate therefore remains unresolved.
- The checksum-verified `2692_README.txt` confirms that CTF headers may include image size/step, acquisition coordinate system, and detector/sample orientation. Those are EBSD-instrument coordinates; the README contains no `registration` entry and no mapping to the nominal B8/P3 XYPT coordinate system. They may improve local microscopy metadata but cannot by themselves establish the required build transform.
- `AMB2022-01-MS-PE_descriptions.pdf` confirms B8/P3/L9 and B8/P3/L10-W3 are as-built XY sections and states that some published XY data pass through L9's midplane. It points readers to the AMB2022-01 benchmark page for geometry/naming/coordinate-system diagrams, but supplies no numeric elevation, layer index, origin, orientation matrix, or XYPT transform.
- `NIST.AMS.100-35.pdf` was downloaded from NIST and text-extracted using preinstalled `pdftotext`. Its abstract explicitly states that the method does not provide an absolute calibration or the ability to ascribe real melt-pool temperatures to sensor signals. It identifies inverse Sakuma-Hattori parameters `A`, `B`, and `C`, not a fourth `e` coefficient in the AMB2022-03 HDF5 formula.
- The official AMB2022-01 Measurement and Result Descriptions PDF is now stored at `/root/matsci-gnn-pinn-data/evidence/phase207/AMB2022-01_Measurement_and_Result_Descriptions_v1.0.pdf` (2,326,291 bytes; SHA-256 `e0eebd0229fca7aebaae9fd22f3892d9a605948c5a834aee51f5d8a3b2e6b981`). It confirms B8/P3/L9 as an as-built XY section, but contains no `registration` term or coordinate-system specification for that EBSD target. Its nominal-coordinate discussion pertains to a different B7/P3 residual-strain measurement and cannot be transferred to B8/P3/L9.
- A bounded HTTP `Range` read (64 KiB, status 206) of the 988,870,559-byte B8/P3/L9 CTF avoids an unnecessary full download while exposing its native header: local grid `XCells=5336`, `YCells=2818`, `XStep=1`, `YStep=1`, `AcqE1=AcqE2=AcqE3=0`, and Euler angles in sample coordinate system `CS0`. Its rows begin at local `(X,Y)=(0,0)`. No nominal build origin, elevation/layer association, or XYPT transform appears in the header, so even the native EBSD data cannot close the spatial-registration gap.

## Phase 207 Acquisition Result

- Direct extraction of the full `2716_README.txt` ThermalCal block confirms the complete documented attribute list: `Cal_Method`, `Coeff_a`, `Coeff_b`, `Coeff_c`, `Model`, `Model_input`, `Model_output`, `R-square`, and `RMSE`. It does not define any `e` symbol, coefficient, emissivity field, or source-code URL. The README says deeper descriptions and links to analysis code would arrive in future publications/updates; none is listed in public version 1.0.0.
- The linked official AMB2022-03 Measurement and Challenge Descriptions PDF was downloaded to the remote evidence directory (1,281,604 bytes; SHA-256 `e2c188e82f7dc39c0be8553c6c11d8f106ff6f226e238f57d602d42981087431`). Text extraction finds zero occurrences of `ThermalCal`, `RegressionF`, `Coeff_a/b/c/e`, `Sakuma`, `emissivity`, or `calibration`. It cannot supply the missing formula definition.
- The remote evidence bundle is complete at `/root/matsci-gnn-pinn-data/evidence/phase207/`. Its machine-readable `phase207_acquisition_manifest.json` records eight acquired official/official-linked files, source URLs, byte counts, SHA-256 values, integrity status, bounded CTF probe, and the two unresolved gates. Manifest SHA-256: `616ed1aa8a29c55d98c2ba902999956bd0e392004d78b4c76c4771b84fffca11`.
- **Formal Phase 207 decision:** public official materials do not define the AMB2022-03 HDF5 formula's `e` or provide the required B8/P3/L9-to-XYPT transform. Keep converted-temperature features, absolute-temperature claims, EBSD pixel labels, coordinate-transform fitting, and physical-target model evaluation disabled. The only appropriate next acquisition is a response from the NIST dataset contact or a newly published official revision containing those two artifacts.

## Phase 208 Public-Web Completeness Audit

- Expanded Exa/GitHub discovery surfaced two new primary-source leads not included in the first PDR-component audit: the public NIST GitHub repository `usnistgov/ambench`, which NIST describes as retaining prior AMBench metadata/schema versions, and the NIST-linked paper *Additive Manufacturing In-situ and Ex-Situ Geometric Data Registration* (PMC10502900). Neither has yet been accepted as evidence; both require artifact-level inspection for an actual B8/P3/L9 transform or calibration implementation.
- The new search also confirmed that NIST exposes `mds2-2715` as “3D Builds In-situ Thermography and Data Processing Scripts (AMB2022-01).” Existing server files will be searched alongside its public manifest because common calibration/registration utilities could clarify the otherwise ambiguous AMB2022-03 metadata without treating a similar 3D-build routine as proof by default.
- Repeated public GitHub code search for `ThermalCal AMB2022` returned zero hits. This strengthens the inference that any solution is not indexed under those terms, but it does not rule out source or metadata stored inside `usnistgov/ambench` or an archival attachment.
- Artifact-level inspection of the existing checksum-verified NIST `mds2-2715` script bundle found `DataProcessingScripts/AMB2022_HDF5_Temperature_v1.m` and the accompanying `2715_README.txt`. They explicitly create `Calibration/ThermalCal` and `Calibration/Registration`, write `Cal_Method = RegressionF_ArrayAvg`, and document camera `Xgrid`/`Ygrid` registration to machine coordinates. This is a **candidate** implementation for interpreting AMB2022-03, not a resolution: equality of camera, calibration run, coefficient values, model string, and data-release version must be demonstrated before transfer.
- The NIST-owned `usnistgov/ambench` repository is current and public (`main`, updated 2026-05-13, 75 tree entries) but its path names contain no AMB2022, ThermalCal, B8/P3/L9, calibration, or registration identifiers. Content/history inspection is still required before it can be ruled out as metadata-only.
- Direct PMC HTML fetches for both registration papers were blocked by a browser-check page. This is an access-layer issue, so the next route is the NCBI/Europe PMC full-text XML APIs or the paper's publisher/NIST manuscript record, rather than treating the article as unavailable.
- The official `mds2-2715` temperature-packaging script provides a concrete interpretation candidate for the AMB2022-03 `e`: it writes `Model = f(x) = Te05 = (1.438775E-2)./a./log(c./(S_masked/emiss)+1)-b./a - 273.15`, along with `Cal_Method = RegressionF_ArrayAvg`, an `emiss` attribute of 0.5, and an emissivity-corrected temperature output. Thus `e` in the shorter AMB2022-03 string is plausibly an emissivity variable, **not** Euler's constant. This remains `candidate_needs_verification` until exact model-string/coefficients/camera-calibration identity is checked against AMB2022-03 itself.
- The remote clone of NIST's official `usnistgov/ambench` repository is pinned at commit `77adb06c6de95b9b97e1dd26d46561f29db927af` and contains versioned `data-releases/*.zip` bundles, not only schemas. Changelogs show AMB2022 specimen XML records and later support for calibration and coordinate systems. The B8/P3/L9 record must therefore be searched inside the released XML before retaining the registration blocker as public-unavailable.
- Europe PMC full-text XML endpoint calls returned HTTP 404 for both PMC IDs. This is logged as an endpoint mismatch/access limitation; NCBI Open Access metadata or publisher/NIST full text remains the next independent article route.
- The publisher page for DOI `10.1007/s40192-024-00371-5` explicitly says that coordinate rotation is applied *when necessary* to align EBSD measurement coordinates with the AM coordinate system. Its data-availability statement points only to the complete NIST `mds2-2692` microstructure record. Neither statement supplies a rotation angle, specimen-plane origin, elevation/layer association, or XYPT transform, so the paper is currently corroborating context rather than a registration artifact.
- The versioned AMBench XML exposed by the NIST repository identifies B8/P3/L9 as a 2.5 mm as-built leg, an EDM cut, and `Image_25`/L9. Its companion EBSD XML is internally inconsistent: the identifier denotes `XY_as-built`, while one description says heat-treated/XZ bridge region. This inconsistency reinforces that neither XML can be promoted to spatial-registration ground truth without a numerical transform.
- Direct publisher Table 3 inspection resolves the identity ambiguity but not the mapping: `AMB2022-718-AMMT-B8-P3-L9` is listed as `XY` / `As built`, while the heat-treated XY sample is `B8-P3-L6`. Table 3 contains no position, origin, orientation angle, pixel scale, section elevation, layer number, or coordinate-transform parameter. It can be used as verified sample-plane/condition evidence only.
- The article's Fig. 1 caption is an overview of bridge geometry, leg labels, hollow L10, and the thermography region; Fig. 2 names the AMMT coordinate center on the substrate. The public article page and its native HTML expose no `Supplementary`, `MOESM`, or electronic-supplement link. This is a bounded publisher-page check, not proof that an unpublished attachment cannot exist.
- Direct official PID inspection found no `coordinate`, `origin`, `transform`, `matrix`, `pixel`, `layer`, `plane`, `elevation`, `XYPT`, or `registration` field in the B8/P3/L9 sample PID, and only the already-known contradictory prose in the EBSD PID's embedded XML. The linked `Image_25` endpoint responds with a 2500 x 1254 PNG (481,064 bytes), not a numeric transform artifact. A schematic PNG cannot establish a reproducible EBSD-to-XYPT mapping without associated calibrated scale/origin metadata.
- In the resumed Agent-Reach Exa audit, AMB2022-03/ThermalCal queries return the NIST record, official challenge/measurement PDFs, and the related AMB2022-01 script record (`mds2-2715`), not an AMB2022-03 implementation or formula definition. One parallel query hit Exa's free-tier HTTP 429 limit and is explicitly treated as an access limitation rather than a negative result.
- GitHub's authenticated public code index returned zero results for exact `RegressionF_ArrayAvg`, exact `AMB2022_HDF5_Temperature_v1.m`, `ThermalCal org:usnistgov`, `B8-P3-L9 org:usnistgov`, and `mds2-2716 org:usnistgov`. This narrows the searchable public-code surface only; it cannot rule out an unindexed, private, or later-released artifact.

## Phase 208 Decision

- **ThermalCal:** unresolved. Public materials support only a candidate interpretation of `e` as emissivity from the related AMB2022-01 release; they do not formally bind that variable or executable calibration parameterization to AMB2022-03. Converted-temperature features and absolute-temperature claims remain disabled.
- **B8/P3/L9 registration:** unresolved. Public materials verify the sample as an as-built XY cross section and provide local EBSD/microscopy context, but no reproducible EBSD-to-nominal-XYPT origin, scale, orientation, and layer/elevation mapping. Pixel-level physical-target fitting/evaluation remains disabled.
- The negative decision is deliberately bounded to the official NIST records/PIDs/versioned AMBench artifacts, the Springer publisher pages, GitHub's public code index, and the successful portion of the Exa search as checked on 2026-07-14. It is not a claim that no private, unindexed, or future material exists.

### Phase 208 Cross-Record Results

- A focused metadata comparison establishes a high-strength calibration provenance link: AMB2022-03 and the official AMB2022-01 `mds2-2715` ThermalCal implementation share `Cal_Method = RegressionF_ArrayAvg`, `Coeff_a = 0.9655`, `Coeff_b = 197.2`, `Coeff_c = 4.392e7`, `R-square = 0.9988`, `RMSE = 4.923`, model input `Signal [DL]`, and output `Emissivity-Corrected Temperature [C]`. The 2715 script explicitly sets `emiss = 0.5` and implements `T_e0.5 = (1.438775e-2/a)/log(c/(S_masked/emiss)+1) - b/a - 273.15`.
- The AMB2022-03 HDF5 model string is malformed (`T(x) = 14388/a/log((c*e/x+1)-b/a;`) and omits the `emiss` attribute. The identical calibration signature plus official 2715 implementation makes `e = emissivity` with the documented value 0.5 a credible public interpretation, but the conversion must first pass an explicit numerical/units and scope gate. It supports only an emissivity-corrected radiance-equivalent quantity, never an absolute melt-pool temperature claim.
- The scripted equality report initially printed false for numeric fields because one side retained NumPy scalar formatting during string comparison. The visible values are identical; a typed comparison will be used before changing the gate status.
- The latest NIST AMBench 3.0 metadata release contains `AMB2022-718-AMMT-B8-P3-L9.xml` and `AMB2022_SEM_718_B8-P3-L9_XY_as-built.xml`. It adds specimen identity, EDM-cut provenance, `2.5 mm leg - as built`, EBSD montage parameters, and public PID/blob references. It contains no explicit coordinate-system/transform reference for the target measurement.
- The 3.0 SEM metadata description is internally contradictory (`name ... XY_as-built` and specimen condition as-built, but prose says heat-treated XZ bridge region). This rules out treating its qualitative section description as a registration truth. It does expose `https://ambench2022.nist.gov/rest/blob/download/575/` (the labelled bridge-geometry image) and two public PID records as next source objects to inspect.
- The next implementation-level audit invalidates the shortcut from metadata identity to executable equivalence. The actual official `AMB2022_01_3DBuildLayer_TAMandCR_Calcs_v2.m` calculation uses `emiss=0.5` but numerical parameters `a=8.388e-07`, `b=2.949e-05`, `c=9.305e+07`, whereas the 2715 packager's HDF5 attributes expose `.9655`, `197.2`, and `4.392e7`. The packager's same-looking Model string cannot therefore be treated as the exact AMB2022-03 calibration implementation. The thermal formula remains `unresolved_search_more`; no conversion gate is reopened.
- The B8/P3 public-source route is stronger than previously known: both specimen and EBSD PIDs resolve publicly, as does bridge-geometry blob `575` (PNG) and Springer article DOI `10.1007/s40192-024-00371-5`, *Location-Specific Microstructure Characterization Within AM Bench 2022 Nickel Alloy 718 3D Builds*. These sources must be mined for coordinate/section information before the spatial gap can be closed or retained.

## Phase 209 Offline Validation

- The local workspace retains 52 files under `data/` (881,319,635 bytes), but per the project rule these local copies are not treated as the authoritative training/evaluation data source. They were inventoried only; no raw HDF5, TIFF, workbook, or model artifact was opened.
- All ten local `.sha256` sidecars contain a structurally valid 64-hex digest. They are hash-only sidecars, so their format check does not constitute a recomputation or a file-content match.
- `data_availability_register.csv` has eight complete rows with seven `verified` and one `unresolved` record. `project_state.json` is valid JSON. Literature registers retain deliberately incomplete candidate metadata and are not promoted by this preflight.
- All 26 Phase 181--206 server scripts pass AST syntax parsing. Twenty-seven adjacent Phase 180--206 test modules exist, but were not executed locally because the project policy keeps real test execution with the remote data environment.
## Phase 212 Remote CPU Gate Revalidation (2026-07-22)

- The new remote instance is synchronized at commit `803edc9ee444cd4766e779bc0e3013376bea0eed`; the active repository and data roots are `/root/matsci-gnn-pinn` and `/root/matsci-gnn-pinn-data`.
- Its base Python is 3.10.12 with `h5py` and `numpy`, but no `pip`, `openpyxl`, or `pytest`. The pre-existing project `.venv` contains only `pyvenv.cfg`, so rebuilding that isolated environment cannot discard installed project dependencies.
- Phase 201 is metadata-only and does not read raw Signal arrays; Phase 202 consumes only the Phase 201 JSON. Phase 192 is the existing workbook structural validator and needs `openpyxl` in read-only mode. These are sufficient for the Phase 212 CPU gate without a GPU or a new model fit.
- The remote APT archive path is currently unavailable: index refresh exceeded 120 seconds without output, and a bounded targeted installation reached archive-download retries for `security.ubuntu.com` before its 180-second limit. The Phase 212 workbook check therefore needs a standard-library ZIP/XML implementation rather than a retry of this external packaging path.
- The new standard-library fallback passed its remote `unittest` check and the real Phase 192 intake reported no blockers. Phase 201 remains metadata-only, and Phase 202 confirms the existing scientific boundary: `e` remains undefined in the HDF5 formula, the HDF5-to-official formula mapping is ambiguous, and temperature conversion, fitting, and model training remain disabled. The workbook's local SHA-256 is recorded for provenance only; an official `mds2-2718` manifest hash match has not yet been established because the first remote NIST request was reset during TLS handshake.

## Phase 212 Completion: Remote CPU Evidence Revalidation

- The NIST-provided `pdrdownload.py` list-only route succeeded after the direct JSON metadata endpoint reset its TLS connection. The retained official `mds2-2718` file-list entry identifies `AMB2022-718-SH1-MeltPool_Cross-Section_Measurement_Results.xlsx` as a 25,811-byte XLSX with SHA-256 `2cfaac96aaca3dabb77b7029f842cdcc7e75c5a2cf3577d0734823246364a931`; the remote intake file matches both values exactly.
- The formal provenance evidence is `/root/matsci-gnn-pinn-data/derived/phase212/phase212_mds2_2718_workbook_provenance.json`; the retained official listing is `/root/matsci-gnn-pinn-data/evidence/phase212/mds2-2718_filelisting.csv` with SHA-256 `b96a8ac06b3ab13edec409cfc59289222d4f74cc01251b9838fe3f079d8a37e6`.
- Phase 192 revalidated the real thermography/XYPT/workbook schema with no blockers through the standard-library XLSX reader. Phase 201 revalidated the ThermalCal metadata without opening a raw Signal array. Phase 202 correctly remains conversion-blocked by the undefined HDF5 `e` and missing unambiguous mapping to the NIST equation.
- The repeatable result `/root/matsci-gnn-pinn-data/derived/phase212/phase212_remote_cpu_gate_revalidation.json` reports `phase212_remote_cpu_gate_revalidation_complete_cpu_evidence_boundary_preserved`. It explicitly retains `temperature_conversion_allowed=false`, `calibration_fitting_allowed=false`, `model_training_allowed=false`, and `gpu_required_now=false`.
