# AM-Bench Dense Temperature Server Run v1

## Run Context

- Server: Ubuntu 22, NVIDIA A100-SXM4-40GB
- Repository commit at run start: `1d410b8`
- Environment: `gnnpinn`, Python 3.11, PyTorch 2.5.0+cu124
- Dataset: NIST AM-Bench 2022 / AMB2022-03 / `mds2-2716`
- Source HDF5: `Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5`
- Target: `temperature_C`
- Split: frame-based split manifest
- Download readiness: `True`; missing required files: `[]`; mismatched required files: `[]`

## Data Conversion

- Output table: `data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_dense_a100_sxm4_40gb_v1.csv`
- Valid points: 41054
- Source shape: [700, 640, 304]
- Sampling: frame_step=5, max_frames=120, row_step=2, max_rows=320, col_step=2, max_cols=152, min_signal=100

## Test Split Metrics

| Method | Test RMSE | Test MAE | Test Relative L2 |
| --- | ---: | ---: | ---: |
| Mean baseline | 108.387552 | 85.212557 | 0.091644 |
| Macro PINN data-only | 148.265776 | 102.813809 | 0.125361 |
| Macro PINN + PDE residual | 151.394768 | 106.453019 | 0.128007 |

## All-Point Training Output Metrics

These are the top-level metrics printed by `gnnpinn.train.macro_pinn`; they are useful for run sanity checks but are not the primary split-aware comparison.

| Method | RMSE | MAE | Relative L2 |
| --- | ---: | ---: | ---: |
| Macro PINN data-only | 117.397597 | 87.990419 | 0.099355 |
| Macro PINN + PDE residual | 117.998573 | 89.049259 | 0.099864 |

## Artifacts

- Data manifest: `outputs/data_audits/ambench_line_0_1_temperature_dense_a100_sxm4_40gb_v1_manifest.json`
- Split manifest: `outputs/data_splits/ambench_line_0_1_temperature_dense_a100_sxm4_40gb_v1_split.json`
- Baseline metrics: `outputs/baselines/ambench_line_0_1_temperature_dense_a100_sxm4_40gb_v1_mean_baseline.json`
- Data-only run: `outputs/runs/ambench_line_0_1_temperature_dense_a100_sxm4_40gb_macro_pinn_data_only_v1/`
- PDE run: `outputs/runs/ambench_line_0_1_temperature_dense_a100_sxm4_40gb_macro_pinn_pde_v1/`
- Environment freeze files saved inside each run directory: `conda_from_history.yml`, `pip_freeze.txt`, `env_report.txt`, `nvidia_smi.txt`

## Interpretation

This first dense server run confirms that the AM-Bench HDF5 download, SHA256 verification, calibrated temperature conversion, frame split, baseline evaluation, CUDA training path, and artifact capture work end to end on the A100 server.

The Macro PINN variants do not yet outperform the train-fitted mean baseline on the held-out frame split. This is still a useful result: it indicates that the current coordinate MLP and naive PDE residual are not extracting enough structure from the sampled thermal field. The next research step should focus on coordinate/time normalization, melt-pool-aware sampling, model capacity and optimizer tuning, physically calibrated PDE coefficients, and comparing against stronger interpolation baselines before claiming physical improvement.
