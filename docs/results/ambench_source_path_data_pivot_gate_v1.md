# AM-Bench Source-Path Data Pivot Gate v1

## Status

Phase 53 is closed as a negative source-path data gate.

The current AMB2022-03 / `mds2-2716` bundle does not provide a paper-facing registered source path for the single-track `ThermalData/Line_*` tables. It does provide pad thermography tables and pad XYPT scan commands, but the HDF5 files do not contain camera-pixel to galvo-mm registration metadata. Therefore, source-inversion broad12/broad21 validation remains blocked.

## Implementation

New scripts:

```text
scripts/server/phase53_source_path_data_pivot_gate.py
scripts/server/run_phase53_source_path_pivot_gate_a100.sh
```

Updated guard:

```text
scripts/server/phase52_registered_source_path_probe.py
```

The Phase 52 coordinate guard now checks both range overlap and span ratio, so camera-pixel ranges such as `0..300` / `0..620` are not treated as safely comparable to galvo-mm XYPT ranges.

## A100 Validation

Server:

```text
root@223.109.239.30 -p 22036
GPU: NVIDIA A100-SXM4-40GB
```

Command:

```bash
cd /root/matsci-gnn-pinn
bash scripts/server/run_phase53_source_path_pivot_gate_a100.sh
```

Targeted server tests:

```text
tests/test_phase52_registered_source_path_probe.py
tests/test_phase53_source_path_data_pivot_gate.py
```

Result:

```text
5 passed
```

## Data Inventory

Thermography:

```text
data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716/Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5
```

Scan strategy:

```text
data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716/ScanStrategy/AMB2022-03-AMMT-718-Pad_XYPT.h5
```

Inventory:

| Item | Value |
| --- | --- |
| Thermography groups | 27 |
| Single-track groups | 21 `Line_*` groups |
| Pad thermography groups | `X_pad1`, `X_pad2`, `Y_pad1`, `Y_pad1_SS`, `Y_pad2`, `Y_pad2_SS` |
| Scan-strategy HDF5 files | `AMB2022-03-AMMT-718-Pad_XYPT.h5` |
| XYPT groups | `XYPT/Xpad`, `XYPT/Ypad` |
| Single-track scan-path groups | none |
| HDF5 registration metadata keys | none found |

Formal inventory decision:

```text
negative: scan strategy exposes pad XYPT only; thermography has pad tables, but no HDF5 camera-pixel to galvo-mm registration metadata was found
```

## Pad Rescale Diagnostic

Because pad thermography exists, Phase 53 ran a diagnostic-only rescale probe on `X_pad1` and `Y_pad1`. This is not a paper-facing registration because it independently rescales camera pixels and galvo millimeters.

Sampling:

| Pad | Source shape | Rows | Sampling |
| --- | ---: | ---: | --- |
| `X_pad1` | `40001 x 640 x 304` | 101 | 120 frames, 32 rows, 31 cols, balanced hot/gradient cap 64 per frame |
| `Y_pad1` | `10001 x 640 x 304` | 372 | 120 frames, 32 rows, 31 cols, balanced hot/gradient cap 64 per frame |

Metrics:

| Pad | Feature set | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Coverage90 |
| --- | --- | ---: | ---: | ---: | ---: |
| `X_pad1` | base | 127.723559 | 190.914071 | 178.656473 | 1.000000 |
| `X_pad1` | registered source path, rescale diagnostic | 157.298079 | 170.275221 | 170.508595 | 1.000000 |
| `Y_pad1` | base | 153.859838 | 218.406789 | 183.031892 | 0.966102 |
| `Y_pad1` | registered source path, rescale diagnostic | 221.030923 | 241.662999 | 203.527790 | 0.796610 |

Decision:

- `X_pad1` improves hot/gradient but worsens global RMSE, so it fails the combined gate.
- `Y_pad1` worsens global, hot, gradient, and coverage, so it fails the combined gate.
- Both pad probes have `coordinate.compatible=false` and `span_ratio_compatible=false`; they are diagnostic-only.

## Artifacts

```text
outputs/reports/phase53_source_path_data_pivot_summary.json
outputs/reports/phase53_source_path_pivot_gate_rollup.json
outputs/reports/phase53_x_pad1_registered_source_path_rescale_diagnostic_summary.json
outputs/reports/phase53_y_pad1_registered_source_path_rescale_diagnostic_summary.json
outputs/reports/phase53_x_pad1_temperature_diag_manifest.json
outputs/reports/phase53_y_pad1_temperature_diag_manifest.json
```

Server-side generated tables:

```text
data/interim/ambench/2022_single_track/AMB2022-03/x_pad1_temperature_phase53_diag.csv
data/interim/ambench/2022_single_track/AMB2022-03/y_pad1_temperature_phase53_diag.csv
```

## Decision

Do not run broad12/broad21 source-inversion validation for the current source-path family.

The source-inversion branch can resume only if one of the following becomes available:

- a single-track scan-path source aligned to `ThermalData/Line_*`;
- a documented camera-pixel to galvo-mm registration for pad thermography;
- a different benchmark target where scan path, thermography table, coordinates, and timing are aligned.

Until then, the paper-facing route should pivot back to the broad-data process-conditioned route guard / process-axis selector evidence, where the project already has stronger broad12/broad21 artifacts and strong-baseline comparisons.
