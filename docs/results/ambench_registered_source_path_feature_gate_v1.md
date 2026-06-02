# AM-Bench Registered Source-Path Feature Gate v1

## Status

Phase 52 is closed as a data-incompatibility diagnostic for the current `Line_0_1` table.

The branch tested whether the AM-Bench scan-strategy XYPT file can provide physically registered source-path features for the Phase 51 dense `Line_0_1` validation table. It cannot be used for that table without an additional single-track scan-path source or a documented camera-pixel to galvo-mm registration.

## Implementation

New script:

```text
scripts/server/phase52_registered_source_path_probe.py
```

The script first checks compatibility before building features:

- inspect XYPT groups and power-on segments;
- inspect field-table `line_id`, `dataset_path`, and coordinate ranges;
- match `Xpad`/`Ypad` only to pad tables, not to `Line_*` single-track tables;
- reject coordinate systems that are not safely registered unless explicitly run in diagnostic rescale mode.

This prevents a pad scan strategy from being fitted to a single-track thermal table.

## A100 Validation

Server:

```text
root@223.109.239.30 -p 22036
GPU: NVIDIA A100-SXM4-40GB
```

Remote test:

```bash
PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 /home/vipuser/miniconda3/bin/conda run -n gnnpinn \
  python -X utf8 -m pytest -q tests/test_phase52_registered_source_path_probe.py --basetemp /tmp/p52pytest
```

Result:

```text
2 passed
```

Formal gate:

```bash
PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 /home/vipuser/miniconda3/bin/conda run -n gnnpinn \
  python -X utf8 scripts/server/phase52_registered_source_path_probe.py \
  --scan-strategy data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716/ScanStrategy/AMB2022-03-AMMT-718-Pad_XYPT.h5 \
  --table data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_phase51_dense.csv \
  --target temperature_C \
  --split-manifest outputs/data_splits/phase51_line0_1_temperature_dense_split.json \
  --json-output outputs/reports/phase52_line0_1_registered_source_path_summary.json
```

## Data Inspection

XYPT file:

```text
data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716/ScanStrategy/AMB2022-03-AMMT-718-Pad_XYPT.h5
```

The file contains pad scan strategy groups only:

| Group | Points | Power-on points | Power-on segments | X range | Y range |
| --- | ---: | ---: | ---: | --- | --- |
| `XYPT/Xpad` | 117,999 | 12,267 | 47 | `[-0.6520, 13.6520]` mm | `[28.2470, 33.2470]` mm |
| `XYPT/Ypad` | 25,499 | 12,528 | 24 | `[-1.6104, 2.9834]` mm | `[27.0940, 34.3990]` mm |

The target table is a single-track thermography table:

| Item | Value |
| --- | --- |
| Table | `line_0_1_temperature_phase51_dense.csv` |
| Rows | 10,205 |
| `line_id` | `Line_0_1` |
| `dataset_path` | `ThermalData/Line_0_1/Signal` |
| Coordinate system | camera pixel indices |

The AM-Bench README confirms that this XYPT file contains `Xpad` and `Ypad` scan strategies, while thermography `Line_X_Y_Z` groups correspond to individual laser tracks.

## Decision

The formal decision is negative:

```text
scan strategy file contains pad XYPT groups but table is a single-track Line_* dataset
```

Do not use `AMB2022-03-AMMT-718-Pad_XYPT.h5` as a registered source path for `ThermalData/Line_0_1/Signal`.

The next source-path branch requires one of:

- a single-track scan-path source aligned to `ThermalData/Line_*`;
- a documented mapping from pad XYPT coordinates to pad thermography tables, then run the gate on `X_pad*` or `Y_pad*`, not on `Line_0_1`;
- an explicit pivot away from source-path inversion toward the already stronger broad-data process-conditioned route guard.

Until one of those is available, broad12/broad21 source-inversion expansion remains blocked.
