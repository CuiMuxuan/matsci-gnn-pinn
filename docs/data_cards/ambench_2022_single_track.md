# Data Card: AM-Bench 2022 Single-Track Starter Case

## Purpose

This data card registers the first real public-data direction for the executable prototype. It is intentionally conservative: Phase 0 records source pages and local readiness before any heavy modeling or large downloads.

## Source

- NIST AM-Bench: https://www.nist.gov/ambench
- Data management systems: https://www.nist.gov/ambench/am-bench-data-management-systems
- Benchmark test data: https://www.nist.gov/ambench/benchmark-test-data

Current source note: the NIST benchmark data page says some AM-Bench data links are temporarily de-activated while data management systems are upgraded. The 2018 AMB2018-02 individual laser trace challenge is the closest starter case for single-track melt-pool and cooling-rate experiments.

## Intended Role

| Project Component | Role |
|---|---|
| Phase 0 data audit | Register source, local root, expected modalities, and missing files. |
| Phase 1 macro PINN | Temperature/melt-pool field baseline when local data are available. |
| Direction 1 closure | Learn sparse corrections to heat/material closure terms. |
| Direction 3 coupling | Later connect thermal history to microstructure graph and feedback parameters. |

## Expected Modalities

- Process metadata: power, speed, scan path, layer thickness.
- Thermal observations: temperature, melt-pool geometry, thermal images, or equivalent derived fields.
- Microstructure observations: EBSD, SEM, grain statistics, or related benchmark artifacts.
- Mechanics observations: strain, stress, deflection, or mechanical test outputs when available.

## Local Layout

Expected local root:

```text
data/raw/ambench/2022_single_track/
```

The audit command scans this local root using the patterns in:

```text
configs/data/ambench_2022_single_track.yaml
```

## Readiness Gates

| Gate | Requirement |
|---|---|
| G0 source registration | Source URLs and config exist. |
| G1 local files | At least one local data file exists. |
| G2 modality match | At least one expected modality pattern matches. |
| G3 report artifact | Audit writes JSON and Markdown reports. |

## Current Status

Source registered. A mapping-based adapter is available for AM-Bench-like raw CSV point tables. Replace the example source path and column names after downloading or receiving the target AM-Bench files.

```bash
python -m gnnpinn.data.audit --config configs/data/ambench_2022_single_track.yaml
python -m gnnpinn.data.loaders.ambench --mapping configs/data/ambench_field_table_mapping.example.yaml
```
