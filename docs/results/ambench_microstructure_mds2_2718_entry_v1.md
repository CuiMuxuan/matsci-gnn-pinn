# AM-Bench mds2-2718 Microstructure Entry v1

## Context

Phase 17 moves from synthetic coordinate/RBF graph conditioning to real/semireal microstructure conditioning. The first real source is AM-Bench 2022 / AMB2022-03 optical microscopy:

- Dataset: `mds2-2718`
- DOI: https://doi.org/10.18434/mds2-2718
- Manifest: `configs/data/ambench_mds2_2718_sources.yaml`
- Server: A100-SXM4-40GB, `/root/matsci-gnn-pinn`
- Commit: `050dfe8`

## Commands

Dry-run:

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python -m gnnpinn.data.ambench_downloads \
  --dataset-id mds2-2718 \
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718 \
  --download \
  --dry-run \
  --output outputs/data_audits/ambench_mds2_2718_download_dry_run.json
```

Download and validate:

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python -m gnnpinn.data.ambench_downloads \
  --dataset-id mds2-2718 \
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718 \
  --download \
  --verify-sha256 \
  --output outputs/data_audits/ambench_mds2_2718_download_report.json
```

The first TIFF download stopped early at `33743629 / 36962776` bytes and was correctly rejected as incomplete. A single-file retry succeeded:

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python -m gnnpinn.data.ambench_downloads \
  --dataset-id mds2-2718 \
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718 \
  --download \
  --overwrite \
  --file-id single_track_cross_section_representative_tif \
  --verify-sha256 \
  --output outputs/data_audits/ambench_mds2_2718_tif_redownload_report.json
```

Microstructure inspection:

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python -m gnnpinn.data.loaders.ambench_microstructure \
  --image data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718/Single_Track_Cross_Sections/AMB2022-718-SH1-BP1-P2-L2.1-3_m.tif \
  --sample-id AMB2022-718-SH1-BP1-P2-L2.1-3_m \
  --threshold-quantile 0.9 \
  --grid-rows 8 \
  --grid-cols 8 \
  --graph-k 4 \
  --output outputs/data_audits/ambench_mds2_2718_micrograph_inspection.json
```

## Download Result

After the retry:

| Field | Value |
| --- | --- |
| `validation_failed` | `false` |
| `ready` | `true` |
| `missing_required` | `[]` |
| `mismatched_required` | `[]` |
| Representative TIFF bytes | `36962776` |

Downloaded first subset:

```text
2718_README.txt
AMB2022-718-SH1-MeltPool_Cross-Section_Measurement_Results.xlsx
Single_Track_Cross_Sections/AMB2022-718-SH1-BP1-P2-L2.1-3_m.tif
Single_Track_Cross_Sections/AMB2022-718-SH1-BP1-P2-L2.1-3_m.tif.sha256
```

## Inspection Result

Representative sample:

| Field | Value |
| --- | --- |
| sample id | `AMB2022-718-SH1-BP1-P2-L2.1-3_m` |
| parsed | `true` |
| build plate | `BP1` |
| process | `P2` |
| line | `L2.1` |
| replicate | `3` |
| masked | `true` |
| image shape | `[3008, 4096, 3]` |
| gray shape | `[3008, 4096]` |
| dtype | `uint8` |
| intensity mean | `45.471656` |
| intensity std | `24.575059` |
| q90 threshold | `64.1496` |
| mask fraction | `0.100182` |
| graph nodes | `64` |
| graph edges | `512` |

Node features:

```text
center_row_norm
center_col_norm
mean_intensity_norm
std_intensity_norm
mask_fraction
```

## Interpretation

This confirms a real AM-Bench optical microscopy file can be downloaded, verified, parsed into sample metadata, and converted into a coarse `MicrostructureGraph` artifact. The graph is intentionally simple: it is not yet a grain graph, but it creates a real-data bridge for replacing synthetic coordinate/RBF graph features in the closure branch.

## Follow-up Implementation

The next data bridge has been implemented after this first inspection:

- `gnnpinn.data.loaders.ambench_microstructure --mode aggregate` converts inspection JSON into graph feature JSONL/CSV.
- `RealMicroGraphFeatureProvider` reads the JSONL and exposes fixed `g0/g1/...` features.
- `gnnpinn.train.macro_pinn` supports `--closure-graph-mode real_micro`.

The next server action is to aggregate the first real inspection into `data/processed/.../micro_graph_features.jsonl`, then run a small real-micro closure smoke before expanding to more `P/L/replicate` images.
