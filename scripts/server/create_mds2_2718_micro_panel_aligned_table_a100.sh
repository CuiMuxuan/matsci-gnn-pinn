#!/usr/bin/env bash
set -euo pipefail

cd /root/matsci-gnn-pinn
mkdir -p outputs/data_audits data/interim/ambench/2022_single_track/AMB2022-03

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
SOURCE_TABLE="${SOURCE_TABLE:-data/interim/ambench/2022_single_track/AMB2022-03/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1.csv}"
MICRO_FEATURES="${MICRO_FEATURES:-data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_panel.jsonl}"
RUN_ID="${RUN_ID:-ambench_line_0_1_temperature_hot_gradient_mds2_2718_micro_panel_framecycle_v1}"
OUTPUT_TABLE="${OUTPUT_TABLE:-data/interim/ambench/2022_single_track/AMB2022-03/${RUN_ID}.csv}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-outputs/data_audits/${RUN_ID}_manifest.json}"
FRAME_COLUMN="${FRAME_COLUMN:-frame_index}"
MAPPING_MODE="${MAPPING_MODE:-frame_cycle}"
CONSTANT_SAMPLE_ID="${CONSTANT_SAMPLE_ID:-}"

SOURCE_TABLE="$SOURCE_TABLE" \
MICRO_FEATURES="$MICRO_FEATURES" \
OUTPUT_TABLE="$OUTPUT_TABLE" \
OUTPUT_MANIFEST="$OUTPUT_MANIFEST" \
FRAME_COLUMN="$FRAME_COLUMN" \
MAPPING_MODE="$MAPPING_MODE" \
CONSTANT_SAMPLE_ID="$CONSTANT_SAMPLE_ID" \
"$CONDA_BIN" run -n "$CONDA_ENV" python - <<'PY'
import csv
import json
import os
from collections import Counter
from pathlib import Path


source_table = Path(os.environ["SOURCE_TABLE"])
micro_features = Path(os.environ["MICRO_FEATURES"])
output_table = Path(os.environ["OUTPUT_TABLE"])
output_manifest = Path(os.environ["OUTPUT_MANIFEST"])
frame_column = os.environ["FRAME_COLUMN"]
mapping_mode = os.environ["MAPPING_MODE"]
constant_sample_id = os.environ["CONSTANT_SAMPLE_ID"]

records = []
with micro_features.open(encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            records.append(json.loads(line))

sample_ids = [str(record["sample_id"]) for record in records]
if not sample_ids:
    raise ValueError(f"No sample records found in {micro_features}")

with source_table.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        raise ValueError(f"{source_table} has no header")
    source_fieldnames = list(reader.fieldnames)
    rows = list(reader)

if not rows:
    raise ValueError(f"{source_table} has no rows")

fieldnames = [name for name in source_fieldnames if name != "micro_sample_id"]
fieldnames.append("micro_sample_id")

if mapping_mode == "frame_cycle":
    if frame_column not in source_fieldnames:
        raise ValueError(f"{source_table} has no frame column {frame_column!r}")
    unique_frames = sorted({int(float(row[frame_column])) for row in rows})
    frame_to_sample = {
        frame: sample_ids[index % len(sample_ids)]
        for index, frame in enumerate(unique_frames)
    }

    def select_sample(row):
        return frame_to_sample[int(float(row[frame_column]))]

elif mapping_mode == "constant":
    selected = constant_sample_id or sample_ids[0]
    if selected not in sample_ids:
        raise ValueError(f"CONSTANT_SAMPLE_ID {selected!r} is not present in {micro_features}")
    unique_frames = sorted({int(float(row[frame_column])) for row in rows}) if frame_column in source_fieldnames else []
    frame_to_sample = {frame: selected for frame in unique_frames}

    def select_sample(row):
        return selected

else:
    raise ValueError(f"Unsupported MAPPING_MODE={mapping_mode!r}")

output_table.parent.mkdir(parents=True, exist_ok=True)
counts = Counter()
with output_table.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        out = {name: row.get(name, "") for name in fieldnames if name != "micro_sample_id"}
        sample_id = select_sample(row)
        out["micro_sample_id"] = sample_id
        counts[sample_id] += 1
        writer.writerow(out)

manifest = {
    "source_table": str(source_table),
    "output_table": str(output_table),
    "micro_features": str(micro_features),
    "mapping_mode": mapping_mode,
    "frame_column": frame_column,
    "n_rows": len(rows),
    "n_micro_records": len(sample_ids),
    "sample_ids": sample_ids,
    "sample_counts": dict(counts),
    "n_unique_frames": len(unique_frames),
    "frame_mapping_preview": [
        {"frame": frame, "micro_sample_id": frame_to_sample[frame]}
        for frame in unique_frames[:18]
    ],
    "note": (
        "Prototype alignment table. frame_cycle assigns sorted thermal frames "
        "to panel microstructure records cyclically; it exercises per-row "
        "real_micro selection but is not a physical process-ground-truth mapping."
    ),
}
output_manifest.parent.mkdir(parents=True, exist_ok=True)
output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Wrote: {output_table}")
print(f"Wrote: {output_manifest}")
print(json.dumps({k: manifest[k] for k in ["n_rows", "n_micro_records", "n_unique_frames", "sample_counts"]}, indent=2))
PY
