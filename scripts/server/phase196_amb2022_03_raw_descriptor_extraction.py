#!/usr/bin/env python3
"""Extract the Phase 195-frozen raw thermography descriptors for AMB2022-03."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PHASE195 = Path(
    os.environ.get(
        "AMB2022_03_PHASE195_DESCRIPTOR_DESIGN",
        "/root/matsci-gnn-pinn-ops/phase195_amb2022_03_thermal_descriptor_design.json",
    )
)
DEFAULT_SIGNAL = Path(
    os.environ.get(
        "AMB2022_03_THERMOGRAPHY_SIGNAL",
        "/root/matsci-gnn-pinn-data/raw/ambench/2022_external_intake/AMB2022-03-thermography/"
        "Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5",
    )
)
EXPECTED_SIGNAL_SHAPE = (700, 640, 304)
EXPECTED_SIGNAL_DTYPE = "uint16"
EXPECTED_BIT_DEPTH = 12.0
RAW_THRESHOLD_DL = 100.0
EXPECTED_SIGNAL_UNITS = "digital levels"
EXPECTED_LINE_SIGNAL_GROUP_COUNT = 21
EXPECTED_NON_LINE_SIGNAL_GROUP_IDS = (
    "X_pad1",
    "X_pad2",
    "Y_pad1",
    "Y_pad1_SS",
    "Y_pad2",
    "Y_pad2_SS",
)
EXPECTED_NON_LINE_SIGNAL_GROUP_COUNT = len(EXPECTED_NON_LINE_SIGNAL_GROUP_IDS)
FRAME_CHUNK_SIZE = 16
RAW_QUANTILE = 0.99
HISTOGRAM_BINS = 1 << int(EXPECTED_BIT_DEPTH)
DESCRIPTOR_IDS = (
    "signal_mean_dl",
    "signal_std_dl",
    "signal_max_dl",
    "signal_p99_dl",
    "above_threshold_fraction",
    "active_frame_fraction",
    "frame_max_mean_dl",
    "frame_max_std_dl",
)
CSV_FIELDS = ("thermal_group_id", *DESCRIPTOR_IDS)


def _h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("h5py is required for Phase 196") from exc
    return h5py


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scalar_attribute(item: Any, key: str) -> float:
    values = np.asarray(item.attrs[key]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Expected scalar attribute {key}, got shape {values.shape}")
    return float(values[0])


def _signal_schema(signal: Any) -> dict[str, Any]:
    return {
        "signal_shape": [int(value) for value in signal.shape],
        "signal_dtype": str(signal.dtype),
        "bit_depth": _scalar_attribute(signal, "bit_depth"),
        "threshold_level": _scalar_attribute(signal, "threshold_level"),
        "signal_units": str(signal.attrs.get("units")),
    }


def _schema_blockers(schema: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if tuple(schema.get("signal_shape", ())) != EXPECTED_SIGNAL_SHAPE:
        blockers.append("raw_signal_shape_mismatch")
    if schema.get("signal_dtype") != EXPECTED_SIGNAL_DTYPE:
        blockers.append("raw_signal_dtype_not_uint16")
    if float(schema.get("bit_depth", 0.0)) != EXPECTED_BIT_DEPTH:
        blockers.append("raw_signal_bit_depth_not_12")
    if float(schema.get("threshold_level", 0.0)) != RAW_THRESHOLD_DL:
        blockers.append("raw_signal_threshold_not_100")
    if schema.get("signal_units") != EXPECTED_SIGNAL_UNITS:
        blockers.append("raw_signal_units_not_digital_levels")
    return blockers


class _StreamingMoments:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values: np.ndarray) -> None:
        flattened = np.asarray(values, dtype=np.float64).reshape(-1)
        chunk_count = int(flattened.size)
        if not chunk_count:
            return
        chunk_mean = float(np.mean(flattened, dtype=np.float64))
        centered = flattened - chunk_mean
        chunk_m2 = float(np.dot(centered, centered))
        if not self.count:
            self.count = chunk_count
            self.mean = chunk_mean
            self.m2 = chunk_m2
            return
        total_count = self.count + chunk_count
        delta = chunk_mean - self.mean
        self.m2 += chunk_m2 + delta * delta * self.count * chunk_count / total_count
        self.mean += delta * chunk_count / total_count
        self.count = total_count

    @property
    def population_std(self) -> float:
        if not self.count:
            raise ValueError("Cannot calculate moments for an empty signal")
        return math.sqrt(max(self.m2 / self.count, 0.0))


def _histogram_value_at_rank(histogram: np.ndarray, rank: int) -> int:
    total = int(histogram.sum(dtype=np.uint64))
    if rank < 0 or rank >= total:
        raise ValueError(f"Histogram rank {rank} is outside [0, {total})")
    return int(np.searchsorted(np.cumsum(histogram, dtype=np.uint64), rank, side="right"))


def _linear_quantile_from_histogram(histogram: np.ndarray, quantile: float) -> float:
    sample_count = int(histogram.sum(dtype=np.uint64))
    if sample_count <= 0:
        raise ValueError("Cannot calculate a quantile for an empty signal")
    position = (sample_count - 1) * quantile
    lower_rank = math.floor(position)
    upper_rank = math.ceil(position)
    lower_value = _histogram_value_at_rank(histogram, lower_rank)
    upper_value = _histogram_value_at_rank(histogram, upper_rank)
    return float(lower_value + (position - lower_rank) * (upper_value - lower_value))


def describe_raw_signal(signal: Any, *, threshold_level: float) -> dict[str, float]:
    """Stream one uint16 Signal dataset and return the Phase 195-frozen descriptors."""

    if len(signal.shape) != 3 or any(int(size) <= 0 for size in signal.shape):
        raise ValueError(f"Expected a non-empty three-dimensional Signal dataset, got {signal.shape}")
    if str(signal.dtype) != EXPECTED_SIGNAL_DTYPE:
        raise ValueError(f"Expected {EXPECTED_SIGNAL_DTYPE} Signal values, got {signal.dtype}")
    if threshold_level != RAW_THRESHOLD_DL:
        raise ValueError(f"Expected fixed raw threshold {RAW_THRESHOLD_DL:g}, got {threshold_level:g}")

    signal_moments = _StreamingMoments()
    frame_max_moments = _StreamingMoments()
    histogram = np.zeros(HISTOGRAM_BINS, dtype=np.uint64)
    total_samples = 0
    samples_above_threshold = 0
    active_frames = 0
    maximum_value = 0
    frame_count = int(signal.shape[0])

    for frame_start in range(0, frame_count, FRAME_CHUNK_SIZE):
        frame_stop = min(frame_start + FRAME_CHUNK_SIZE, frame_count)
        block = np.asarray(signal[frame_start:frame_stop])
        if block.dtype != np.uint16:
            raise ValueError(f"Signal dtype changed while reading: {block.dtype}")
        if int(block.max()) >= HISTOGRAM_BINS:
            raise ValueError("Raw Signal contains values above the declared 12-bit range")
        flat = block.reshape(-1)
        frame_maximums = block.max(axis=(1, 2))
        signal_moments.update(flat)
        frame_max_moments.update(frame_maximums)
        histogram += np.bincount(flat, minlength=HISTOGRAM_BINS).astype(np.uint64, copy=False)
        total_samples += int(flat.size)
        samples_above_threshold += int(np.count_nonzero(flat >= threshold_level))
        active_frames += int(np.count_nonzero(frame_maximums >= threshold_level))
        maximum_value = max(maximum_value, int(frame_maximums.max()))

    if signal_moments.count != total_samples or frame_max_moments.count != frame_count:
        raise AssertionError("Streaming moment counts do not match the Signal dataset shape")
    return {
        "signal_mean_dl": signal_moments.mean,
        "signal_std_dl": signal_moments.population_std,
        "signal_max_dl": float(maximum_value),
        "signal_p99_dl": _linear_quantile_from_histogram(histogram, RAW_QUANTILE),
        "above_threshold_fraction": samples_above_threshold / total_samples,
        "active_frame_fraction": active_frames / frame_count,
        "frame_max_mean_dl": frame_max_moments.mean,
        "frame_max_std_dl": frame_max_moments.population_std,
    }


def _phase195_ready(phase195: dict[str, Any]) -> bool:
    gate = phase195.get("gate", {})
    return (
        gate.get("status") == "phase195_thermal_descriptor_design_ready_phase196_descriptor_extraction"
        and bool(gate.get("phase196_descriptor_extraction_allowed"))
        and gate.get("calibrated_temperature_descriptor_allowed") is False
        and gate.get("calibration_fitting_allowed") is False
        and gate.get("model_training_allowed") is False
    )


def extract_descriptor_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read only raw Line_* Signal arrays and their metadata, one dataset at a time."""

    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        line_paths: dict[str, str] = {}
        non_line_group_ids: list[str] = []

        def visitor(name: str, item: Any) -> None:
            if not isinstance(item, h5py.Dataset) or Path(name).name != "Signal":
                return
            group_id = Path(name).parent.name
            if group_id.startswith("Line_"):
                if group_id in line_paths:
                    raise ValueError(f"Duplicate Line_* Signal group: {group_id}")
                line_paths[group_id] = name
            else:
                non_line_group_ids.append(group_id)

        handle.visititems(visitor)
        rows: list[dict[str, Any]] = []
        schema_by_group: dict[str, dict[str, Any]] = {}
        schema_mismatch_ids: list[str] = []
        for group_id, signal_path in sorted(line_paths.items()):
            signal = handle[signal_path]
            schema = _signal_schema(signal)
            schema_by_group[group_id] = schema
            if _schema_blockers(schema):
                schema_mismatch_ids.append(group_id)

        if schema_mismatch_ids:
            raise ValueError(f"Phase 196 raw-signal schema mismatch: {sorted(schema_mismatch_ids)}")
        for group_id, signal_path in sorted(line_paths.items()):
            signal = handle[signal_path]
            rows.append(
                {
                    "thermal_group_id": group_id,
                    **describe_raw_signal(signal, threshold_level=RAW_THRESHOLD_DL),
                }
            )

    unique_shapes = sorted({tuple(schema["signal_shape"]) for schema in schema_by_group.values()})
    audit = {
        "source_hdf5_name": path.name,
        "line_signal_group_ids": sorted(line_paths),
        "line_signal_group_count": len(line_paths),
        "non_line_signal_group_ids": sorted(non_line_group_ids),
        "non_line_signal_group_count": len(non_line_group_ids),
        "unique_signal_shapes": [list(shape) for shape in unique_shapes],
        "schema_mismatch_ids": sorted(schema_mismatch_ids),
        "frame_chunk_size": FRAME_CHUNK_SIZE,
        "cross_section_targets_read": False,
        "thermalcal_read": False,
        "streaming_policy": "One raw Signal dataset is processed in frame chunks; no full HDF5 collection is materialized.",
    }
    return rows, audit


def build_gate(
    phase195: dict[str, Any], descriptor_rows: list[dict[str, Any]], extraction_audit: dict[str, Any]
) -> dict[str, Any]:
    blockers: list[str] = []
    if not _phase195_ready(phase195):
        blockers.append("phase195_descriptor_design_gate_not_ready")
    if extraction_audit.get("cross_section_targets_read") is not False:
        blockers.append("cross_section_target_boundary_broken")
    if extraction_audit.get("thermalcal_read") is not False:
        blockers.append("thermalcal_conversion_boundary_broken")
    if int(extraction_audit.get("line_signal_group_count", 0)) != EXPECTED_LINE_SIGNAL_GROUP_COUNT:
        blockers.append("unexpected_line_signal_group_count")
    if int(extraction_audit.get("non_line_signal_group_count", 0)) != EXPECTED_NON_LINE_SIGNAL_GROUP_COUNT:
        blockers.append("unexpected_non_line_signal_group_count")
    if sorted(extraction_audit.get("non_line_signal_group_ids", [])) != list(EXPECTED_NON_LINE_SIGNAL_GROUP_IDS):
        blockers.append("unexpected_non_line_signal_group_ids")
    if extraction_audit.get("unique_signal_shapes") != [list(EXPECTED_SIGNAL_SHAPE)]:
        blockers.append("raw_signal_shape_contract_broken")
    if extraction_audit.get("schema_mismatch_ids"):
        blockers.append("raw_signal_schema_contract_broken")
    if len(descriptor_rows) != EXPECTED_LINE_SIGNAL_GROUP_COUNT:
        blockers.append("descriptor_row_count_mismatch")
    if len({str(row.get("thermal_group_id", "")) for row in descriptor_rows}) != len(descriptor_rows):
        blockers.append("duplicate_thermal_group_id")
    for row in descriptor_rows:
        if tuple(row.keys()) != CSV_FIELDS:
            blockers.append("descriptor_field_contract_broken")
            break
        if not str(row["thermal_group_id"]).startswith("Line_"):
            blockers.append("descriptor_row_not_single_track_line")
            break
        if not all(math.isfinite(float(row[field])) for field in DESCRIPTOR_IDS):
            blockers.append("nonfinite_raw_descriptor")
            break
    blockers = sorted(set(blockers))
    ready = not blockers
    return {
        "status": (
            "phase196_raw_descriptor_extraction_ready_phase197_calibration_table_design"
            if ready
            else "phase196_raw_descriptor_extraction_incomplete_or_boundary_broken"
        ),
        "phase197_calibration_table_design_allowed": ready,
        "calibrated_temperature_descriptor_allowed": False,
        "calibration_fitting_allowed": False,
        "model_training_allowed": False,
        "hyperparameter_search_allowed": False,
        "post_b8_model_reselection_allowed": False,
        "descriptor_target_leakage_allowed": False,
        "simulation_as_external_validation_allowed": False,
        "blocking_audits": blockers,
        "next_action": (
            "join the frozen descriptor table to the pre-existing Phase 193 identifier table, then audit the fixed grouped folds without fitting"
            if ready
            else "repair the raw-signal extraction boundary or schema before any target join"
        ),
    }


def build_payload(
    phase195: dict[str, Any], descriptor_rows: list[dict[str, Any]], extraction_audit: dict[str, Any]
) -> dict[str, Any]:
    return {
        "phase": 196,
        "objective": "fixed_label_free_raw_thermography_descriptor_extraction",
        "descriptor_ids": list(DESCRIPTOR_IDS),
        "raw_quantile_policy": "Exact 0.99 quantile using the NumPy linear quantile convention over the raw 12-bit histogram.",
        "feature_boundary": (
            "Only Line_* raw Signal arrays and their raw-signal attributes are read. Cross-section labels and ThermalCal "
            "temperature conversion are excluded from this phase."
        ),
        "condition_descriptors": descriptor_rows,
        "extraction_audit": extraction_audit,
        "gate": build_gate(phase195, descriptor_rows, extraction_audit),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase195", type=Path, default=DEFAULT_PHASE195)
    parser.add_argument("--signal", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--descriptors-csv", type=Path, required=True)
    args = parser.parse_args()
    descriptor_rows, extraction_audit = extract_descriptor_rows(args.signal)
    payload = build_payload(_read_json(args.phase195), descriptor_rows, extraction_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.descriptors_csv, descriptor_rows)
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
