#!/usr/bin/env python3
"""Freeze label-free raw-thermography descriptors before AMB2022-03 calibration fitting."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PHASE194 = Path(
    os.environ.get(
        "AMB2022_01_PHASE194_CALIBRATION_PROTOCOL",
        "/root/matsci-gnn-pinn-ops/phase194_amb2022_03_calibration_protocol.json",
    )
)
DEFAULT_SIGNAL = Path(
    os.environ.get(
        "AMB2022_03_THERMOGRAPHY_SIGNAL",
        "/root/matsci-gnn-pinn-data/raw/ambench/2022_external_intake/AMB2022-03-thermography/"
        "Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5",
    )
)
DESCRIPTOR_ROWS = (
    {
        "descriptor_id": "signal_mean_dl",
        "definition": "Mean raw digital level over all frames and pixels.",
    },
    {
        "descriptor_id": "signal_std_dl",
        "definition": "Standard deviation of raw digital level over all frames and pixels.",
    },
    {
        "descriptor_id": "signal_max_dl",
        "definition": "Maximum raw digital level over all frames and pixels.",
    },
    {
        "descriptor_id": "signal_p99_dl",
        "definition": "Fixed 0.99 raw-signal quantile over all frames and pixels.",
    },
    {
        "descriptor_id": "above_threshold_fraction",
        "definition": "Fraction of raw samples at or above the HDF5 threshold_level attribute.",
    },
    {
        "descriptor_id": "active_frame_fraction",
        "definition": "Fraction of frames containing at least one sample at or above threshold_level.",
    },
    {
        "descriptor_id": "frame_max_mean_dl",
        "definition": "Mean of the per-frame maximum raw digital level.",
    },
    {
        "descriptor_id": "frame_max_std_dl",
        "definition": "Standard deviation of the per-frame maximum raw digital level.",
    },
)
DESCRIPTOR_FIELDS = ("descriptor_id", "definition", "uses_target", "uses_temperature_conversion")


def _h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("h5py is required for Phase 195") from exc
    return h5py


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scalar_attribute(item: Any, key: str) -> float:
    values = np.asarray(item.attrs[key]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Expected scalar attribute {key}, got shape {values.shape}")
    return float(values[0])


def inspect_signal_schema(path: Path) -> dict[str, Any]:
    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        first_signal: Any | None = None
        first_group: Any | None = None

        def visitor(name: str, item: Any) -> None:
            nonlocal first_signal, first_group
            if first_signal is None and isinstance(item, h5py.Dataset) and Path(name).name == "Signal":
                first_signal = item
                first_group = handle[name.rsplit("/", 1)[0]]

        handle.visititems(visitor)
        if first_signal is None or first_group is None:
            raise ValueError("No Signal dataset found in AMB2022-03 thermography HDF5")
        calibration = handle["Calibration/ThermalCal"]
        return {
            "signal_shape": [int(value) for value in first_signal.shape],
            "signal_dtype": str(first_signal.dtype),
            "bit_depth": _scalar_attribute(first_signal, "bit_depth"),
            "threshold_level": _scalar_attribute(first_signal, "threshold_level"),
            "threshold_zeros": str(first_signal.attrs.get("threshold_zeros")),
            "signal_units": str(first_signal.attrs.get("units")),
            "calibration_method": str(calibration.attrs.get("Cal_Method")),
            "calibration_model_text": str(calibration.attrs.get("Model")),
            "calibration_input": str(calibration.attrs.get("Model_input")),
            "calibration_output": str(calibration.attrs.get("Model_output")),
            "group_process_attributes": sorted(first_group.attrs.keys()),
        }


def _phase194_ready(phase194: dict[str, Any]) -> bool:
    gate = phase194.get("gate", {})
    return (
        gate.get("status") == "phase194_calibration_protocol_design_ready_phase195_thermal_descriptor_extraction_design"
        and bool(gate.get("phase195_thermal_descriptor_extraction_design_allowed"))
        and gate.get("calibration_fitting_allowed") is False
        and gate.get("model_training_allowed") is False
    )


def build_gate(phase194: dict[str, Any], signal_schema: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if not _phase194_ready(phase194):
        blockers.append("phase194_calibration_protocol_gate_not_ready")
    if len(signal_schema.get("signal_shape", [])) != 3:
        blockers.append("raw_signal_not_three_dimensional")
    if signal_schema.get("signal_dtype") != "uint16":
        blockers.append("raw_signal_dtype_not_uint16")
    if float(signal_schema.get("bit_depth", 0.0)) != 12.0:
        blockers.append("raw_signal_bit_depth_not_12")
    if float(signal_schema.get("threshold_level", 0.0)) != 100.0:
        blockers.append("raw_signal_threshold_not_100")
    if signal_schema.get("signal_units") != "digital levels":
        blockers.append("raw_signal_units_not_digital_levels")
    blockers = sorted(set(blockers))
    ready = not blockers
    return {
        "status": (
            "phase195_thermal_descriptor_design_ready_phase196_descriptor_extraction"
            if ready
            else "phase195_thermal_descriptor_design_incomplete_or_schema_mismatch"
        ),
        "phase196_descriptor_extraction_allowed": ready,
        "calibrated_temperature_descriptor_allowed": False,
        "calibration_fitting_allowed": False,
        "model_training_allowed": False,
        "hyperparameter_search_allowed": False,
        "post_b8_model_reselection_allowed": False,
        "descriptor_target_leakage_allowed": False,
        "simulation_as_external_validation_allowed": False,
        "blocking_audits": blockers,
        "next_action": (
            "extract the fixed raw-signal descriptors one condition at a time without reading cross-section targets"
            if ready
            else "repair raw-signal schema interpretation before descriptor extraction"
        ),
    }


def build_design(phase194: dict[str, Any], signal_schema: dict[str, Any]) -> dict[str, Any]:
    rows = [
        {
            **row,
            "uses_target": False,
            "uses_temperature_conversion": False,
        }
        for row in DESCRIPTOR_ROWS
    ]
    return {
        "phase": 195,
        "objective": "fixed_label_free_raw_thermography_descriptor_design",
        "signal_schema": signal_schema,
        "memory_policy": "Process exactly one Signal dataset at a time; do not materialize the full HDF5 collection.",
        "feature_normalization_policy": "Any normalization is fit inside each leave-one-process-setting-out training fold only.",
        "temperature_conversion_boundary": (
            "ThermalCal metadata is retained for a later formula audit; no calibrated-temperature descriptor is permitted in this phase."
        ),
        "descriptors": rows,
        "gate": build_gate(phase194, signal_schema),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DESCRIPTOR_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase194", type=Path, default=DEFAULT_PHASE194)
    parser.add_argument("--signal", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--descriptors-csv", type=Path, required=True)
    args = parser.parse_args()
    payload = build_design(_read_json(args.phase194), inspect_signal_schema(args.signal))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.descriptors_csv, payload["descriptors"])
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
