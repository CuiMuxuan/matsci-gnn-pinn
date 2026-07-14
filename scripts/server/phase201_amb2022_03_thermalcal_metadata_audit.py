#!/usr/bin/env python3
"""Audit AMB2022-03 ThermalCal metadata before any temperature conversion is considered."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PHASE200 = Path(
    os.environ.get(
        "AMB2022_03_PHASE200_RESIDUAL_AUDIT",
        "/root/matsci-gnn-pinn-ops/phase200_amb2022_03_residual_audit.json",
    )
)
DEFAULT_SIGNAL = Path(
    os.environ.get(
        "AMB2022_03_THERMOGRAPHY_SIGNAL",
        "/root/matsci-gnn-pinn-data/raw/ambench/2022_external_intake/AMB2022-03-thermography/"
        "Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5",
    )
)
MAX_CALIBRATION_COEFFICIENT_COUNT = 64
REQUIRED_THERMALCAL_ATTRIBUTES = ("Cal_Method", "Model", "Model_input", "Model_output")


def _h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("h5py is required for Phase 201") from exc
    return h5py


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _attribute_summary(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    if array.size > MAX_CALIBRATION_COEFFICIENT_COUNT:
        return {
            "dtype": str(array.dtype),
            "shape": [int(size) for size in array.shape],
            "values": None,
        }
    if array.dtype.kind in {"S", "U", "O"}:
        values: list[Any] = [_decode_scalar(item) for item in array.reshape(-1)]
    else:
        values = [float(item) for item in array.reshape(-1)]
    return {
        "dtype": str(array.dtype),
        "shape": [int(size) for size in array.shape],
        "values": values,
    }


def _scalar_text(attributes: dict[str, Any], key: str) -> str | None:
    summary = attributes.get(key, {})
    values = summary.get("values") if isinstance(summary, dict) else None
    if not isinstance(values, list) or len(values) != 1:
        return None
    return str(values[0])


def _coefficient_dataset_summary(dataset: Any) -> dict[str, Any]:
    element_count = int(np.prod(dataset.shape, dtype=np.int64))
    numeric = np.dtype(dataset.dtype).kind in {"i", "u", "f"}
    values = (
        [float(value) for value in np.asarray(dataset[...]).reshape(-1)]
        if numeric and element_count <= MAX_CALIBRATION_COEFFICIENT_COUNT
        else None
    )
    return {
        "dataset_path": str(dataset.name),
        "dtype": str(dataset.dtype),
        "shape": [int(size) for size in dataset.shape],
        "element_count": element_count,
        "numeric": numeric,
        "values": values,
    }


def inspect_thermalcal(path: Path) -> dict[str, Any]:
    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        calibration = handle["Calibration/ThermalCal"]
        attributes = {str(key): _attribute_summary(value) for key, value in calibration.attrs.items()}
        coefficient_datasets: list[dict[str, Any]] = []
        if isinstance(calibration, h5py.Dataset):
            object_kind = "dataset"
            coefficient_datasets.append(_coefficient_dataset_summary(calibration))
        elif isinstance(calibration, h5py.Group):
            object_kind = "group"

            def calibration_visitor(name: str, item: Any) -> None:
                if isinstance(item, h5py.Dataset):
                    coefficient_datasets.append(_coefficient_dataset_summary(item))

            calibration.visititems(calibration_visitor)
        else:  # pragma: no cover
            raise TypeError(f"Unsupported ThermalCal HDF5 object: {type(calibration)!r}")
        dataset_coefficient_values = [
            value
            for dataset in coefficient_datasets
            for value in (dataset["values"] or [])
        ]
        attribute_coefficients = {
            key: float(summary["values"][0])
            for key, summary in sorted(attributes.items())
            if key.startswith("Coeff_")
            and isinstance(summary.get("values"), list)
            and len(summary["values"]) == 1
            and isinstance(summary["values"][0], (float, int))
        }
        coefficient_values = [*attribute_coefficients.values(), *dataset_coefficient_values]
        signal_attributes: dict[str, dict[str, Any]] | None = None

        def visitor(name: str, item: Any) -> None:
            nonlocal signal_attributes
            if signal_attributes is None and isinstance(item, h5py.Dataset) and Path(name).name == "Signal":
                signal_attributes = {str(key): _attribute_summary(value) for key, value in item.attrs.items()}

        handle.visititems(visitor)
    return {
        "thermalcal_dataset_path": "Calibration/ThermalCal",
        "thermalcal_object_kind": object_kind,
        "thermalcal_coefficient_datasets": coefficient_datasets,
        "thermalcal_attribute_coefficients": attribute_coefficients,
        "thermalcal_coefficient_values": coefficient_values,
        "thermalcal_attributes": attributes,
        "raw_signal_attributes": signal_attributes or {},
        "raw_signal_arrays_read": False,
        "cross_section_targets_read": False,
        "calibration_fitting_performed": False,
    }


def _phase200_ready(phase200: dict[str, Any]) -> bool:
    gate = phase200.get("gate", {})
    return (
        gate.get("status")
        in {
            "phase200_residual_audit_complete_no_robust_additive_thermal_signal",
            "phase200_residual_audit_complete_additive_signal_not_escalated",
        }
        and bool(gate.get("phase201_mechanistic_stress_test_design_allowed"))
        and gate.get("model_training_allowed") is False
    )


def build_gate(phase200: dict[str, Any], intake: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if not _phase200_ready(phase200):
        blockers.append("phase200_residual_audit_gate_not_ready")
    attributes = intake.get("thermalcal_attributes", {})
    missing_attributes = [key for key in REQUIRED_THERMALCAL_ATTRIBUTES if key not in attributes]
    if missing_attributes:
        blockers.append("thermalcal_required_metadata_missing")
    coefficient_values = intake.get("thermalcal_coefficient_values")
    if not isinstance(coefficient_values, list) or not coefficient_values:
        blockers.append("thermalcal_coefficients_not_small_finite_vector")
    elif not all(np.isfinite(float(value)) for value in coefficient_values):
        blockers.append("thermalcal_coefficients_nonfinite")
    input_text = _scalar_text(attributes, "Model_input")
    output_text = _scalar_text(attributes, "Model_output")
    model_text = _scalar_text(attributes, "Model")
    if input_text != "Signal [DL]":
        blockers.append("thermalcal_input_not_raw_digital_level")
    if output_text is None or "Temperature" not in output_text:
        blockers.append("thermalcal_output_not_temperature")
    if model_text is None or not all(token in model_text for token in ("T", "x", "log")):
        blockers.append("thermalcal_model_text_not_formula_like")
    if intake.get("raw_signal_arrays_read") is not False:
        blockers.append("raw_signal_array_boundary_broken")
    if intake.get("cross_section_targets_read") is not False:
        blockers.append("cross_section_target_boundary_broken")
    if intake.get("calibration_fitting_performed") is not False:
        blockers.append("calibration_fit_boundary_broken")
    blockers = sorted(set(blockers))
    ready = not blockers
    return {
        "status": (
            "phase201_thermalcal_metadata_audit_ready_phase202_formula_contract_design"
            if ready
            else "phase201_thermalcal_metadata_audit_incomplete_or_ambiguous"
        ),
        "phase202_formula_contract_design_allowed": ready,
        "calibrated_temperature_descriptor_allowed": False,
        "calibration_formula_execution_allowed": False,
        "calibration_fitting_allowed": False,
        "model_training_allowed": False,
        "hyperparameter_search_allowed": False,
        "simulation_as_external_validation_allowed": False,
        "blocking_audits": blockers,
        "next_action": (
            "write and audit an explicit coefficient-to-formula/domain contract before computing any converted temperature value"
            if ready
            else "resolve ThermalCal metadata ambiguity before considering temperature conversion"
        ),
    }


def build_payload(phase200: dict[str, Any], intake: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": 201,
        "objective": "thermalcal_metadata_and_formula_readiness_audit",
        "temperature_conversion_boundary": (
            "This phase reads only the small ThermalCal metadata/coefficient object and the first Signal dataset attributes. "
            "It does not read a raw Signal array, calculate temperature, read cross-section targets, or fit calibration parameters."
        ),
        "thermalcal_intake": intake,
        "gate": build_gate(phase200, intake),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase200", type=Path, default=DEFAULT_PHASE200)
    parser.add_argument("--signal", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(_read_json(args.phase200), inspect_thermalcal(args.signal))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
