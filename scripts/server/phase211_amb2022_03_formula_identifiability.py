#!/usr/bin/env python3
"""Audit whether verified AMB2022-03 data can identify the ThermalCal formula.

The analysis reads HDF5 object metadata and a checksum-verified related-release
packaging script. It never reads a Signal array, computes a temperature, or fits
an emissivity parameter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_PHASE210 = Path(
    os.environ.get(
        "AMB2022_PHASE210_INTEGRITY",
        "/root/matsci-gnn-pinn-data/derived/phase210/phase210_remote_data_integrity.json",
    )
)
DEFAULT_SIGNAL = Path(
    os.environ.get(
        "AMB2022_03_THERMOGRAPHY_SIGNAL",
        "/root/matsci-gnn-pinn-data/raw/ambench/2022_external_intake/"
        "AMB2022-03-thermography/Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5",
    )
)
DEFAULT_RELATED_SCRIPT = Path(
    os.environ.get(
        "AMB2022_01_RELATED_TEMPERATURE_SCRIPT",
        "/root/matsci-gnn-pinn-data/raw/ambench/2022_3d_build/AMB2022-01/mds2-2715/"
        "official/DataProcessingScripts/AMB2022_HDF5_Temperature_v1.m",
    )
)

REQUIRED_COEFFICIENTS = ("Coeff_a", "Coeff_b", "Coeff_c")


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("h5py is required for the Phase 211 metadata audit") from exc
    return h5py


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _decode_scalar(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return _decode_scalar(value[0]) if len(value) == 1 else [_decode_scalar(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def inspect_formula_observability(path: Path) -> dict[str, Any]:
    """Read only object names, shapes, and attributes; never dereference Signal data."""

    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        calibration = handle["Calibration/ThermalCal"]
        attributes = {str(key): _decode_scalar(value) for key, value in calibration.attrs.items()}
        calibration_datasets: list[dict[str, Any]] = []
        signal_dataset_count = 0

        def visitor(name: str, item: Any) -> None:
            nonlocal signal_dataset_count
            if not isinstance(item, h5py.Dataset):
                return
            if name.startswith("Calibration/"):
                calibration_datasets.append(
                    {
                        "path": name,
                        "shape": [int(size) for size in item.shape],
                        "dtype": str(item.dtype),
                    }
                )
            if Path(name).name == "Signal":
                signal_dataset_count += 1

        handle.visititems(visitor)
    coefficients: dict[str, float] = {}
    for key in REQUIRED_COEFFICIENTS:
        value = attributes.get(key)
        if isinstance(value, (int, float)):
            coefficients[key] = float(value)
    return {
        "thermalcal_object_kind": type(calibration).__name__,
        "thermalcal_attributes": attributes,
        "thermalcal_attribute_coefficients": coefficients,
        "calibration_dataset_count": len(calibration_datasets),
        "calibration_datasets": calibration_datasets,
        "signal_dataset_count": signal_dataset_count,
        "raw_signal_arrays_read": False,
        "temperature_conversion_executed": False,
        "calibration_fitting_performed": False,
    }


def parse_related_temperature_script(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    attributes: dict[str, Any] = {}
    for key in ("Model", "Model_input", "Model_output"):
        pattern = re.compile(rf"h5writeatt\([^\n;]*'{re.escape(key)}'\s*,\s*'([^']*)'", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            attributes[key] = match.group(1)
    for key in ("emiss", *REQUIRED_COEFFICIENTS):
        pattern = re.compile(rf"h5writeatt\([^\n;]*'{re.escape(key)}'\s*,\s*([+\-0-9.eE]+)", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            attributes[key] = float(match.group(1))
    return {
        "source_path": str(path),
        "packaging_attributes": attributes,
        "candidate_emissivity": attributes.get("emiss"),
        "contains_emissivity_attribute": "emiss" in attributes,
    }


def assess_identifiability(
    observability: dict[str, Any], related_script: dict[str, Any]
) -> dict[str, Any]:
    attributes = observability.get("thermalcal_attributes", {})
    if not isinstance(attributes, dict):
        attributes = {}
    model_text = str(attributes.get("Model", ""))
    hdf5_coefficients = observability.get("thermalcal_attribute_coefficients", {})
    if not isinstance(hdf5_coefficients, dict):
        hdf5_coefficients = {}
    related_attributes = related_script.get("packaging_attributes", {})
    if not isinstance(related_attributes, dict):
        related_attributes = {}
    coefficient_match = all(
        key in hdf5_coefficients
        and key in related_attributes
        and math.isclose(float(hdf5_coefficients[key]), float(related_attributes[key]))
        for key in REQUIRED_COEFFICIENTS
    )
    input_output_match = (
        attributes.get("Model_input") == related_attributes.get("Model_input")
        and attributes.get("Model_output") == related_attributes.get("Model_output")
    )
    has_undefined_e = "e" in model_text.lower() and "emiss" not in attributes and "Coeff_e" not in attributes
    calibration_dataset_count = int(observability.get("calibration_dataset_count", 0))
    observed_calibration_pairs_available = calibration_dataset_count > 0
    related_candidate_available = bool(
        related_script.get("contains_emissivity_attribute")
        and coefficient_match
        and input_output_match
    )
    return {
        "hdf5_model_text": model_text,
        "hdf5_contains_undefined_e_symbol": has_undefined_e,
        "observed_calibration_pairs_available": observed_calibration_pairs_available,
        "calibration_dataset_count": calibration_dataset_count,
        "signal_dataset_count": int(observability.get("signal_dataset_count", 0)),
        "related_release_candidate_available": related_candidate_available,
        "related_release_candidate_emissivity": (
            related_script.get("candidate_emissivity") if related_candidate_available else None
        ),
        "related_release_coefficient_match": coefficient_match,
        "related_release_input_output_match": input_output_match,
        "joint_c_e_scale_invariance": (
            "For a c*e/S term, jointly rescaling c by lambda and e by 1/lambda leaves the term unchanged."
        ),
        "e_data_identifiable": False,
        "e_estimate_from_amb2022_03": None,
        "identifiability_reason": (
            "AMB2022-03 exposes no calibration datasets or observed Signal-temperature pairs, so it supplies no "
            "estimation objective for e. The related-release emissivity is a provenance candidate, not a fit."
        ),
        "temperature_conversion_executed": False,
        "calibration_fitting_performed": False,
    }


def _phase210_ready(phase210: dict[str, Any]) -> bool:
    gate = phase210.get("gate", {})
    return (
        gate.get("status") == "phase210_remote_data_integrity_complete_phase211_formula_identifiability"
        and gate.get("phase211_formula_identifiability_allowed") is True
        and gate.get("raw_signal_integrity_verified") is True
        and gate.get("model_training_allowed") is False
    )


def build_gate(phase210: dict[str, Any], assessment: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if not _phase210_ready(phase210):
        blockers.append("phase210_integrity_gate_not_ready")
    if assessment.get("raw_signal_arrays_read") is not False:
        blockers.append("raw_signal_array_boundary_broken")
    if assessment.get("temperature_conversion_executed") is not False:
        blockers.append("temperature_conversion_executed_before_resolution")
    if assessment.get("calibration_fitting_performed") is not False:
        blockers.append("calibration_fitting_executed_before_resolution")
    complete = not blockers
    return {
        "status": (
            "phase211_candidate_formula_identifiability_complete_temperature_conversion_blocked"
            if complete
            else "phase211_candidate_formula_identifiability_incomplete"
        ),
        "official_formula_recovered": False,
        "e_data_identifiable": False,
        "candidate_emissivity_is_official": False,
        "temperature_conversion_allowed": False,
        "calibration_fitting_allowed": False,
        "model_training_allowed": False,
        "blocking_audits": blockers,
        "next_action": (
            "obtain AMB2022-03 calibration observations or official implementation; do not use the related-release candidate as an official formula"
            if complete
            else "repair the integrity or read-boundary evidence before interpreting formula identifiability"
        ),
    }


def build_payload(
    phase210: dict[str, Any], observability: dict[str, Any], related_script: dict[str, Any]
) -> dict[str, Any]:
    assessment = assess_identifiability(observability, related_script)
    assessment["raw_signal_arrays_read"] = observability.get("raw_signal_arrays_read")
    return {
        "phase": 211,
        "objective": "candidate_formula_identifiability_without_temperature_execution",
        "analysis_boundary": (
            "The audit reads only HDF5 object metadata and related-release source text. "
            "It does not read raw Signal arrays, derive temperature values, or fit e."
        ),
        "observability": observability,
        "related_release_candidate": related_script,
        "identifiability_assessment": assessment,
        "gate": build_gate(phase210, assessment),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase210", type=Path, default=DEFAULT_PHASE210)
    parser.add_argument("--signal", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--related-script", type=Path, default=DEFAULT_RELATED_SCRIPT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(
        _read_json(args.phase210),
        inspect_formula_observability(args.signal),
        parse_related_temperature_script(args.related_script),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
