#!/usr/bin/env python3
"""Record the verified ThermalCal formula boundary without executing temperature conversion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_PHASE201 = Path(
    os.environ.get(
        "AMB2022_03_PHASE201_THERMALCAL_AUDIT",
        "/root/matsci-gnn-pinn-ops/phase201_amb2022_03_thermalcal_metadata_audit.json",
    )
)
OFFICIAL_REFERENCE = {
    "title": "Thermal Calibration of Commercial Melt Pool Monitoring Sensors on a Laser Powder Bed Fusion System",
    "author": "Brandon Lane",
    "report_id": "NIST AMS 100-35",
    "doi": "10.6028/NIST.AMS.100-35",
    "official_pdf_url": "https://nvlpubs.nist.gov/nistpubs/ams/NIST.AMS.100-35.pdf",
    "formula_locator": "Eq. (6)-(7), printed page 12 / PDF page 13",
}
CANONICAL_INVERSE_FORMULA = "T_equiv_C(S) = c2 / (a * ln(c / S + 1)) - b / a"
CANONICAL_FORWARD_FORMULA = "S = c / exp(c2 / (a * T_equiv_C + b))"
CANONICAL_CONSTANTS = {"c2_um_K": 14388.0}
REQUIRED_COEFFICIENTS = ("Coeff_a", "Coeff_b", "Coeff_c")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scalar_text(attributes: dict[str, Any], key: str) -> str | None:
    summary = attributes.get(key, {})
    values = summary.get("values") if isinstance(summary, dict) else None
    if not isinstance(values, list) or len(values) != 1:
        return None
    return str(values[0])


def _phase201_ready(phase201: dict[str, Any]) -> bool:
    gate = phase201.get("gate", {})
    return (
        gate.get("status") == "phase201_thermalcal_metadata_audit_ready_phase202_formula_contract_design"
        and bool(gate.get("phase202_formula_contract_design_allowed"))
        and gate.get("calibrated_temperature_descriptor_allowed") is False
        and gate.get("calibration_formula_execution_allowed") is False
        and gate.get("model_training_allowed") is False
    )


def build_formula_contract(phase201: dict[str, Any]) -> dict[str, Any]:
    intake = phase201.get("thermalcal_intake", {})
    attributes = intake.get("thermalcal_attributes", {})
    attribute_coefficients = intake.get("thermalcal_attribute_coefficients", {})
    hdf5_model_text = _scalar_text(attributes, "Model")
    hdf5_input = _scalar_text(attributes, "Model_input")
    hdf5_output = _scalar_text(attributes, "Model_output")
    missing_coefficients = [key for key in REQUIRED_COEFFICIENTS if key not in attribute_coefficients]
    undefined_symbols: list[str] = []
    if hdf5_model_text is not None and "e" in hdf5_model_text.lower() and "Coeff_e" not in attributes:
        undefined_symbols.append("e")
    return {
        "official_reference": OFFICIAL_REFERENCE,
        "canonical_inverse_formula": CANONICAL_INVERSE_FORMULA,
        "canonical_forward_formula": CANONICAL_FORWARD_FORMULA,
        "canonical_constants": CANONICAL_CONSTANTS,
        "canonical_domain": [
            "S must be strictly positive because the inverse formula contains c / S.",
            "a must be nonzero.",
            "The logarithm argument c / S + 1 must be positive.",
        ],
        "hdf5_model_text": hdf5_model_text,
        "hdf5_model_input": hdf5_input,
        "hdf5_model_output": hdf5_output,
        "hdf5_coefficients": attribute_coefficients,
        "missing_hdf5_coefficients": missing_coefficients,
        "undefined_hdf5_model_symbols": undefined_symbols,
        "official_source_absolute_temperature_disclaimer": (
            "The NIST report states that this calibration method does not provide an absolute calibration or the ability "
            "to ascribe real melt-pool temperatures to monitoring-sensor signals."
        ),
        "hdf5_to_canonical_formula_unambiguous": False,
        "temperature_conversion_executed": False,
        "cross_section_targets_read": False,
        "calibration_fitting_performed": False,
    }


def build_gate(phase201: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if not _phase201_ready(phase201):
        blockers.append("phase201_thermalcal_metadata_gate_not_ready")
    if contract.get("hdf5_model_input") != "Signal [DL]":
        blockers.append("hdf5_model_input_contract_broken")
    if not str(contract.get("hdf5_model_output", "")).startswith("Emissivity-Corrected Temperature"):
        blockers.append("hdf5_model_output_contract_broken")
    if contract.get("missing_hdf5_coefficients"):
        blockers.append("hdf5_required_coefficient_missing")
    if contract.get("undefined_hdf5_model_symbols"):
        blockers.append("hdf5_formula_contains_undefined_symbol")
    if contract.get("hdf5_to_canonical_formula_unambiguous") is not True:
        blockers.append("hdf5_formula_not_unambiguously_mapped_to_official_equation")
    if contract.get("temperature_conversion_executed") is not False:
        blockers.append("temperature_conversion_executed_before_resolution")
    if contract.get("cross_section_targets_read") is not False:
        blockers.append("cross_section_target_boundary_broken")
    if contract.get("calibration_fitting_performed") is not False:
        blockers.append("calibration_fit_boundary_broken")
    blockers = sorted(set(blockers))
    formula_contract_complete = "phase201_thermalcal_metadata_gate_not_ready" not in blockers
    return {
        "status": (
            "phase202_formula_contract_complete_temperature_conversion_blocked"
            if formula_contract_complete
            else "phase202_formula_contract_incomplete"
        ),
        "phase203_calibration_documentation_resolution_allowed": formula_contract_complete,
        "calibrated_temperature_descriptor_allowed": False,
        "calibration_formula_execution_allowed": False,
        "calibration_fitting_allowed": False,
        "model_training_allowed": False,
        "hyperparameter_search_allowed": False,
        "simulation_as_external_validation_allowed": False,
        "blocking_audits": blockers,
        "next_action": (
            "obtain the official AMB2022-03 calibration implementation or a documented definition of HDF5 symbol e; do not convert Signal to temperature"
            if formula_contract_complete
            else "repair the ThermalCal evidence trail before making any formula statement"
        ),
    }


def build_payload(phase201: dict[str, Any]) -> dict[str, Any]:
    contract = build_formula_contract(phase201)
    return {
        "phase": 202,
        "objective": "official_formula_to_hdf5_contract_audit_without_temperature_execution",
        "formula_contract": contract,
        "gate": build_gate(phase201, contract),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase201", type=Path, default=DEFAULT_PHASE201)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(_read_json(args.phase201))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
