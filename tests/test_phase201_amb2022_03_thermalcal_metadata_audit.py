from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase201_amb2022_03_thermalcal_metadata_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase201_thermalcal_metadata_audit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase200() -> dict[str, object]:
    return {
        "gate": {
            "status": "phase200_residual_audit_complete_no_robust_additive_thermal_signal",
            "phase201_mechanistic_stress_test_design_allowed": True,
            "model_training_allowed": False,
        }
    }


def _intake(*, complete: bool = True) -> dict[str, object]:
    attributes = {
        "Cal_Method": {"values": ["RegressionF_ArrayAvg"]},
        "Model": {"values": ["T(x) = a/log(x)"]},
        "Model_input": {"values": ["Signal [DL]"]},
        "Model_output": {"values": ["Emissivity-Corrected Temperature [oC]"]},
    }
    if not complete:
        attributes.pop("Model")
    return {
        "thermalcal_attributes": attributes,
        "thermalcal_coefficient_values": [1.0, 2.0, 3.0],
        "raw_signal_arrays_read": False,
        "cross_section_targets_read": False,
        "calibration_fitting_performed": False,
    }


def test_phase201_admits_formula_contract_design_but_not_temperature_conversion():
    module = _load_module()
    payload = module.build_payload(_phase200(), _intake())

    assert payload["gate"]["phase202_formula_contract_design_allowed"] is True
    assert payload["gate"]["calibrated_temperature_descriptor_allowed"] is False
    assert payload["gate"]["calibration_formula_execution_allowed"] is False
    assert payload["thermalcal_intake"]["cross_section_targets_read"] is False


def test_phase201_blocks_incomplete_thermalcal_metadata():
    module = _load_module()
    gate = module.build_gate(_phase200(), _intake(complete=False))

    assert gate["phase202_formula_contract_design_allowed"] is False
    assert "thermalcal_required_metadata_missing" in gate["blocking_audits"]
