from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase202_amb2022_03_formula_contract.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase202_formula_contract", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase201(*, with_e: bool = True) -> dict[str, object]:
    model = "T(x) = 14388/a/log((c*e/x+1)-b/a;" if with_e else "T(x) = 14388/a/log(c/x+1)-b/a"
    return {
        "thermalcal_intake": {
            "thermalcal_attributes": {
                "Model": {"values": [model]},
                "Model_input": {"values": ["Signal [DL]"]},
                "Model_output": {"values": ["Emissivity-Corrected Temperature [oC]"]},
            },
            "thermalcal_attribute_coefficients": {
                "Coeff_a": 0.9655,
                "Coeff_b": 197.2,
                "Coeff_c": 43920000.0,
            },
        },
        "gate": {
            "status": "phase201_thermalcal_metadata_audit_ready_phase202_formula_contract_design",
            "phase202_formula_contract_design_allowed": True,
            "calibrated_temperature_descriptor_allowed": False,
            "calibration_formula_execution_allowed": False,
            "model_training_allowed": False,
        },
    }


def test_phase202_records_official_equation_but_blocks_temperature_conversion_for_undefined_e():
    module = _load_module()
    payload = module.build_payload(_phase201())

    assert payload["formula_contract"]["canonical_constants"]["c2_um_K"] == 14388.0
    assert payload["formula_contract"]["undefined_hdf5_model_symbols"] == ["e"]
    assert payload["gate"]["phase203_calibration_documentation_resolution_allowed"] is True
    assert payload["gate"]["calibrated_temperature_descriptor_allowed"] is False
    assert "hdf5_formula_contains_undefined_symbol" in payload["gate"]["blocking_audits"]


def test_phase202_keeps_conversion_blocked_even_without_e_until_mapping_is_documented():
    module = _load_module()
    payload = module.build_payload(_phase201(with_e=False))

    assert payload["formula_contract"]["undefined_hdf5_model_symbols"] == []
    assert payload["gate"]["calibration_formula_execution_allowed"] is False
    assert "hdf5_formula_not_unambiguously_mapped_to_official_equation" in payload["gate"]["blocking_audits"]
