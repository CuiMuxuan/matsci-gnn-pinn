from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase211_amb2022_03_formula_identifiability.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase211_formula_identifiability", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase210(*, ready: bool = True) -> dict[str, object]:
    return {
        "gate": {
            "status": (
                "phase210_remote_data_integrity_complete_phase211_formula_identifiability"
                if ready
                else "phase210_remote_data_integrity_incomplete_or_mismatched"
            ),
            "phase211_formula_identifiability_allowed": ready,
            "raw_signal_integrity_verified": ready,
            "model_training_allowed": False,
        }
    }


def _observability(module) -> dict[str, object]:
    return {
        "thermalcal_attributes": {
            "Model": "T(x) = 14388/a/log((c*e/x+1)-b/a;",
            "Model_input": "Signal [DL]",
            "Model_output": "Emissivity-Corrected Temperature [^oC]",
        },
        "thermalcal_attribute_coefficients": {
            "Coeff_a": 0.9655,
            "Coeff_b": 197.2,
            "Coeff_c": 4.392e7,
        },
        "calibration_dataset_count": 0,
        "signal_dataset_count": 27,
        "raw_signal_arrays_read": False,
        "temperature_conversion_executed": False,
        "calibration_fitting_performed": False,
    }


def test_related_release_candidate_is_not_data_identified():
    module = _load_module()
    related = {
        "contains_emissivity_attribute": True,
        "candidate_emissivity": 0.5,
        "packaging_attributes": {
            "emiss": 0.5,
            "Coeff_a": 0.9655,
            "Coeff_b": 197.2,
            "Coeff_c": 4.392e7,
            "Model_input": "Signal [DL]",
            "Model_output": "Emissivity-Corrected Temperature [^oC]",
        },
    }

    assessment = module.assess_identifiability(_observability(module), related)
    payload = module.build_payload(_phase210(), _observability(module), related)

    assert assessment["related_release_candidate_available"] is True
    assert assessment["related_release_candidate_emissivity"] == 0.5
    assert assessment["e_data_identifiable"] is False
    assert assessment["e_estimate_from_amb2022_03"] is None
    assert payload["gate"]["temperature_conversion_allowed"] is False


def test_gate_blocks_when_integrity_is_not_ready():
    module = _load_module()
    related = {"contains_emissivity_attribute": False, "packaging_attributes": {}}

    payload = module.build_payload(_phase210(ready=False), _observability(module), related)

    assert payload["gate"]["status"] == "phase211_candidate_formula_identifiability_incomplete"
    assert "phase210_integrity_gate_not_ready" in payload["gate"]["blocking_audits"]


def test_related_script_parser_extracts_emissivity_and_coefficients(tmp_path: Path):
    module = _load_module()
    script = tmp_path / "AMB2022_HDF5_Temperature_v1.m"
    script.write_text(
        "h5writeatt(file,'/Calibration/ThermalCal/','Model_input','Signal [DL]');\n"
        "h5writeatt(file,'/Calibration/ThermalCal/','Model_output','Emissivity-Corrected Temperature [^oC]');\n"
        "h5writeatt(file,'/Calibration/ThermalCal/','emiss', 0.5);\n"
        "h5writeatt(file,'/Calibration/ThermalCal/','Coeff_a', 0.9655);\n"
        "h5writeatt(file,'/Calibration/ThermalCal/','Coeff_b', 197.2);\n"
        "h5writeatt(file,'/Calibration/ThermalCal/','Coeff_c', 4.392E7);\n",
        encoding="utf-8",
    )

    parsed = module.parse_related_temperature_script(script)

    assert parsed["candidate_emissivity"] == 0.5
    assert parsed["packaging_attributes"]["Coeff_c"] == 4.392e7
