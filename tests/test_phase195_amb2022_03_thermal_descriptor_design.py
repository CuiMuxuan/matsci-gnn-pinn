from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase195_amb2022_03_thermal_descriptor_design.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase195_descriptor_design", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase194() -> dict[str, object]:
    return {
        "gate": {
            "status": "phase194_calibration_protocol_design_ready_phase195_thermal_descriptor_extraction_design",
            "phase195_thermal_descriptor_extraction_design_allowed": True,
            "calibration_fitting_allowed": False,
            "model_training_allowed": False,
        }
    }


def _schema(*, threshold: float = 100.0) -> dict[str, object]:
    return {
        "signal_shape": [700, 640, 304],
        "signal_dtype": "uint16",
        "bit_depth": 12.0,
        "threshold_level": threshold,
        "signal_units": "digital levels",
    }


def test_phase195_admits_label_free_raw_descriptor_extraction_only():
    module = _load_module()
    design = module.build_design(_phase194(), _schema())
    gate = design["gate"]
    assert gate["phase196_descriptor_extraction_allowed"] is True
    assert gate["calibrated_temperature_descriptor_allowed"] is False
    assert gate["calibration_fitting_allowed"] is False
    assert all(row["uses_target"] is False for row in design["descriptors"])
    assert all(row["uses_temperature_conversion"] is False for row in design["descriptors"])


def test_phase195_blocks_descriptor_extraction_when_fixed_threshold_changes():
    module = _load_module()
    gate = module.build_gate(_phase194(), _schema(threshold=99.0))
    assert gate["phase196_descriptor_extraction_allowed"] is False
    assert "raw_signal_threshold_not_100" in gate["blocking_audits"]
