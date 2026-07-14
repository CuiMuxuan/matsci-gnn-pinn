from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase194_amb2022_03_calibration_protocol_design.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase194_calibration_protocol", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase193() -> dict[str, object]:
    return {
        "gate": {
            "status": "phase193_identifier_join_design_ready_phase194_calibration_protocol_design",
            "phase194_calibration_protocol_design_allowed": True,
            "calibration_fitting_allowed": False,
            "model_training_allowed": False,
        }
    }


def _conditions() -> list[dict[str, object]]:
    rows = []
    for setting in range(7):
        for repeat in range(3):
            rows.append(
                {
                    "thermal_group_id": f"Line_{setting}_{repeat}",
                    "laser_power_W": 200.0 + setting,
                    "scan_speed_mm_s": 900.0,
                    "spot_size_um": 67.0,
                }
            )
    return rows


def test_phase194_freezes_leave_one_process_setting_out_without_leakage():
    module = _load_module()
    protocol = module.build_protocol(_phase193(), _conditions())
    gate = protocol["gate"]
    assert len(protocol["settings"]) == 7
    assert len(protocol["folds"]) == 7
    assert gate["phase195_thermal_descriptor_extraction_design_allowed"] is True
    assert gate["calibration_fitting_allowed"] is False
    for fold in protocol["folds"]:
        assert fold["held_out_setting_id"] not in fold["training_setting_ids"]


def test_phase194_blocks_process_setting_with_wrong_replicate_count():
    module = _load_module()
    rows = _conditions()[:-1]
    protocol = module.build_protocol(_phase193(), rows)
    assert protocol["gate"]["phase195_thermal_descriptor_extraction_design_allowed"] is False
    assert "process_setting_replicate_contract_not_three" in protocol["gate"]["blocking_audits"]
