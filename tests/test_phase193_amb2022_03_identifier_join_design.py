from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase193_amb2022_03_identifier_join_design.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase193_identifier_join", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase192() -> dict[str, object]:
    return {
        "gate": {
            "status": "phase192_amb2022_03_calibration_intake_ready_phase193_identifier_join_design",
            "phase193_identifier_join_design_allowed": True,
            "calibration_fitting_allowed": False,
            "model_training_allowed": False,
        }
    }


def _audit(*, mismatch: bool = False) -> dict[str, object]:
    return {
        "thermal_single_track_count": 2,
        "workbook_single_track_row_count": 4,
        "joined_condition_count": 2,
        "unmatched_thermal_ids": [],
        "process_mismatch_thermal_ids": ["Line_0_1"] if mismatch else [],
        "extra_workbook_case_keys": [],
        "excluded_pad_signal_ids": ["X_pad1", "Y_pad1"],
    }


def _rows() -> list[dict[str, object]]:
    return [
        {"thermal_group_id": "Line_0_1", "n_cross_section_replicates": 2},
        {"thermal_group_id": "Line_0_2", "n_cross_section_replicates": 2},
    ]


def test_normalized_line_identifier_matches_workbook_punctuation_convention():
    module = _load_module()
    assert module.normalize_condition_id("Line_1_2_3") == module.normalize_condition_id("Line 1.2_3")


def test_condition_join_requires_matching_process_metadata_and_two_replicates():
    module = _load_module()
    thermal = {
        "Line_0_1": {
            "laser_power_W": 285.0,
            "scan_speed_mm_s": 960.0,
            "spot_size_um": 67.0,
        }
    }
    workbook_rows = [
        {
            "Sample": "AMB2022-718-SH1-BP1",
            "Case and Line No.": "Line 0_1",
            "Power (W)": 285.0,
            "Velocity (mm/s)": 960.0,
            "Beam diameter (gauss, avg) (µm)": 67.0,
            "Depth (µm)": 100.0,
            "Width (µm)": 120.0,
        },
        {
            "Sample": "AMB2022-718-SH1-BP1",
            "Case and Line No.": "Line 0_1",
            "Power (W)": 285.0,
            "Velocity (mm/s)": 960.0,
            "Beam diameter (gauss, avg) (µm)": 67.0,
            "Depth (µm)": 102.0,
            "Width (µm)": 122.0,
        },
    ]
    rows, audit = module.build_condition_join(thermal, workbook_rows, ["X_pad1"])
    assert len(rows) == 1
    assert rows[0]["n_cross_section_replicates"] == 2
    assert rows[0]["depth_um_mean"] == 101.0
    assert audit["unmatched_thermal_ids"] == []
    assert audit["process_mismatch_thermal_ids"] == []


def test_phase193_admits_unambiguous_condition_join_but_not_calibration_fitting():
    module = _load_module()
    gate = module.build_gate(_phase192(), _audit(), _rows())
    assert gate["phase194_calibration_protocol_design_allowed"] is True
    assert gate["calibration_fitting_allowed"] is False
    assert gate["excluded_pad_signal_count"] == 2
    assert gate["model_training_allowed"] is False


def test_phase193_blocks_process_mismatch():
    module = _load_module()
    gate = module.build_gate(_phase192(), _audit(mismatch=True), _rows())
    assert gate["phase194_calibration_protocol_design_allowed"] is False
    assert "thermal_workbook_process_mismatch" in gate["blocking_audits"]
