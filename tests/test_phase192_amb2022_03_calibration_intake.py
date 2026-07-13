from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase192_amb2022_03_calibration_intake.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase192_calibration_intake", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase191(*, ready: bool = True) -> dict[str, object]:
    return {
        "gate": {
            "status": (
                "phase191_external_confirmation_design_ready_phase192_local_calibration_intake"
                if ready
                else "phase191_external_confirmation_design_waiting_for_calibration_intake"
            ),
            "phase192_local_calibration_intake_allowed": ready,
            "model_training_allowed": False,
            "post_b8_model_reselection_allowed": False,
        }
    }


def _workbook(module, *, complete: bool = True) -> dict[str, object]:
    headers = set(module.WORKBOOK_REQUIRED_COLUMNS)
    if not complete:
        headers.remove("Depth (µm)")
    return {"headers": {"Sheet1": sorted(headers)}, "data_row_count": 10}


def test_phase192_admits_identifier_join_design_but_not_calibration_fitting():
    module = _load_module()
    gate = module.build_gate(
        phase191=_phase191(),
        thermography_summary={"signal_group_count": 5},
        xypt_summary={"xypt_group_count": 2},
        workbook_summary=_workbook(module),
    )
    assert gate["phase193_identifier_join_design_allowed"] is True
    assert gate["calibration_fitting_allowed"] is False
    assert gate["model_training_allowed"] is False
    assert gate["independent_3d_temperature_confirmation_allowed"] is False


def test_phase192_blocks_missing_cross_section_physical_label_column():
    module = _load_module()
    gate = module.build_gate(
        phase191=_phase191(),
        thermography_summary={"signal_group_count": 5},
        xypt_summary={"xypt_group_count": 2},
        workbook_summary=_workbook(module, complete=False),
    )
    assert gate["phase193_identifier_join_design_allowed"] is False
    assert "workbook_column_missing_Depth (µm)" in gate["blocking_audits"]
