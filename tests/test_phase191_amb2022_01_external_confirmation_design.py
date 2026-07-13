from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase191_amb2022_01_external_confirmation_design.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase191_external_confirmation", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase190(*, ready: bool = True) -> dict[str, object]:
    return {
        "gate": {
            "status": (
                "phase190_spatial_failure_analysis_ready_phase191_external_confirmation_design"
                if ready
                else "phase190_spatial_failure_analysis_incomplete_or_unverified"
            ),
            "phase191_external_confirmation_design_allowed": ready,
            "model_training_allowed": False,
            "post_b8_model_reselection_allowed": False,
        }
    }


def _file_rows(module, *, verified: bool = True) -> list[dict[str, object]]:
    return [{"id": row["id"], "verified": verified} for row in module.FILE_REQUIREMENTS]


def test_phase191_admits_only_local_calibration_intake_and_blocks_external_claims():
    module = _load_module()
    gate = module.build_gate(_phase190(), _file_rows(module))
    assert gate["phase192_local_calibration_intake_allowed"] is True
    assert gate["phase192_amb2022_02_truth_discovery_allowed"] is True
    assert gate["independent_3d_temperature_confirmation_allowed"] is False
    assert gate["amb2022_02_templates_as_ground_truth_allowed"] is False
    assert gate["simulation_as_external_validation_allowed"] is False
    assert gate["model_training_allowed"] is False


def test_phase191_blocks_calibration_intake_when_amb2022_03_signal_is_unverified():
    module = _load_module()
    rows = _file_rows(module)
    for row in rows:
        if row["id"] == "amb2022_03_staring_signal":
            row["verified"] = False
            break
    gate = module.build_gate(_phase190(), rows)
    assert gate["phase192_local_calibration_intake_allowed"] is False
    assert "missing_or_unverified_amb2022_03_staring_signal" in gate["blocking_audits"]


def test_phase191_blocks_all_intake_when_phase190_is_not_verified():
    module = _load_module()
    gate = module.build_gate(_phase190(ready=False), _file_rows(module))
    assert gate["phase192_local_calibration_intake_allowed"] is False
    assert gate["phase192_amb2022_02_truth_discovery_allowed"] is False
    assert "phase190_spatial_failure_gate_not_ready" in gate["blocking_audits"]
