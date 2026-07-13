from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase181_amb2022_01_trigger_layer_space_registration_gate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase181_trigger_layer_space", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _target(build_id: str, target_name: str, *, ready: bool = True) -> dict[str, object]:
    return {
        "build_id": build_id,
        "target_name": target_name,
        "ready": ready,
    }


def test_trigger_layer_space_gate_admits_complete_registered_targets():
    module = _load_module()
    targets = [
        _target(build_id, target_name)
        for build_id in ("B6", "B7", "B8")
        for target_name in ("TAM", "SCR")
    ]
    gate = module.build_gate(
        scan_summary={"trigger_layer_time_ready": True},
        target_summaries=targets,
    )
    assert gate["status"] == "phase181_trigger_layer_space_registration_ready_phase182_dataset_construction"
    assert gate["phase182_dataset_construction_allowed"] is True
    assert gate["model_training_allowed"] is False
    assert gate["raw_frame_event_causal_training_allowed"] is False
    assert gate["absolute_wall_clock_trajectory_claim_allowed"] is False


def test_trigger_layer_space_gate_blocks_missing_or_unregistered_target():
    module = _load_module()
    targets = [
        _target(build_id, target_name, ready=not (build_id == "B8" and target_name == "SCR"))
        for build_id in ("B6", "B7", "B8")
        for target_name in ("TAM", "SCR")
    ]
    gate = module.build_gate(
        scan_summary={"trigger_layer_time_ready": False},
        target_summaries=targets,
    )
    assert gate["status"] == "phase181_trigger_layer_space_registration_blocked"
    assert gate["phase182_dataset_construction_allowed"] is False
    assert "xypt_trigger_layer_time_not_auditable" in gate["blocking_audits"]
    assert "B8_scr_not_registered" in gate["blocking_audits"]
