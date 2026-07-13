from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase183_amb2022_01_layer_space_data_quality_gate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase183_layer_space_quality", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inspection() -> dict[str, object]:
    per_split = {}
    for split_name, build_id in (("train", "B6"), ("val", "B7"), ("test", "B8")):
        per_split[split_name] = {
            "rows": 100_000,
            "build_ids": [build_id],
            "valid_both_fraction": 0.7,
            "active_laser_fraction": 0.5,
            "tam_std_valid": 0.001,
            "scr_std_valid": 1000.0,
        }
    return {
        "dataset_row_count": 300_000,
        "manifest_row_count": 300_000,
        "feature_matrix_finite": True,
        "contains_build_identity_feature": False,
        "scr_units": "C/s",
        "manifest_primary_split": {
            "strategy": "build_holdout_replication",
            "build_identity_in_feature_matrix": False,
        },
        "per_split": per_split,
    }


def test_quality_gate_admits_valid_build_holdout_dataset():
    module = _load_module()
    gate = module.build_gate(_inspection())
    assert gate["status"] == "phase183_layer_space_data_quality_ready_phase184_baseline_design"
    assert gate["phase184_baseline_design_allowed"] is True
    assert gate["model_training_allowed"] is False


def test_quality_gate_blocks_feature_leakage_and_low_coverage():
    module = _load_module()
    inspected = _inspection()
    inspected["contains_build_identity_feature"] = True
    inspected["per_split"]["test"]["valid_both_fraction"] = 0.1
    gate = module.build_gate(inspected)
    assert gate["status"] == "phase183_layer_space_data_quality_blocked"
    assert "build_identity_feature_present" in gate["blocking_audits"]
    assert "test_insufficient_dual_target_coverage" in gate["blocking_audits"]
