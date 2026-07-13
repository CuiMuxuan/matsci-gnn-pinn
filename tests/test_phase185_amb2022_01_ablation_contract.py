from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase185_amb2022_01_ablation_contract.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase185_ablation_contract", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase184(*, gains: bool = True) -> dict[str, object]:
    value = 1.0 if gains else 0.0
    return {
        "gate": {
            "status": "phase184_low_capacity_baselines_ready_phase185_ablation_contract",
            "phase185_ablation_contract_allowed": True,
            "scan_feature_gains_vs_coordinate": {
                "tam_s": {
                    "validation_rmse_gain_vs_coordinate": value,
                    "test_rmse_gain_vs_coordinate": value,
                },
                "scr_C_per_s": {
                    "validation_rmse_gain_vs_coordinate": value,
                    "test_rmse_gain_vs_coordinate": value,
                },
            },
        }
    }


def test_ablation_gate_requires_positive_scan_history_gains():
    module = _load_module()
    ready = module.build_gate(_phase184(gains=True))
    blocked = module.build_gate(_phase184(gains=False))
    assert ready["phase186_feature_ablation_execution_allowed"] is True
    assert ready["model_training_allowed"] is False
    assert "tam_s_no_validation_scan_history_gain" in blocked["blocking_audits"]


def test_contract_contains_required_negative_control_and_candidate():
    module = _load_module()
    rows = module.build_ablation_rows()
    ids = {row["variant_id"] for row in rows}
    assert "layerwise_shuffled_scan_history_control" in ids
    assert "heat_kernel_history_ridge" in ids
    assert "physics_regularized_history_mlp" in ids
