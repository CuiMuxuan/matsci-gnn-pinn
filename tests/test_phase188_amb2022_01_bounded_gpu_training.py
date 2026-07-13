from __future__ import annotations

import importlib.util
import math
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase188_amb2022_01_bounded_gpu_training.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase188_bounded_gpu", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metric_rows() -> list[dict[str, object]]:
    rows = []
    for target in ("tam_s", "scr_C_per_s"):
        for variant, rmse in (("small_data_only_mlp", 1.0), ("physics_regularized_history_mlp", 0.9)):
            for seed in (1871, 1872, 1873):
                for split in ("train", "val", "test"):
                    rows.append({"target": target, "variant_id": variant, "seed": seed, "split": split, "rmse": rmse})
    return rows


def test_phase188_gate_requires_all_fixed_seed_rows_and_monotonic_audits():
    module = _load_module()
    audits = [
        {
            "variant_id": "physics_regularized_history_mlp",
            "seed": seed,
            "monotonic_violation_fraction_test": 0.1,
        }
        for seed in (1871, 1872, 1873)
    ]
    gate = module.build_gate(_metric_rows(), audits)
    assert gate["training_complete"] is True
    assert gate["model_training_allowed"] is False
    assert math.isclose(gate["candidate_monotonic_violation_fraction_test_mean"], 0.1)


def test_phase188_gate_blocks_missing_candidate_seed():
    module = _load_module()
    audits = [
        {
            "variant_id": "physics_regularized_history_mlp",
            "seed": 1871,
            "monotonic_violation_fraction_test": 0.1,
        }
    ]
    gate = module.build_gate(_metric_rows(), audits)
    assert gate["training_complete"] is False
    assert "candidate_monotonic_audit_missing" in gate["blocking_audits"]


def test_phase188_gate_blocks_duplicate_metric_seed():
    rows = _metric_rows()
    for row in rows:
        if (
            row["target"] == "tam_s"
            and row["variant_id"] == "small_data_only_mlp"
            and row["seed"] == 1873
            and row["split"] == "test"
        ):
            row["seed"] = 1871
            break
    audits = [
        {
            "variant_id": "physics_regularized_history_mlp",
            "seed": seed,
            "monotonic_violation_fraction_test": 0.1,
        }
        for seed in (1871, 1872, 1873)
    ]
    gate = _load_module().build_gate(rows, audits)
    assert gate["training_complete"] is False
    assert "tam_s_test_small_data_only_mlp_seed_count" in gate["blocking_audits"]
