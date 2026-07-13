from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase184_amb2022_01_layer_space_ridge_baselines.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase184_layer_space_ridge", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ridge_recovers_centered_linear_relation():
    module = _load_module()
    x = np.arange(20, dtype=float).reshape(-1, 1)
    y = 2.5 * x[:, 0] - 3.0
    model = module.fit_ridge(x, y, alpha=1e-8)
    pred = module.predict_ridge(model, x)
    assert np.max(np.abs(pred - y)) < 1e-5


def test_baseline_gate_requires_complete_target_comparisons():
    module = _load_module()
    rows = []
    for target in module.TARGETS:
        for profile in ("coordinate_layer", "full"):
            for split in module.SPLITS:
                rows.append(
                    {
                        "target": target,
                        "model": "ridge",
                        "profile": profile,
                        "split": split,
                        "rmse": 1.0 if profile == "coordinate_layer" else 0.9,
                    }
                )
    gate = module.build_gate(rows)
    assert gate["phase185_ablation_contract_allowed"] is True
    assert gate["model_training_allowed"] is False
    assert gate["scan_feature_gains_vs_coordinate"]["tam_s"]["test_rmse_gain_vs_coordinate"] > 0.0
