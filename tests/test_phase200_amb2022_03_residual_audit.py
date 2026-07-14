from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase200_amb2022_03_residual_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase200_residual_audit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase199(fold_metrics: list[dict[str, object]], pooled_metrics: list[dict[str, object]]) -> dict[str, object]:
    return {
        "fold_metrics": fold_metrics,
        "pooled_metrics": pooled_metrics,
        "gate": {
            "status": "phase199_fixed_baseline_execution_ready_phase200_residual_audit",
            "phase200_residual_audit_allowed": True,
            "calibration_fitting_allowed": False,
            "model_training_allowed": False,
        },
    }


def _metrics(module):
    fold_metrics: list[dict[str, object]] = []
    pooled_metrics: list[dict[str, object]] = []
    offsets = {
        "train_mean_control": 3.0,
        "process_ridge_control": 0.0,
        "raw_thermal_ridge": 4.0,
        "process_plus_raw_thermal_ridge": 2.0,
        "shuffled_raw_thermal_negative_control": 1.0,
    }
    for target in module.TARGET_COLUMNS:
        for variant in module.VARIANT_IDS:
            for fold_index in range(7):
                fold_metrics.append(
                    {
                        "fold_id": f"fold_{fold_index}",
                        "target_mean_column": target,
                        "variant_id": variant,
                        "rmse_um": 10.0 + offsets[variant] + fold_index * 0.1,
                    }
                )
            pooled_metrics.append(
                {
                    "target_mean_column": target,
                    "variant_id": variant,
                    "rmse_um": 10.0 + offsets[variant],
                    "mae_um": 8.0 + offsets[variant],
                    "median_repeatability_normalized_absolute_residual": 2.0,
                }
            )
    return fold_metrics, pooled_metrics


def test_phase200_reports_no_robust_additive_signal_without_escalating_model_capacity():
    module = _load_module()
    fold_metrics, pooled_metrics = _metrics(module)
    payload = module.build_payload(_phase199(fold_metrics, pooled_metrics))

    assert payload["residual_audit"]["metric_contract_complete"] is True
    assert payload["residual_audit"]["thermal_additive_signal_supported_across_both_targets"] is False
    assert payload["gate"]["phase201_mechanistic_stress_test_design_allowed"] is True
    assert payload["gate"]["thermal_descriptor_model_escalation_allowed"] is False
    assert payload["gate"]["status"].endswith("no_robust_additive_thermal_signal")


def test_phase200_blocks_incomplete_metric_contract():
    module = _load_module()
    fold_metrics, pooled_metrics = _metrics(module)
    incomplete_phase199 = _phase199(fold_metrics[:-1], deepcopy(pooled_metrics))
    payload = module.build_payload(incomplete_phase199)

    assert payload["gate"]["phase201_mechanistic_stress_test_design_allowed"] is False
    assert "fixed_metric_contract_incomplete" in payload["gate"]["blocking_audits"]
