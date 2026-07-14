from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase199_amb2022_03_fixed_baseline_execution.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase199_fixed_baseline_execution", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase197() -> dict[str, object]:
    return {
        "gate": {
            "status": "phase197_calibration_table_design_ready_phase198_baseline_contract_design",
            "phase198_baseline_contract_design_allowed": True,
            "calibration_fitting_allowed": False,
            "model_training_allowed": False,
        }
    }


def _phase198(module) -> dict[str, object]:
    process_features = ("laser_power_W", "scan_speed_mm_s", "spot_size_um")
    raw_features = module.RAW_THERMAL_FEATURES
    all_features = process_features + raw_features
    variants = [
        {"variant_id": "train_mean_control", "feature_columns": ""},
        {"variant_id": "process_ridge_control", "feature_columns": ";".join(process_features)},
        {"variant_id": "raw_thermal_ridge", "feature_columns": ";".join(raw_features)},
        {"variant_id": "process_plus_raw_thermal_ridge", "feature_columns": ";".join(all_features)},
        {"variant_id": "shuffled_raw_thermal_negative_control", "feature_columns": ";".join(all_features)},
    ]
    return {
        "ridge_alpha": 1.0,
        "shuffle_seed_base": 1981,
        "targets": [
            {"target_mean_column": "depth_um_mean", "repeatability_std_column": "depth_um_std"},
            {"target_mean_column": "width_um_mean", "repeatability_std_column": "width_um_std"},
        ],
        "baseline_contract": variants,
        "gate": {
            "status": "phase198_baseline_contract_ready_phase199_fixed_baseline_execution",
            "phase199_fixed_baseline_execution_allowed": True,
            "calibration_fitting_allowed": False,
            "model_training_allowed": False,
        },
    }


def _fixtures(module):
    rows: list[dict[str, object]] = []
    folds: list[dict[str, str]] = []
    for setting_index in range(7):
        setting_id = f"P{setting_index}"
        held_out_ids: list[str] = []
        for replicate_index in range(3):
            group_id = f"Line_{setting_index}_{replicate_index}"
            held_out_ids.append(group_id)
            row = {
                "thermal_group_id": group_id,
                "setting_id": setting_id,
                "laser_power_W": float(setting_index),
                "scan_speed_mm_s": float(setting_index + 1),
                "spot_size_um": float(setting_index + 2),
                "depth_um_mean": float(10 + setting_index + replicate_index),
                "depth_um_std": 1.0,
                "width_um_mean": float(20 + setting_index + replicate_index),
                "width_um_std": 2.0,
            }
            row.update(
                {
                    feature: float(setting_index * 5 + replicate_index + feature_index)
                    for feature_index, feature in enumerate(module.RAW_THERMAL_FEATURES)
                }
            )
            rows.append(row)
        folds.append(
            {
                "fold_id": f"fold_{setting_index}",
                "held_out_setting_id": setting_id,
                "held_out_thermal_group_ids": ";".join(held_out_ids),
                "training_setting_ids": ";".join(f"P{other}" for other in range(7) if other != setting_index),
            }
        )
    return rows, folds


def test_phase199_executes_all_fixed_variants_without_selection():
    module = _load_module()
    phase198 = _phase198(module)
    rows, folds = _fixtures(module)
    predictions, fold_metrics, pooled_metrics, audit = module.evaluate_contract(rows, folds, phase198)
    payload = module.build_payload(_phase197(), phase198, predictions, fold_metrics, pooled_metrics, audit)

    assert len(predictions) == 21 * 5 * 2
    assert len(fold_metrics) == 7 * 5 * 2
    assert len(pooled_metrics) == 5 * 2
    assert payload["gate"]["phase200_residual_audit_allowed"] is True
    assert audit["feature_normalization_fit_scope"].startswith("Training rows only")
    assert audit["post_holdout_model_selection_performed"] is False


def test_phase199_gate_blocks_duplicate_predictions_or_selection():
    module = _load_module()
    phase198 = _phase198(module)
    rows, folds = _fixtures(module)
    predictions, fold_metrics, pooled_metrics, audit = module.evaluate_contract(rows, folds, phase198)
    duplicate_predictions = predictions + [deepcopy(predictions[0])]
    selection_audit = deepcopy(audit)
    selection_audit["post_holdout_model_selection_performed"] = True
    gate = module.build_gate(_phase197(), phase198, duplicate_predictions, fold_metrics, pooled_metrics, selection_audit)

    assert gate["phase200_residual_audit_allowed"] is False
    assert "duplicate_prediction_key" in gate["blocking_audits"]
    assert "post_holdout_model_selection_detected" in gate["blocking_audits"]
