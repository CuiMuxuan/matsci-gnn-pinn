#!/usr/bin/env python3
"""Execute the fixed low-capacity AMB2022-03 leave-one-setting-out baseline contract."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PHASE197 = Path(
    os.environ.get(
        "AMB2022_03_PHASE197_CALIBRATION_TABLE",
        "/root/matsci-gnn-pinn-ops/phase197_amb2022_03_calibration_table_design.json",
    )
)
DEFAULT_PHASE198 = Path(
    os.environ.get(
        "AMB2022_03_PHASE198_BASELINE_CONTRACT",
        "/root/matsci-gnn-pinn-ops/phase198_amb2022_03_baseline_contract.json",
    )
)
DEFAULT_TABLE = Path(
    os.environ.get(
        "AMB2022_03_PHASE197_CALIBRATION_TABLE_CSV",
        "/root/matsci-gnn-pinn-ops/phase197_amb2022_03_calibration_table.csv",
    )
)
DEFAULT_FOLDS = Path(
    os.environ.get(
        "AMB2022_03_PHASE197_FOLDS_CSV",
        "/root/matsci-gnn-pinn-ops/phase197_amb2022_03_leave_setting_out_folds.csv",
    )
)
EPSILON = 1e-12
EXPECTED_CONDITION_COUNT = 21
EXPECTED_PROCESS_SETTING_COUNT = 7
EXPECTED_CONDITIONS_PER_SETTING = 3
RAW_THERMAL_FEATURES = (
    "signal_mean_dl",
    "signal_std_dl",
    "signal_max_dl",
    "signal_p99_dl",
    "above_threshold_fraction",
    "active_frame_fraction",
    "frame_max_mean_dl",
    "frame_max_std_dl",
)
PREDICTION_FIELDS = (
    "fold_id",
    "held_out_setting_id",
    "variant_id",
    "target_mean_column",
    "target_repeatability_std_column",
    "thermal_group_id",
    "observed_mean_um",
    "observed_repeatability_std_um",
    "prediction_um",
    "residual_um",
    "absolute_residual_um",
    "repeatability_normalized_absolute_residual",
)
FOLD_METRIC_FIELDS = (
    "fold_id",
    "held_out_setting_id",
    "variant_id",
    "target_mean_column",
    "n_test_rows",
    "rmse_um",
    "mae_um",
    "nrmse_train_std",
    "mean_observed_repeatability_std_um",
    "positive_repeatability_count",
    "median_repeatability_normalized_absolute_residual",
)
POOLED_METRIC_FIELDS = (
    "variant_id",
    "target_mean_column",
    "n_test_rows",
    "rmse_um",
    "mae_um",
    "mean_observed_repeatability_std_um",
    "positive_repeatability_count",
    "median_repeatability_normalized_absolute_residual",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _phase197_ready(phase197: dict[str, Any]) -> bool:
    gate = phase197.get("gate", {})
    return (
        gate.get("status") == "phase197_calibration_table_design_ready_phase198_baseline_contract_design"
        and bool(gate.get("phase198_baseline_contract_design_allowed"))
        and gate.get("calibration_fitting_allowed") is False
        and gate.get("model_training_allowed") is False
    )


def _phase198_ready(phase198: dict[str, Any]) -> bool:
    gate = phase198.get("gate", {})
    return (
        gate.get("status") == "phase198_baseline_contract_ready_phase199_fixed_baseline_execution"
        and bool(gate.get("phase199_fixed_baseline_execution_allowed"))
        and gate.get("calibration_fitting_allowed") is False
        and gate.get("model_training_allowed") is False
    )


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> dict[str, np.ndarray | float]:
    if alpha < 0.0:
        raise ValueError("Ridge alpha must be non-negative")
    x_mean = x_train.mean(axis=0)
    x_scale = x_train.std(axis=0)
    x_scale[x_scale <= EPSILON] = 1.0
    y_mean = float(y_train.mean())
    y_scale = float(y_train.std())
    if y_scale <= EPSILON:
        y_scale = 1.0
    x_standard = (x_train - x_mean) / x_scale
    y_standard = (y_train - y_mean) / y_scale
    weights = np.linalg.solve(
        x_standard.T @ x_standard + np.eye(x_standard.shape[1], dtype=np.float64) * alpha,
        x_standard.T @ y_standard,
    )
    return {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "weights": weights,
    }


def predict_ridge(model: dict[str, np.ndarray | float], features: np.ndarray) -> np.ndarray:
    x_mean = np.asarray(model["x_mean"], dtype=np.float64)
    x_scale = np.asarray(model["x_scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    return ((features - x_mean) / x_scale @ weights * float(model["y_scale"]) + float(model["y_mean"])).astype(
        np.float64, copy=False
    )


def _split_csv_values(value: str) -> list[str]:
    return [] if not value else [part for part in value.split(";") if part]


def _variant_feature_columns(variant: dict[str, Any]) -> list[str]:
    value = str(variant.get("feature_columns", ""))
    return _split_csv_values(value)


def _metric_values(
    observed: np.ndarray, prediction: np.ndarray, repeatability_std: np.ndarray, train_std: float | None
) -> dict[str, Any]:
    residual = prediction - observed
    absolute_residual = np.abs(residual)
    positive_repeatability = repeatability_std > 0.0
    normalized = absolute_residual[positive_repeatability] / repeatability_std[positive_repeatability]
    return {
        "rmse_um": float(math.sqrt(np.mean(residual**2))),
        "mae_um": float(np.mean(absolute_residual)),
        "nrmse_train_std": (
            float(math.sqrt(np.mean(residual**2)) / train_std) if train_std is not None and train_std > EPSILON else None
        ),
        "mean_observed_repeatability_std_um": float(np.mean(repeatability_std)),
        "positive_repeatability_count": int(np.sum(positive_repeatability)),
        "median_repeatability_normalized_absolute_residual": float(np.median(normalized)) if normalized.size else None,
    }


def _shuffled_training_features(features: np.ndarray, feature_columns: list[str], seed: int) -> np.ndarray:
    output = features.copy()
    raw_indices = [feature_columns.index(name) for name in RAW_THERMAL_FEATURES if name in feature_columns]
    if len(raw_indices) != len(RAW_THERMAL_FEATURES):
        raise ValueError("The shuffled descriptor control must include every raw-thermal descriptor")
    permutation = np.random.default_rng(seed).permutation(len(output))
    output[:, raw_indices] = output[permutation][:, raw_indices]
    return output


def evaluate_contract(
    calibration_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    phase198: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    table_by_group = {str(row["thermal_group_id"]): row for row in calibration_rows}
    if len(table_by_group) != len(calibration_rows):
        raise ValueError("Calibration table contains duplicate thermal_group_id values")
    variant_rows = list(phase198.get("baseline_contract", []))
    targets = list(phase198.get("targets", []))
    ridge_alpha = float(phase198["ridge_alpha"])
    shuffle_seed_base = int(phase198["shuffle_seed_base"])
    all_group_ids = set(table_by_group)
    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    audit_fold_rows: list[dict[str, Any]] = []

    for fold_index, fold in enumerate(sorted(fold_rows, key=lambda row: str(row["fold_id"]))):
        fold_id = str(fold["fold_id"])
        held_out_setting_id = str(fold["held_out_setting_id"])
        held_out_ids = _split_csv_values(str(fold["held_out_thermal_group_ids"]))
        if len(held_out_ids) != EXPECTED_CONDITIONS_PER_SETTING or any(group_id not in table_by_group for group_id in held_out_ids):
            raise ValueError(f"Invalid held-out IDs for {fold_id}")
        test_rows = [table_by_group[group_id] for group_id in held_out_ids]
        train_ids = sorted(all_group_ids - set(held_out_ids))
        train_rows = [table_by_group[group_id] for group_id in train_ids]
        if any(str(row["setting_id"]) != held_out_setting_id for row in test_rows):
            raise ValueError(f"Held-out setting mismatch for {fold_id}")
        audit_fold_rows.append(
            {
                "fold_id": fold_id,
                "held_out_setting_id": held_out_setting_id,
                "train_row_count": len(train_rows),
                "test_row_count": len(test_rows),
                "shuffle_seed": shuffle_seed_base + fold_index,
            }
        )
        for target in targets:
            mean_column = str(target["target_mean_column"])
            std_column = str(target["repeatability_std_column"])
            y_train = np.asarray([float(row[mean_column]) for row in train_rows], dtype=np.float64)
            y_test = np.asarray([float(row[mean_column]) for row in test_rows], dtype=np.float64)
            std_test = np.asarray([float(row[std_column]) for row in test_rows], dtype=np.float64)
            train_std = float(y_train.std())
            for variant in variant_rows:
                variant_id = str(variant["variant_id"])
                feature_columns = _variant_feature_columns(variant)
                if variant_id == "train_mean_control":
                    prediction = np.full(len(test_rows), float(y_train.mean()), dtype=np.float64)
                else:
                    x_train = np.asarray(
                        [[float(row[column]) for column in feature_columns] for row in train_rows], dtype=np.float64
                    )
                    x_test = np.asarray(
                        [[float(row[column]) for column in feature_columns] for row in test_rows], dtype=np.float64
                    )
                    if variant_id == "shuffled_raw_thermal_negative_control":
                        x_train = _shuffled_training_features(x_train, feature_columns, shuffle_seed_base + fold_index)
                    model = fit_ridge(x_train, y_train, ridge_alpha)
                    prediction = predict_ridge(model, x_test)
                residual = prediction - y_test
                absolute_residual = np.abs(residual)
                for row_index, row in enumerate(test_rows):
                    repeatability_std = float(std_test[row_index])
                    prediction_rows.append(
                        {
                            "fold_id": fold_id,
                            "held_out_setting_id": held_out_setting_id,
                            "variant_id": variant_id,
                            "target_mean_column": mean_column,
                            "target_repeatability_std_column": std_column,
                            "thermal_group_id": str(row["thermal_group_id"]),
                            "observed_mean_um": float(y_test[row_index]),
                            "observed_repeatability_std_um": repeatability_std,
                            "prediction_um": float(prediction[row_index]),
                            "residual_um": float(residual[row_index]),
                            "absolute_residual_um": float(absolute_residual[row_index]),
                            "repeatability_normalized_absolute_residual": (
                                float(absolute_residual[row_index] / repeatability_std)
                                if repeatability_std > 0.0
                                else None
                            ),
                        }
                    )
                fold_metric_rows.append(
                    {
                        "fold_id": fold_id,
                        "held_out_setting_id": held_out_setting_id,
                        "variant_id": variant_id,
                        "target_mean_column": mean_column,
                        "n_test_rows": len(test_rows),
                        **_metric_values(y_test, prediction, std_test, train_std),
                    }
                )

    pooled_metric_rows: list[dict[str, Any]] = []
    for target in targets:
        mean_column = str(target["target_mean_column"])
        for variant in variant_rows:
            variant_id = str(variant["variant_id"])
            rows = [
                row
                for row in prediction_rows
                if row["target_mean_column"] == mean_column and row["variant_id"] == variant_id
            ]
            observed = np.asarray([float(row["observed_mean_um"]) for row in rows], dtype=np.float64)
            prediction = np.asarray([float(row["prediction_um"]) for row in rows], dtype=np.float64)
            repeatability_std = np.asarray(
                [float(row["observed_repeatability_std_um"]) for row in rows], dtype=np.float64
            )
            metrics = _metric_values(observed, prediction, repeatability_std, None)
            metrics.pop("nrmse_train_std")
            pooled_metric_rows.append(
                {
                    "variant_id": variant_id,
                    "target_mean_column": mean_column,
                    "n_test_rows": len(rows),
                    **metrics,
                }
            )
    audit = {
        "calibration_table_row_count": len(calibration_rows),
        "fold_count": len(fold_rows),
        "variant_ids": [str(row["variant_id"]) for row in variant_rows],
        "target_mean_columns": [str(row["target_mean_column"]) for row in targets],
        "prediction_row_count": len(prediction_rows),
        "fold_metric_row_count": len(fold_metric_rows),
        "pooled_metric_row_count": len(pooled_metric_rows),
        "fold_execution": audit_fold_rows,
        "feature_normalization_fit_scope": "Training rows only inside each held-out-setting fold.",
        "target_scaling_fit_scope": "Training rows only inside each held-out-setting fold.",
        "hyperparameter_search_performed": False,
        "post_holdout_model_selection_performed": False,
        "neural_model_training_performed": False,
        "raw_hdf5_reopened": False,
        "raw_workbook_reopened": False,
    }
    return prediction_rows, fold_metric_rows, pooled_metric_rows, audit


def build_gate(
    phase197: dict[str, Any],
    phase198: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
    fold_metric_rows: list[dict[str, Any]],
    pooled_metric_rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    if not _phase197_ready(phase197):
        blockers.append("phase197_calibration_table_gate_not_ready")
    if not _phase198_ready(phase198):
        blockers.append("phase198_baseline_contract_gate_not_ready")
    variant_ids = [str(value) for value in audit.get("variant_ids", [])]
    target_columns = [str(value) for value in audit.get("target_mean_columns", [])]
    expected_prediction_count = (
        EXPECTED_CONDITION_COUNT * len(variant_ids) * len(target_columns)
    )
    expected_fold_metric_count = EXPECTED_PROCESS_SETTING_COUNT * len(variant_ids) * len(target_columns)
    expected_pooled_metric_count = len(variant_ids) * len(target_columns)
    if int(audit.get("calibration_table_row_count", 0)) != EXPECTED_CONDITION_COUNT:
        blockers.append("unexpected_calibration_table_row_count")
    if int(audit.get("fold_count", 0)) != EXPECTED_PROCESS_SETTING_COUNT:
        blockers.append("unexpected_fold_count")
    if int(audit.get("prediction_row_count", 0)) != expected_prediction_count:
        blockers.append("prediction_row_count_contract_broken")
    if int(audit.get("fold_metric_row_count", 0)) != expected_fold_metric_count:
        blockers.append("fold_metric_row_count_contract_broken")
    if int(audit.get("pooled_metric_row_count", 0)) != expected_pooled_metric_count:
        blockers.append("pooled_metric_row_count_contract_broken")
    keys = {
        (
            str(row["fold_id"]),
            str(row["variant_id"]),
            str(row["target_mean_column"]),
            str(row["thermal_group_id"]),
        )
        for row in prediction_rows
    }
    if len(keys) != len(prediction_rows):
        blockers.append("duplicate_prediction_key")
    if any(not math.isfinite(float(row["prediction_um"])) for row in prediction_rows):
        blockers.append("nonfinite_prediction")
    if any(not math.isfinite(float(row["rmse_um"])) for row in fold_metric_rows + pooled_metric_rows):
        blockers.append("nonfinite_metric")
    if audit.get("feature_normalization_fit_scope") != "Training rows only inside each held-out-setting fold.":
        blockers.append("feature_normalization_scope_broken")
    if audit.get("target_scaling_fit_scope") != "Training rows only inside each held-out-setting fold.":
        blockers.append("target_scaling_scope_broken")
    if audit.get("hyperparameter_search_performed") is not False:
        blockers.append("hyperparameter_search_detected")
    if audit.get("post_holdout_model_selection_performed") is not False:
        blockers.append("post_holdout_model_selection_detected")
    if audit.get("neural_model_training_performed") is not False:
        blockers.append("neural_model_training_detected")
    if audit.get("raw_hdf5_reopened") is not False or audit.get("raw_workbook_reopened") is not False:
        blockers.append("raw_source_boundary_broken")
    blockers = sorted(set(blockers))
    ready = not blockers
    return {
        "status": (
            "phase199_fixed_baseline_execution_ready_phase200_residual_audit"
            if ready
            else "phase199_fixed_baseline_execution_incomplete_or_leaky"
        ),
        "phase200_residual_audit_allowed": ready,
        "calibration_fitting_allowed": False,
        "model_training_allowed": False,
        "hyperparameter_search_allowed": False,
        "post_b8_model_reselection_allowed": False,
        "descriptor_target_leakage_allowed": False,
        "simulation_as_external_validation_allowed": False,
        "blocking_audits": blockers,
        "next_action": (
            "audit all preregistered residual patterns and repeatability-relative errors without selecting a winner"
            if ready
            else "repair the fixed baseline execution or leakage audit before interpreting metrics"
        ),
    }


def build_payload(
    phase197: dict[str, Any],
    phase198: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
    fold_metric_rows: list[dict[str, Any]],
    pooled_metric_rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": 199,
        "objective": "fixed_low_capacity_grouped_baseline_execution",
        "interpretation_boundary": (
            "All preregistered variants are reported. Held-out setting metrics are not used for variant selection, and "
            "cross-section repeatability is not a predictive interval or external validation."
        ),
        "fold_metrics": fold_metric_rows,
        "pooled_metrics": pooled_metric_rows,
        "execution_audit": audit,
        "gate": build_gate(phase197, phase198, prediction_rows, fold_metric_rows, pooled_metric_rows, audit),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase197", type=Path, default=DEFAULT_PHASE197)
    parser.add_argument("--phase198", type=Path, default=DEFAULT_PHASE198)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--fold-metrics-csv", type=Path, required=True)
    parser.add_argument("--pooled-metrics-csv", type=Path, required=True)
    args = parser.parse_args()
    phase197 = _read_json(args.phase197)
    phase198 = _read_json(args.phase198)
    prediction_rows, fold_metric_rows, pooled_metric_rows, audit = evaluate_contract(
        _read_csv(args.table), _read_csv(args.folds), phase198
    )
    payload = build_payload(phase197, phase198, prediction_rows, fold_metric_rows, pooled_metric_rows, audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.predictions_csv, prediction_rows, PREDICTION_FIELDS)
    _write_csv(args.fold_metrics_csv, fold_metric_rows, FOLD_METRIC_FIELDS)
    _write_csv(args.pooled_metrics_csv, pooled_metric_rows, POOLED_METRIC_FIELDS)
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
