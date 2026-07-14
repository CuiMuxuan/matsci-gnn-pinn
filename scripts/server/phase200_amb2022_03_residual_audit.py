#!/usr/bin/env python3
"""Audit fixed AMB2022-03 baseline residuals without selecting a model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_PHASE199 = Path(
    os.environ.get(
        "AMB2022_03_PHASE199_FIXED_BASELINES",
        "/root/matsci-gnn-pinn-ops/phase199_amb2022_03_fixed_baseline_execution.json",
    )
)
TARGET_COLUMNS = ("depth_um_mean", "width_um_mean")
VARIANT_IDS = (
    "train_mean_control",
    "process_ridge_control",
    "raw_thermal_ridge",
    "process_plus_raw_thermal_ridge",
    "shuffled_raw_thermal_negative_control",
)
COMPARISON_VARIANT_IDS = (
    "raw_thermal_ridge",
    "process_plus_raw_thermal_ridge",
    "shuffled_raw_thermal_negative_control",
)
REFERENCE_VARIANT_ID = "process_ridge_control"
EXPECTED_FOLD_COUNT = 7


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _phase199_ready(phase199: dict[str, Any]) -> bool:
    gate = phase199.get("gate", {})
    return (
        gate.get("status") == "phase199_fixed_baseline_execution_ready_phase200_residual_audit"
        and bool(gate.get("phase200_residual_audit_allowed"))
        and gate.get("calibration_fitting_allowed") is False
        and gate.get("model_training_allowed") is False
    )


def _metric_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["fold_id"]), str(row["target_mean_column"]), str(row["variant_id"]))


def _pooled_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row["target_mean_column"]), str(row["variant_id"]))


def build_audit(fold_metrics: list[dict[str, Any]], pooled_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    fold_index = {_metric_key(row): row for row in fold_metrics}
    pooled_index = {_pooled_key(row): row for row in pooled_metrics}
    fold_ids = sorted({str(row["fold_id"]) for row in fold_metrics})
    expected_fold_keys = {
        (fold_id, target, variant)
        for fold_id in fold_ids
        for target in TARGET_COLUMNS
        for variant in VARIANT_IDS
    }
    expected_pooled_keys = {(target, variant) for target in TARGET_COLUMNS for variant in VARIANT_IDS}
    metric_contract_complete = (
        len(fold_ids) == EXPECTED_FOLD_COUNT
        and set(fold_index) == expected_fold_keys
        and len(fold_index) == len(fold_metrics)
        and set(pooled_index) == expected_pooled_keys
        and len(pooled_index) == len(pooled_metrics)
    )
    contrasts: dict[str, dict[str, Any]] = {}
    additive_support_by_target: dict[str, bool] = {}
    if metric_contract_complete:
        for target in TARGET_COLUMNS:
            reference = pooled_index[(target, REFERENCE_VARIANT_ID)]
            target_contrasts: dict[str, Any] = {}
            for variant in COMPARISON_VARIANT_IDS:
                comparison = pooled_index[(target, variant)]
                fold_deltas = [
                    {
                        "fold_id": fold_id,
                        "rmse_delta_vs_process_um": float(fold_index[(fold_id, target, variant)]["rmse_um"])
                        - float(fold_index[(fold_id, target, REFERENCE_VARIANT_ID)]["rmse_um"]),
                    }
                    for fold_id in fold_ids
                ]
                improvement_count = sum(row["rmse_delta_vs_process_um"] < 0.0 for row in fold_deltas)
                target_contrasts[variant] = {
                    "pooled_rmse_delta_vs_process_um": float(comparison["rmse_um"]) - float(reference["rmse_um"]),
                    "pooled_mae_delta_vs_process_um": float(comparison["mae_um"]) - float(reference["mae_um"]),
                    "held_out_fold_improvement_count_vs_process": improvement_count,
                    "held_out_fold_count": len(fold_deltas),
                    "fold_rmse_deltas_vs_process_um": fold_deltas,
                }
            primary = target_contrasts["process_plus_raw_thermal_ridge"]
            additive_support_by_target[target] = bool(
                primary["pooled_rmse_delta_vs_process_um"] <= 0.0
                and primary["held_out_fold_improvement_count_vs_process"] >= 5
            )
            contrasts[target] = target_contrasts
    thermal_additive_signal_supported = bool(additive_support_by_target) and all(additive_support_by_target.values())
    repeatability_values = [
        float(row["median_repeatability_normalized_absolute_residual"])
        for row in pooled_metrics
        if row.get("median_repeatability_normalized_absolute_residual") is not None
    ]
    return {
        "fold_ids": fold_ids,
        "fold_metric_row_count": len(fold_metrics),
        "pooled_metric_row_count": len(pooled_metrics),
        "metric_contract_complete": metric_contract_complete,
        "contrasts_vs_process_ridge": contrasts,
        "additive_thermal_support_by_target": additive_support_by_target,
        "thermal_additive_signal_supported_across_both_targets": thermal_additive_signal_supported,
        "all_pooled_median_repeatability_normalized_residuals_exceed_one": bool(repeatability_values)
        and all(value > 1.0 for value in repeatability_values),
        "post_holdout_model_selection_performed": False,
        "neural_model_escalation_performed": False,
    }


def build_gate(phase199: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if not _phase199_ready(phase199):
        blockers.append("phase199_fixed_baseline_gate_not_ready")
    if audit.get("metric_contract_complete") is not True:
        blockers.append("fixed_metric_contract_incomplete")
    if audit.get("post_holdout_model_selection_performed") is not False:
        blockers.append("post_holdout_model_selection_detected")
    if audit.get("neural_model_escalation_performed") is not False:
        blockers.append("neural_model_escalation_detected")
    blockers = sorted(set(blockers))
    ready = not blockers
    thermal_signal = bool(audit.get("thermal_additive_signal_supported_across_both_targets"))
    return {
        "status": (
            "phase200_residual_audit_complete_additive_signal_not_escalated"
            if ready and thermal_signal
            else "phase200_residual_audit_complete_no_robust_additive_thermal_signal"
            if ready
            else "phase200_residual_audit_incomplete_or_reselectable"
        ),
        "phase201_mechanistic_stress_test_design_allowed": ready,
        "thermal_descriptor_model_escalation_allowed": False,
        "calibration_fitting_allowed": False,
        "model_training_allowed": False,
        "hyperparameter_search_allowed": False,
        "post_b8_model_reselection_allowed": False,
        "simulation_as_external_validation_allowed": False,
        "blocking_audits": blockers,
        "next_action": (
            "design a mechanism-focused stress test; do not add model capacity or claim descriptor-driven geometry prediction"
            if ready
            else "repair the fixed residual audit before interpreting baseline contrasts"
        ),
    }


def build_payload(phase199: dict[str, Any]) -> dict[str, Any]:
    audit = build_audit(phase199.get("fold_metrics", []), phase199.get("pooled_metrics", []))
    return {
        "phase": 200,
        "objective": "fixed_baseline_residual_and_repeatability_audit_without_model_selection",
        "contrast_policy": (
            "Compare every pre-registered thermal variant to process_ridge_control descriptively. A robust additive signal "
            "requires non-worse pooled RMSE and improvement in at least five of seven held-out settings for each target; "
            "this threshold is not a model-selection rule."
        ),
        "interpretation_boundary": (
            "Duplicate-section standard deviation is measurement repeatability, not a predictive interval. Failure to show "
            "a robust linear descriptor gain does not justify a larger model or an external-validation claim."
        ),
        "residual_audit": audit,
        "gate": build_gate(phase199, audit),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase199", type=Path, default=DEFAULT_PHASE199)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(_read_json(args.phase199))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
