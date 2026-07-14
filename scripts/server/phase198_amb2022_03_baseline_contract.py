#!/usr/bin/env python3
"""Freeze the no-selection baseline and repeatability-aware report contract for AMB2022-03."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_PHASE197 = Path(
    os.environ.get(
        "AMB2022_03_PHASE197_CALIBRATION_TABLE",
        "/root/matsci-gnn-pinn-ops/phase197_amb2022_03_calibration_table_design.json",
    )
)
RIDGE_ALPHA = 1.0
SHUFFLE_SEED_BASE = 1981
PROCESS_FEATURES = ("laser_power_W", "scan_speed_mm_s", "spot_size_um")
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
TARGETS = (
    ("depth_um_mean", "depth_um_std", "depth_um"),
    ("width_um_mean", "width_um_std", "width_um"),
)
VARIANT_FIELDS = (
    "variant_id",
    "family",
    "feature_columns",
    "uses_process_metadata",
    "uses_raw_thermal_descriptors",
    "training_descriptor_permutation",
    "ridge_alpha",
    "fit_scope",
    "evaluation_scope",
    "selection_policy",
    "purpose",
)
REPEATABILITY_REPORT_FIELDS = (
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _phase197_ready(phase197: dict[str, Any]) -> bool:
    gate = phase197.get("gate", {})
    return (
        gate.get("status") == "phase197_calibration_table_design_ready_phase198_baseline_contract_design"
        and bool(gate.get("phase198_baseline_contract_design_allowed"))
        and gate.get("calibration_fitting_allowed") is False
        and gate.get("model_training_allowed") is False
    )


def build_variant_rows() -> list[dict[str, Any]]:
    all_features = PROCESS_FEATURES + RAW_THERMAL_FEATURES
    fit_scope = (
        "For every held-out process setting, fit all feature scaling and target centering/scaling on the remaining six "
        "settings only."
    )
    evaluation_scope = "Predict all three conditions in each held-out process setting across all seven fixed folds."
    selection_policy = "Report every preregistered variant; do not select or retune from held-out fold metrics."
    return [
        {
            "variant_id": "train_mean_control",
            "family": "constant_control",
            "feature_columns": "",
            "uses_process_metadata": False,
            "uses_raw_thermal_descriptors": False,
            "training_descriptor_permutation": "none",
            "ridge_alpha": "",
            "fit_scope": "Compute each target training-fold mean only.",
            "evaluation_scope": evaluation_scope,
            "selection_policy": selection_policy,
            "purpose": "No-feature lower reference.",
        },
        {
            "variant_id": "process_ridge_control",
            "family": "ridge_control",
            "feature_columns": ";".join(PROCESS_FEATURES),
            "uses_process_metadata": True,
            "uses_raw_thermal_descriptors": False,
            "training_descriptor_permutation": "none",
            "ridge_alpha": RIDGE_ALPHA,
            "fit_scope": fit_scope,
            "evaluation_scope": evaluation_scope,
            "selection_policy": selection_policy,
            "purpose": "Nominal process-parameter reference without thermal observations.",
        },
        {
            "variant_id": "raw_thermal_ridge",
            "family": "ridge_control",
            "feature_columns": ";".join(RAW_THERMAL_FEATURES),
            "uses_process_metadata": False,
            "uses_raw_thermal_descriptors": True,
            "training_descriptor_permutation": "none",
            "ridge_alpha": RIDGE_ALPHA,
            "fit_scope": fit_scope,
            "evaluation_scope": evaluation_scope,
            "selection_policy": selection_policy,
            "purpose": "Thermography-only linear reference.",
        },
        {
            "variant_id": "process_plus_raw_thermal_ridge",
            "family": "ridge_candidate",
            "feature_columns": ";".join(all_features),
            "uses_process_metadata": True,
            "uses_raw_thermal_descriptors": True,
            "training_descriptor_permutation": "none",
            "ridge_alpha": RIDGE_ALPHA,
            "fit_scope": fit_scope,
            "evaluation_scope": evaluation_scope,
            "selection_policy": selection_policy,
            "purpose": "Primary additive process-plus-observation association test.",
        },
        {
            "variant_id": "shuffled_raw_thermal_negative_control",
            "family": "negative_control",
            "feature_columns": ";".join(all_features),
            "uses_process_metadata": True,
            "uses_raw_thermal_descriptors": True,
            "training_descriptor_permutation": (
                "Jointly permute the eight raw-thermal columns across training rows only with seed "
                "SHUFFLE_SEED_BASE plus sorted-fold index; leave process columns, targets, and held-out rows unchanged."
            ),
            "ridge_alpha": RIDGE_ALPHA,
            "fit_scope": fit_scope,
            "evaluation_scope": evaluation_scope,
            "selection_policy": selection_policy,
            "purpose": "Preserve descriptor marginals while breaking training-row descriptor-to-target alignment.",
        },
    ]


def build_gate(phase197: dict[str, Any], variant_rows: list[dict[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    if not _phase197_ready(phase197):
        blockers.append("phase197_calibration_table_gate_not_ready")
    expected_ids = {
        "train_mean_control",
        "process_ridge_control",
        "raw_thermal_ridge",
        "process_plus_raw_thermal_ridge",
        "shuffled_raw_thermal_negative_control",
    }
    by_id = {str(row.get("variant_id", "")): row for row in variant_rows}
    if set(by_id) != expected_ids or len(by_id) != len(variant_rows):
        blockers.append("baseline_variant_contract_broken")
    ridge_rows = [row for row in variant_rows if row.get("variant_id") != "train_mean_control"]
    if any(float(row.get("ridge_alpha", -1.0)) != RIDGE_ALPHA for row in ridge_rows):
        blockers.append("ridge_alpha_not_fixed")
    if any("held-out" not in str(row.get("evaluation_scope", "")) for row in variant_rows):
        blockers.append("heldout_evaluation_scope_missing")
    if any("do not select or retune" not in str(row.get("selection_policy", "")) for row in variant_rows):
        blockers.append("post_holdout_selection_not_blocked")
    shuffled = by_id.get("shuffled_raw_thermal_negative_control", {})
    if "training rows only" not in str(shuffled.get("training_descriptor_permutation", "")):
        blockers.append("negative_control_shuffle_boundary_broken")
    primary = by_id.get("process_plus_raw_thermal_ridge", {})
    if primary.get("feature_columns") != ";".join(PROCESS_FEATURES + RAW_THERMAL_FEATURES):
        blockers.append("primary_feature_contract_broken")
    blockers = sorted(set(blockers))
    ready = not blockers
    return {
        "status": (
            "phase198_baseline_contract_ready_phase199_fixed_baseline_execution"
            if ready
            else "phase198_baseline_contract_incomplete_or_reselectable"
        ),
        "phase199_fixed_baseline_execution_allowed": ready,
        "calibration_fitting_allowed": False,
        "model_training_allowed": False,
        "hyperparameter_search_allowed": False,
        "post_b8_model_reselection_allowed": False,
        "descriptor_target_leakage_allowed": False,
        "simulation_as_external_validation_allowed": False,
        "blocking_audits": blockers,
        "next_action": (
            "run every fixed ridge/control variant across the seven leave-one-setting-out folds and report repeatability-aware residuals"
            if ready
            else "repair the no-selection low-capacity baseline contract before fitting"
        ),
    }


def build_payload(phase197: dict[str, Any]) -> dict[str, Any]:
    variant_rows = build_variant_rows()
    return {
        "phase": 198,
        "objective": "fixed_low_capacity_leave_one_setting_out_baseline_and_repeatability_report_contract",
        "targets": [
            {
                "target_mean_column": mean_column,
                "repeatability_std_column": std_column,
                "units": units,
            }
            for mean_column, std_column, units in TARGETS
        ],
        "ridge_alpha": RIDGE_ALPHA,
        "shuffle_seed_base": SHUFFLE_SEED_BASE,
        "repeatability_report_fields": list(REPEATABILITY_REPORT_FIELDS),
        "repeatability_boundary": (
            "The duplicate cross-section standard deviation is reported as measurement repeatability, not a predictive "
            "interval or an external-validation substitute. Repeatability-normalized residuals are emitted only when "
            "the observed repeatability standard deviation is positive."
        ),
        "baseline_contract": variant_rows,
        "gate": build_gate(phase197, variant_rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=VARIANT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase197", type=Path, default=DEFAULT_PHASE197)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract-csv", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(_read_json(args.phase197))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.contract_csv, payload["baseline_contract"])
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
