#!/usr/bin/env python3
"""Freeze the AMB2022-01 layer-space ablation contract before model fitting."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_PHASE184 = Path(
    os.environ.get(
        "AMB2022_01_PHASE184_BASELINES",
        "/root/matsci-gnn-pinn-ops/phase184_layer_space_ridge_baselines.json",
    )
)

TARGETS = ("tam_s", "scr_C_per_s")
REQUIRED_SPLITS = ("train", "val", "test")
ABLATION_FIELDS = (
    "variant_id",
    "family",
    "uses_coordinates",
    "uses_scan_history",
    "uses_physics_kernel",
    "uses_neural_network",
    "is_control",
    "negative_control",
    "fit_scope",
    "evaluation_scope",
    "required_seeds",
    "purpose",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_lookup(metrics: list[dict[str, Any]], target: str, profile: str, split: str) -> dict[str, Any]:
    matches = [
        row
        for row in metrics
        if row.get("target") == target
        and row.get("model") == "ridge"
        and row.get("profile") == profile
        and row.get("split") == split
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one metric for {target}/{profile}/{split}, got {len(matches)}")
    return matches[0]


def build_ablation_rows() -> list[dict[str, Any]]:
    seeds = "1841,1842,1843"
    return [
        {
            "variant_id": "train_mean_control",
            "family": "baseline",
            "uses_coordinates": False,
            "uses_scan_history": False,
            "uses_physics_kernel": False,
            "uses_neural_network": False,
            "is_control": True,
            "negative_control": False,
            "fit_scope": "B6 train target mean only",
            "evaluation_scope": "B7 validation and B8 held build",
            "required_seeds": "deterministic",
            "purpose": "irreducible no-feature baseline",
        },
        {
            "variant_id": "coordinate_layer_ridge_control",
            "family": "baseline",
            "uses_coordinates": True,
            "uses_scan_history": False,
            "uses_physics_kernel": False,
            "uses_neural_network": False,
            "is_control": True,
            "negative_control": False,
            "fit_scope": "B6-only standardization and ridge fit",
            "evaluation_scope": "B7 validation and B8 held build",
            "required_seeds": "deterministic",
            "purpose": "spatial/layer field control without scan history",
        },
        {
            "variant_id": "scan_history_ridge_control",
            "family": "baseline",
            "uses_coordinates": True,
            "uses_scan_history": True,
            "uses_physics_kernel": False,
            "uses_neural_network": False,
            "is_control": True,
            "negative_control": False,
            "fit_scope": "B6-only standardization and ridge fit",
            "evaluation_scope": "B7 validation and B8 held build",
            "required_seeds": "deterministic",
            "purpose": "linear scan-history benefit reference",
        },
        {
            "variant_id": "layerwise_shuffled_scan_history_control",
            "family": "negative_control",
            "uses_coordinates": True,
            "uses_scan_history": True,
            "uses_physics_kernel": False,
            "uses_neural_network": False,
            "is_control": True,
            "negative_control": True,
            "fit_scope": "shuffle scan-history columns within each build/layer using fixed seeds",
            "evaluation_scope": "B7 validation and B8 held build",
            "required_seeds": seeds,
            "purpose": "preserve feature marginals while breaking spatial scan-history alignment",
        },
        {
            "variant_id": "heat_kernel_history_ridge",
            "family": "physics_feature",
            "uses_coordinates": True,
            "uses_scan_history": True,
            "uses_physics_kernel": True,
            "uses_neural_network": False,
            "is_control": False,
            "negative_control": False,
            "fit_scope": "B6-only kernel-scale selection and ridge fit",
            "evaluation_scope": "B7 validation and B8 held build",
            "required_seeds": "deterministic kernel grid",
            "purpose": "interpretable heat-diffusion history feature control",
        },
        {
            "variant_id": "small_data_only_mlp",
            "family": "neural_control",
            "uses_coordinates": True,
            "uses_scan_history": True,
            "uses_physics_kernel": False,
            "uses_neural_network": True,
            "is_control": True,
            "negative_control": False,
            "fit_scope": "B6-only normalization, early stopping on B7",
            "evaluation_scope": "B8 held build",
            "required_seeds": seeds,
            "purpose": "capacity-matched nonlinear data-only control",
        },
        {
            "variant_id": "physics_regularized_history_mlp",
            "family": "candidate",
            "uses_coordinates": True,
            "uses_scan_history": True,
            "uses_physics_kernel": True,
            "uses_neural_network": True,
            "is_control": False,
            "negative_control": False,
            "fit_scope": "B6-only normalization and parameter fitting; B7 selection only",
            "evaluation_scope": "B8 held build plus paired B6/B7/B8 residual analysis",
            "required_seeds": seeds,
            "purpose": "candidate interpretable scan-history surrogate",
        },
    ]


def build_gate(phase184: dict[str, Any]) -> dict[str, Any]:
    gate = phase184.get("gate", {})
    blockers: list[str] = []
    expected_status = "phase184_low_capacity_baselines_ready_phase185_ablation_contract"
    if gate.get("status") != expected_status or not gate.get("phase185_ablation_contract_allowed"):
        blockers.append("phase184_baseline_gate_not_ready")
    gains = gate.get("scan_feature_gains_vs_coordinate", {})
    for target in TARGETS:
        target_gains = gains.get(target, {})
        if float(target_gains.get("validation_rmse_gain_vs_coordinate", 0.0)) <= 0.0:
            blockers.append(f"{target}_no_validation_scan_history_gain")
        if float(target_gains.get("test_rmse_gain_vs_coordinate", 0.0)) <= 0.0:
            blockers.append(f"{target}_no_test_scan_history_gain")
    ready = not blockers
    return {
        "status": (
            "phase185_ablation_contract_ready_phase186_feature_ablation_execution"
            if ready
            else "phase185_ablation_contract_blocked"
        ),
        "phase186_feature_ablation_execution_allowed": ready,
        "model_training_allowed": False,
        "a800_training_allowed_now": False,
        "blocking_audits": blockers,
        "next_action": (
            "materialize heat-kernel and shuffled-history controls under the fixed contract"
            if ready
            else "do not escalate to candidate model fitting until scan-history gains are stable"
        ),
    }


def run(phase184_path: Path) -> dict[str, Any]:
    phase184 = _read_json(phase184_path)
    metric_rows = phase184.get("metrics", [])
    comparison = {
        target: {
            split: {
                "coordinate_layer_rmse": float(_metric_lookup(metric_rows, target, "coordinate_layer", split)["rmse"]),
                "scan_history_rmse": float(_metric_lookup(metric_rows, target, "scan_history", split)["rmse"]),
                "full_ridge_rmse": float(_metric_lookup(metric_rows, target, "full", split)["rmse"]),
            }
            for split in REQUIRED_SPLITS
        }
        for target in TARGETS
    }
    return {
        "phase": 185,
        "objective": "fixed_replicate_aware_ablation_contract",
        "phase184_baseline": str(phase184_path),
        "observed_low_capacity_comparison": comparison,
        "ablation_contract": build_ablation_rows(),
        "gate": build_gate(phase184),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ABLATION_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase184", type=Path, default=DEFAULT_PHASE184)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract-csv", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.phase184)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.contract_csv, payload["ablation_contract"])
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
