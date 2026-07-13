#!/usr/bin/env python3
"""Run leakage-controlled low-capacity baselines on the Phase 182 dataset.

Every fitted statistic and ridge coefficient is estimated only from B6.  B7 is
used for validation and B8 for a held-out build result.  This is deliberately a
CPU-only baseline stage, not a substitute for the later physics-informed model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASET = Path(
    os.environ.get(
        "AMB2022_01_PHASE182_DATASET",
        "/root/matsci-gnn-pinn-data/derived/ambench/2022_3d_build/AMB2022-01/"
        "phase182/phase182_layer_space_dataset.h5",
    )
)
DEFAULT_PHASE183_GATE = Path(
    os.environ.get(
        "AMB2022_01_PHASE183_GATE",
        "/root/matsci-gnn-pinn-ops/phase183_layer_space_data_quality_gate.json",
    )
)
FEATURE_PROFILES = {
    "coordinate_layer": ("x_mm", "y_mm", "z_mm", "layer_fraction"),
    "scan_history": (
        "x_mm",
        "y_mm",
        "z_mm",
        "layer_fraction",
        "laser_active",
        "laser_energy_density_J_mm2",
        "mean_power_W",
        "first_laser_time_s",
        "last_laser_time_s",
        "time_since_last_laser_s",
        "energy_weighted_progress",
        "laser_progress_span",
        "staring_trigger_count",
    ),
    "full": None,
}
TARGETS = {
    "tam_s": "target_tam_s",
    "scr_C_per_s": "target_scr_C_per_s",
}
SPLITS = {"train": 0, "val": 1, "test": 2}


def _h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("h5py is required for Phase 184 ridge baselines") from exc
    return h5py


def _json_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def _read_phase183_gate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate = payload.get("gate", {})
    expected = "phase183_layer_space_data_quality_ready_phase184_baseline_design"
    if gate.get("status") != expected or not gate.get("phase184_baseline_design_allowed"):
        raise ValueError(f"Phase 183 does not permit baseline design: {gate.get('status')!r}")
    return payload


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> dict[str, np.ndarray | float]:
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    x_mean = x_train.mean(axis=0)
    x_scale = x_train.std(axis=0)
    x_scale[x_scale <= 1e-12] = 1.0
    y_mean = float(y_train.mean())
    y_scale = float(y_train.std())
    if y_scale <= 1e-12:
        raise ValueError("Training target has no variance")
    x_standard = (x_train - x_mean) / x_scale
    y_standard = (y_train - y_mean) / y_scale
    gram = x_standard.T @ x_standard
    regularizer = np.eye(x_standard.shape[1], dtype=float) * alpha
    weights = np.linalg.solve(gram + regularizer, x_standard.T @ y_standard)
    return {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "weights": weights,
    }


def predict_ridge(model: dict[str, np.ndarray | float], x: np.ndarray) -> np.ndarray:
    x_mean = np.asarray(model["x_mean"], dtype=float)
    x_scale = np.asarray(model["x_scale"], dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    y_mean = float(model["y_mean"])
    y_scale = float(model["y_scale"])
    return ((x - x_mean) / x_scale @ weights * y_scale + y_mean).astype(np.float64, copy=False)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, train_std: float, train_hot_q90: float) -> dict[str, float]:
    residual = y_true - y_pred
    rmse = float(math.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denominator) if denominator > 0.0 else 0.0
    hot = y_true >= train_hot_q90
    hot_rmse = float(math.sqrt(np.mean(residual[hot] ** 2))) if np.any(hot) else float("nan")
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "nrmse_train_std": rmse / train_std if train_std > 0.0 else float("nan"),
        "hot_q90_rmse": hot_rmse,
    }


def _profile_indices(feature_names: list[str], profile_name: str) -> list[int]:
    selected = FEATURE_PROFILES[profile_name]
    if selected is None:
        return list(range(len(feature_names)))
    return [feature_names.index(name) for name in selected]


def evaluate_baselines(
    *,
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    valid_masks: dict[str, np.ndarray],
    split_codes: np.ndarray,
    feature_names: list[str],
    alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for target_name, target_values in targets.items():
        masks = {name: (split_codes == code) & valid_masks[target_name] for name, code in SPLITS.items()}
        if not all(np.any(mask) for mask in masks.values()):
            raise ValueError(f"Target {target_name} has an empty valid split")
        y_train = target_values[masks["train"]].astype(np.float64)
        train_mean = float(y_train.mean())
        train_std = float(y_train.std())
        train_hot_q90 = float(np.quantile(y_train, 0.9))
        diagnostics[target_name] = {
            "train_rows": int(np.sum(masks["train"])),
            "val_rows": int(np.sum(masks["val"])),
            "test_rows": int(np.sum(masks["test"])),
            "train_mean": train_mean,
            "train_std": train_std,
            "train_hot_q90": train_hot_q90,
        }
        for split_name, mask in masks.items():
            y = target_values[mask].astype(np.float64)
            pred = np.full_like(y, train_mean)
            rows.append(
                {
                    "target": target_name,
                    "model": "train_mean",
                    "profile": "none",
                    "split": split_name,
                    "n_rows": int(len(y)),
                    **metrics(y, pred, train_std, train_hot_q90),
                }
            )
        for profile_name in FEATURE_PROFILES:
            indices = _profile_indices(feature_names, profile_name)
            model = fit_ridge(features[masks["train"]][:, indices].astype(np.float64), y_train, alpha)
            for split_name, mask in masks.items():
                y = target_values[mask].astype(np.float64)
                pred = predict_ridge(model, features[mask][:, indices].astype(np.float64))
                rows.append(
                    {
                        "target": target_name,
                        "model": "ridge",
                        "profile": profile_name,
                        "split": split_name,
                        "n_rows": int(len(y)),
                        "feature_count": len(indices),
                        **metrics(y, pred, train_std, train_hot_q90),
                    }
                )
    return rows, diagnostics


def build_gate(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons: dict[str, dict[str, float]] = {}
    blockers: list[str] = []
    for target_name in TARGETS:
        coordinate = {
            row["split"]: row
            for row in metric_rows
            if row["target"] == target_name and row["model"] == "ridge" and row["profile"] == "coordinate_layer"
        }
        full = {
            row["split"]: row
            for row in metric_rows
            if row["target"] == target_name and row["model"] == "ridge" and row["profile"] == "full"
        }
        if set(coordinate) != set(SPLITS) or set(full) != set(SPLITS):
            blockers.append(f"{target_name}_incomplete_baseline_rows")
            continue
        comparisons[target_name] = {
            "validation_rmse_gain_vs_coordinate": float(coordinate["val"]["rmse"] - full["val"]["rmse"]),
            "test_rmse_gain_vs_coordinate": float(coordinate["test"]["rmse"] - full["test"]["rmse"]),
        }
    completed = not blockers
    return {
        "status": (
            "phase184_low_capacity_baselines_ready_phase185_ablation_contract"
            if completed
            else "phase184_low_capacity_baselines_incomplete"
        ),
        "baseline_execution_complete": completed,
        "phase185_ablation_contract_allowed": completed,
        "model_training_allowed": False,
        "a800_training_allowed_now": False,
        "scan_feature_gains_vs_coordinate": comparisons,
        "blocking_audits": blockers,
        "next_action": (
            "define fixed ablations and a replicate-aware model protocol before any GPU fit"
            if completed
            else "repair incomplete baseline metrics before model design"
        ),
    }


def run(dataset_path: Path, phase183_gate: Path, alpha: float) -> dict[str, Any]:
    h5py = _h5py()
    _read_phase183_gate(phase183_gate)
    with h5py.File(dataset_path, "r") as handle:
        features = handle["features"][...]
        feature_names = list(_json_attr(handle.attrs["feature_names"]))
        valid = handle["target_valid_mask"][...].astype(bool)
        targets = {name: handle[path][...] for name, path in TARGETS.items()}
        valid_masks = {"tam_s": valid[:, 0], "scr_C_per_s": valid[:, 1]}
        split_codes = handle["primary_split"][...]
    if not np.isfinite(features).all():
        raise ValueError("Phase 182 features contain non-finite values")
    metric_rows, diagnostics = evaluate_baselines(
        features=features,
        targets=targets,
        valid_masks=valid_masks,
        split_codes=split_codes,
        feature_names=feature_names,
        alpha=alpha,
    )
    return {
        "phase": 184,
        "objective": "low_capacity_leakage_controlled_baselines",
        "dataset": str(dataset_path),
        "phase183_gate": str(phase183_gate),
        "ridge_alpha": alpha,
        "feature_profiles": {
            name: list(FEATURE_PROFILES[name]) if FEATURE_PROFILES[name] is not None else feature_names
            for name in FEATURE_PROFILES
        },
        "target_diagnostics": diagnostics,
        "metrics": metric_rows,
        "gate": build_gate(metric_rows),
    }


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "target",
        "model",
        "profile",
        "split",
        "n_rows",
        "feature_count",
        "rmse",
        "mae",
        "r2",
        "nrmse_train_std",
        "hot_q90_rmse",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--phase183-gate", type=Path, default=DEFAULT_PHASE183_GATE)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.dataset, args.phase183_gate, args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_metrics(args.metrics_output, payload["metrics"])
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
