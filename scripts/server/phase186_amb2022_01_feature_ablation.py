#!/usr/bin/env python3
"""Execute heat-kernel and shuffled-history ablations for AMB2022-01."""

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
DEFAULT_PHASE185 = Path(
    os.environ.get(
        "AMB2022_01_PHASE185_CONTRACT",
        "/root/matsci-gnn-pinn-ops/phase185_ablation_contract.json",
    )
)
KERNEL_ALPHAS_MM2_S = (1.0, 5.0, 20.0)
SHUFFLE_SEEDS = (1841, 1842, 1843)
SPLITS = {"train": 0, "val": 1, "test": 2}
TARGETS = {"tam_s": "target_tam_s", "scr_C_per_s": "target_scr_C_per_s"}
COORDINATE_NAMES = ("x_mm", "y_mm", "z_mm", "layer_fraction", "block_area_mm2")
SCAN_HISTORY_NAMES = (
    "laser_active",
    "laser_dwell_s",
    "laser_energy_J",
    "laser_energy_density_J_mm2",
    "mean_power_W",
    "max_power_W",
    "first_laser_time_s",
    "last_laser_time_s",
    "time_since_last_laser_s",
    "energy_weighted_progress",
    "laser_progress_span",
    "staring_trigger_count",
)


def _h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("h5py is required for Phase 186 feature ablation") from exc
    return h5py


def _json_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def _read_phase185(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate = payload.get("gate", {})
    expected = "phase185_ablation_contract_ready_phase186_feature_ablation_execution"
    if gate.get("status") != expected or not gate.get("phase186_feature_ablation_execution_allowed"):
        raise ValueError(f"Phase 185 does not permit feature ablations: {gate.get('status')!r}")
    return payload


def causal_heat_kernel_features(
    *,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    energy_j: np.ndarray,
    local_time_s: np.ndarray,
    alphas_mm2_s: tuple[float, ...] = KERNEL_ALPHAS_MM2_S,
) -> np.ndarray:
    """Evaluate causal, diffusion-like source history at each active coarse block.

    The target time is the local XYPT-derived last-laser command time. Direct local
    energy remains an existing feature; this descriptor only integrates prior blocks.
    """
    x_mm = np.asarray(x_mm, dtype=float)
    y_mm = np.asarray(y_mm, dtype=float)
    energy_j = np.asarray(energy_j, dtype=float)
    local_time_s = np.asarray(local_time_s, dtype=float)
    if not (len(x_mm) == len(y_mm) == len(energy_j) == len(local_time_s)):
        raise ValueError("Heat-kernel inputs must share a row count")
    result = np.zeros((len(x_mm), len(alphas_mm2_s)), dtype=np.float32)
    active = energy_j > 0.0
    source_idx = np.flatnonzero(active)
    target_idx = np.flatnonzero(active)
    if len(source_idx) == 0:
        return result
    dx = x_mm[target_idx, None] - x_mm[source_idx][None, :]
    dy = y_mm[target_idx, None] - y_mm[source_idx][None, :]
    distance2 = dx**2 + dy**2
    delay = local_time_s[target_idx, None] - local_time_s[source_idx][None, :]
    causal = delay > 1e-6
    for column, alpha in enumerate(alphas_mm2_s):
        if alpha <= 0.0:
            raise ValueError("Effective diffusivity values must be positive")
        safe_delay = np.where(causal, delay, 1.0)
        kernel = np.where(
            causal,
            np.exp(-distance2 / (4.0 * alpha * safe_delay)) / (4.0 * math.pi * alpha * safe_delay),
            0.0,
        )
        history = kernel @ energy_j[source_idx]
        result[target_idx, column] = np.log1p(np.maximum(history, 0.0)).astype(np.float32)
    return result


def shuffle_history_within_layer(
    features: np.ndarray,
    *,
    build_index: np.ndarray,
    layer_index: np.ndarray,
    history_columns: list[int],
    seed: int,
) -> np.ndarray:
    shuffled = np.asarray(features, dtype=np.float32).copy()
    rng = np.random.default_rng(seed)
    pairs = np.column_stack([build_index, layer_index])
    unique_pairs = np.unique(pairs, axis=0)
    for build_id, layer_id in unique_pairs:
        rows = np.flatnonzero((build_index == build_id) & (layer_index == layer_id))
        permutation = rng.permutation(len(rows))
        shuffled[np.ix_(rows, history_columns)] = features[np.ix_(rows[permutation], history_columns)]
    return shuffled


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> dict[str, Any]:
    x_mean = x_train.mean(axis=0)
    x_scale = x_train.std(axis=0)
    x_scale[x_scale <= 1e-12] = 1.0
    y_mean = float(y_train.mean())
    y_scale = float(y_train.std())
    if y_scale <= 1e-12:
        raise ValueError("Training target has no variance")
    standard_x = (x_train - x_mean) / x_scale
    standard_y = (y_train - y_mean) / y_scale
    weights = np.linalg.solve(
        standard_x.T @ standard_x + np.eye(standard_x.shape[1]) * alpha,
        standard_x.T @ standard_y,
    )
    return {"x_mean": x_mean, "x_scale": x_scale, "y_mean": y_mean, "y_scale": y_scale, "weights": weights}


def predict_ridge(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return (
        (x - model["x_mean"]) / model["x_scale"] @ model["weights"] * model["y_scale"] + model["y_mean"]
    )


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y - pred) ** 2)))


def evaluate_ridge(
    *,
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    valid_masks: dict[str, np.ndarray],
    split_codes: np.ndarray,
    variant_id: str,
    alpha: float,
    seed: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_name, values in targets.items():
        masks = {name: (split_codes == code) & valid_masks[target_name] for name, code in SPLITS.items()}
        train_values = values[masks["train"]].astype(np.float64)
        model = fit_ridge(features[masks["train"]].astype(np.float64), train_values, alpha)
        for split_name, mask in masks.items():
            y = values[mask].astype(np.float64)
            pred = predict_ridge(model, features[mask].astype(np.float64))
            rows.append(
                {
                    "variant_id": variant_id,
                    "target": target_name,
                    "split": split_name,
                    "seed": "deterministic" if seed is None else seed,
                    "n_rows": int(len(y)),
                    "feature_count": int(features.shape[1]),
                    "rmse": _rmse(y, pred),
                }
            )
    return rows


def _baseline_full_rmse(phase185: dict[str, Any], target: str, split: str) -> float:
    return float(phase185["observed_low_capacity_comparison"][target][split]["full_ridge_rmse"])


def build_gate(phase185: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    comparisons: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        heat = {
            row["split"]: row
            for row in rows
            if row["target"] == target and row["variant_id"] == "heat_kernel_history_ridge"
        }
        shuffled = [
            row
            for row in rows
            if row["target"] == target and row["variant_id"] == "layerwise_shuffled_scan_history_control"
        ]
        if set(heat) != set(SPLITS) or not shuffled:
            blockers.append(f"{target}_ablation_metrics_incomplete")
            continue
        shuffle_by_split = {
            split: [float(row["rmse"]) for row in shuffled if row["split"] == split]
            for split in SPLITS
        }
        if any(not values for values in shuffle_by_split.values()):
            blockers.append(f"{target}_shuffle_seed_metrics_incomplete")
            continue
        comparisons[target] = {
            "heat_kernel_validation_gain_vs_full": _baseline_full_rmse(phase185, target, "val") - float(heat["val"]["rmse"]),
            "heat_kernel_test_gain_vs_full": _baseline_full_rmse(phase185, target, "test") - float(heat["test"]["rmse"]),
            "shuffle_validation_penalty_vs_full": float(np.mean(shuffle_by_split["val"])) - _baseline_full_rmse(phase185, target, "val"),
            "shuffle_test_penalty_vs_full": float(np.mean(shuffle_by_split["test"])) - _baseline_full_rmse(phase185, target, "test"),
        }
        if comparisons[target]["shuffle_validation_penalty_vs_full"] <= 0.0:
            blockers.append(f"{target}_shuffle_control_not_degrading_validation")
        if comparisons[target]["shuffle_test_penalty_vs_full"] <= 0.0:
            blockers.append(f"{target}_shuffle_control_not_degrading_test")
        if comparisons[target]["heat_kernel_validation_gain_vs_full"] < 0.0:
            blockers.append(f"{target}_heat_kernel_validation_regression")
        if comparisons[target]["heat_kernel_test_gain_vs_full"] < 0.0:
            blockers.append(f"{target}_heat_kernel_test_regression")
    ready = not blockers
    return {
        "status": (
            "phase186_feature_ablation_ready_phase187_candidate_model_design"
            if ready
            else "phase186_feature_ablation_closed_no_stable_mechanism_gain"
        ),
        "feature_ablation_ready": ready,
        "phase187_candidate_model_design_allowed": ready,
        "model_training_allowed": False,
        "a800_training_allowed_now": False,
        "comparisons": comparisons,
        "blocking_audits": blockers,
        "next_action": (
            "design a bounded candidate model with the surviving mechanism and all fixed controls"
            if ready
            else "retain ridge evidence; do not escalate a non-improving heat-kernel mechanism to neural training"
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ("variant_id", "target", "split", "seed", "n_rows", "feature_count", "rmse")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(dataset_path: Path, phase185_path: Path, alpha: float) -> tuple[dict[str, Any], np.ndarray]:
    h5py = _h5py()
    phase185 = _read_phase185(phase185_path)
    with h5py.File(dataset_path, "r") as handle:
        features = handle["features"][...].astype(np.float32)
        feature_names = list(_json_attr(handle.attrs["feature_names"]))
        build_index = handle["build_index"][...]
        layer_index = handle["layer_index"][...]
        split_codes = handle["primary_split"][...]
        valid = handle["target_valid_mask"][...].astype(bool)
        targets = {name: handle[path][...] for name, path in TARGETS.items()}
    if not np.isfinite(features).all():
        raise ValueError("Non-finite Phase 182 model features")
    x_col = feature_names.index("x_mm")
    y_col = feature_names.index("y_mm")
    energy_col = feature_names.index("laser_energy_J")
    time_col = feature_names.index("last_laser_time_s")
    history_cols = [feature_names.index(name) for name in SCAN_HISTORY_NAMES]
    kernel = np.zeros((len(features), len(KERNEL_ALPHAS_MM2_S)), dtype=np.float32)
    for layer in sorted(np.unique(layer_index)):
        representative = np.flatnonzero((build_index == 0) & (layer_index == layer))
        values = causal_heat_kernel_features(
            x_mm=features[representative, x_col],
            y_mm=features[representative, y_col],
            energy_j=features[representative, energy_col],
            local_time_s=features[representative, time_col],
        )
        for build_id in np.unique(build_index):
            rows = np.flatnonzero((build_index == build_id) & (layer_index == layer))
            if len(rows) != len(representative):
                raise ValueError("Build/layer coarse-grid cardinality differs")
            kernel[rows] = values
    full_with_kernel = np.column_stack([features, kernel])
    valid_masks = {"tam_s": valid[:, 0], "scr_C_per_s": valid[:, 1]}
    metric_rows = evaluate_ridge(
        features=full_with_kernel,
        targets=targets,
        valid_masks=valid_masks,
        split_codes=split_codes,
        variant_id="heat_kernel_history_ridge",
        alpha=alpha,
        seed=None,
    )
    for seed in SHUFFLE_SEEDS:
        shuffled = shuffle_history_within_layer(
            features,
            build_index=build_index,
            layer_index=layer_index,
            history_columns=history_cols,
            seed=seed,
        )
        metric_rows.extend(
            evaluate_ridge(
                features=shuffled,
                targets=targets,
                valid_masks=valid_masks,
                split_codes=split_codes,
                variant_id="layerwise_shuffled_scan_history_control",
                alpha=alpha,
                seed=seed,
            )
        )
    payload = {
        "phase": 186,
        "objective": "heat_kernel_and_layerwise_shuffle_feature_ablation",
        "dataset": str(dataset_path),
        "phase185_contract": str(phase185_path),
        "ridge_alpha": alpha,
        "kernel_alphas_mm2_s": list(KERNEL_ALPHAS_MM2_S),
        "kernel_feature_names": [f"causal_heat_kernel_alpha_{alpha:g}_mm2_s" for alpha in KERNEL_ALPHAS_MM2_S],
        "shuffle_seeds": list(SHUFFLE_SEEDS),
        "metrics": metric_rows,
        "gate": build_gate(phase185, metric_rows),
    }
    return payload, kernel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--phase185", type=Path, default=DEFAULT_PHASE185)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--kernel-output", type=Path, required=True)
    args = parser.parse_args()
    payload, kernel = run(args.dataset, args.phase185, args.alpha)
    h5py = _h5py()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.metrics_output, payload["metrics"])
    args.kernel_output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.kernel_output, "w") as handle:
        handle.create_dataset("causal_heat_kernel_features", data=kernel, compression="gzip", compression_opts=4, shuffle=True)
        handle.attrs["kernel_alphas_mm2_s"] = json.dumps(KERNEL_ALPHAS_MM2_S)
        handle.attrs["scope"] = "row-aligned supplemental features for Phase 182; no targets used"
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
