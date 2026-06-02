"""Phase 46 Bayesian inverse-closure feasibility probes.

The probe is intentionally lightweight. It tests whether low-dimensional
Bayesian source/closure parameters are identifiable before any full Bayesian
PINN or A100 AM-Bench expansion is justified.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gnnpinn.data.loaders import load_field_table
from gnnpinn.data.splits import load_split_manifest, split_indices
from gnnpinn.eval.baselines import regression_metric_table


Z90 = 1.6448536269514722


@dataclass(frozen=True)
class ProbeData:
    label: str
    features: np.ndarray
    target: np.ndarray
    feature_names: list[str]
    splits: dict[str, np.ndarray]
    rows: np.ndarray
    cols: np.ndarray
    frames: np.ndarray
    source_prior_score: np.ndarray
    true_theta: np.ndarray | None = None


@dataclass(frozen=True)
class Posterior:
    mean: np.ndarray
    covariance: np.ndarray
    noise_variance: float


def _scale01(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    span = float(values.max() - values.min()) if values.size else 0.0
    if span <= 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - float(values.min())) / span


def _gaussian_source(x: np.ndarray, y: np.ndarray, t: np.ndarray, width: float) -> np.ndarray:
    center_x = 0.25 + 0.50 * t
    center_y = 0.52 + 0.05 * np.sin(2.0 * math.pi * t)
    radius2 = (x - center_x) ** 2 + (y - center_y) ** 2
    return np.exp(-0.5 * radius2 / (width**2))


def make_synthetic_data(
    *,
    n_grid: int,
    n_frames: int,
    seed: int,
    noise_std: float,
) -> ProbeData:
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, n_grid)
    frame_values = np.linspace(0.0, 1.0, n_frames)
    rows: list[int] = []
    cols: list[int] = []
    frames: list[int] = []
    x_values: list[float] = []
    y_values: list[float] = []
    t_values: list[float] = []
    for frame_index, t_value in enumerate(frame_values):
        for row_index, y_value in enumerate(grid):
            for col_index, x_value in enumerate(grid):
                rows.append(row_index)
                cols.append(col_index)
                frames.append(frame_index)
                x_values.append(float(x_value))
                y_values.append(float(y_value))
                t_values.append(float(t_value))

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    t = np.asarray(t_values, dtype=float)
    source_core = _gaussian_source(x, y, t, width=0.105)
    source_tail = _gaussian_source(x, y, t, width=0.24) * (1.0 - 0.35 * t)
    features = np.column_stack(
        [
            np.ones_like(x),
            source_core,
            source_tail,
            x,
            y,
            t,
        ]
    )
    feature_names = [
        "ambient",
        "source_core_amp",
        "source_tail_amp",
        "x_background",
        "y_background",
        "time_drift",
    ]
    true_theta = np.asarray([1050.0, 235.0, 82.0, 24.0, -18.0, 35.0], dtype=float)
    target = features @ true_theta + rng.normal(0.0, noise_std, size=features.shape[0])

    all_indices = np.arange(features.shape[0])
    rng.shuffle(all_indices)
    n_train = int(round(0.60 * len(all_indices)))
    n_val = int(round(0.20 * len(all_indices)))
    splits = {
        "train": np.sort(all_indices[:n_train]),
        "val": np.sort(all_indices[n_train : n_train + n_val]),
        "test": np.sort(all_indices[n_train + n_val :]),
    }
    return ProbeData(
        label="synthetic_heat_source",
        features=features,
        target=target,
        feature_names=feature_names,
        splits=splits,
        rows=np.asarray(rows, dtype=float),
        cols=np.asarray(cols, dtype=float),
        frames=np.asarray(frames, dtype=float),
        source_prior_score=source_core,
        true_theta=true_theta,
    )


def with_physics_guided_attention_features(data: ProbeData) -> ProbeData:
    """Add deterministic hot/gradient/source-gated closure features.

    This is the lightest testable version of an attention idea: instead of a
    trainable attention block, the gate is computed from physically meaningful
    source-prior and gradient proxies available before fitting the closure
    coefficients. The learned parameters remain linear and interpretable.
    """

    gradient = _scale01(_gradient_scores(data, data.source_prior_score))
    hot_proxy = _scale01(data.source_prior_score)
    attention = np.clip(0.5 * hot_proxy + 0.5 * gradient, 0.0, 1.0)
    source_columns = [
        index
        for index, name in enumerate(data.feature_names)
        if "source" in name or "amp" in name
    ]
    if not source_columns:
        source_columns = [index for index in range(1, min(4, data.features.shape[1]))]
    gated_features = []
    gated_names = []
    for column in source_columns:
        gated_features.append(data.features[:, column] * attention)
        gated_names.append(f"attn_{data.feature_names[column]}")
    expanded = np.column_stack([data.features, *gated_features])
    true_theta = None
    if data.true_theta is not None:
        true_theta = np.concatenate([data.true_theta, np.zeros(len(gated_names), dtype=float)])
    return ProbeData(
        label=f"{data.label}_physics_guided_attention",
        features=expanded,
        target=data.target,
        feature_names=[*data.feature_names, *gated_names],
        splits=data.splits,
        rows=data.rows,
        cols=data.cols,
        frames=data.frames,
        source_prior_score=data.source_prior_score,
        true_theta=true_theta,
    )


def with_heat_kernel_features(data: ProbeData) -> ProbeData:
    """Add moving heat-source diffusion-kernel basis features.

    These features are a lightweight Green's-function proxy: each basis assumes
    heat is deposited along a moving source path and decays with spatial
    distance and time lag. The coefficients remain linear/Bayesian.
    """

    x = _scale01(data.cols)
    y = _scale01(data.rows)
    t = _scale01(data.frames)
    source_y = np.full_like(t, 0.5)
    diffusion_scales = [0.06, 0.12, 0.24]
    temporal_decays = [0.15, 0.35, 0.70]
    kernel_features: list[np.ndarray] = []
    kernel_names: list[str] = []
    for diffusion in diffusion_scales:
        for decay in temporal_decays:
            values = np.zeros_like(t, dtype=float)
            for lag in [0.0, 0.1, 0.2, 0.35]:
                source_t = np.clip(t - lag, 0.0, 1.0)
                source_x = 0.20 + 0.60 * source_t
                radius2 = (x - source_x) ** 2 + (y - source_y) ** 2
                lag_weight = np.exp(-lag / decay)
                width2 = diffusion**2 + 0.20 * lag
                values += lag_weight * np.exp(-0.5 * radius2 / max(width2, 1e-8))
            kernel_features.append(_scale01(values))
            kernel_names.append(f"heat_kernel_d{diffusion:g}_tau{decay:g}")
    source_gradient = _source_gradient_score(data)
    source_hot = _scale01(data.source_prior_score)
    kernel_features.append(source_hot * source_gradient)
    kernel_names.append("source_hot_x_gradient")
    expanded = np.column_stack([data.features, *kernel_features])
    true_theta = None
    if data.true_theta is not None:
        true_theta = np.concatenate([data.true_theta, np.zeros(len(kernel_names), dtype=float)])
    return ProbeData(
        label=f"{data.label}_heat_kernel",
        features=expanded,
        target=data.target,
        feature_names=[*data.feature_names, *kernel_names],
        splits=data.splits,
        rows=data.rows,
        cols=data.cols,
        frames=data.frames,
        source_prior_score=np.maximum(data.source_prior_score, np.max(np.column_stack(kernel_features), axis=1)),
        true_theta=true_theta,
    )


def make_table_data(
    *,
    table: Path,
    target: str,
    split_manifest: Path,
) -> ProbeData:
    sample = load_field_table(table, observation_columns=[target])
    split_payload = load_split_manifest(split_manifest)
    target_values = np.asarray(sample.require_observation(target), dtype=float)
    coordinates = np.asarray(sample.coordinates, dtype=float)
    x = _scale01(coordinates[:, 0])
    y = _scale01(coordinates[:, 1]) if coordinates.shape[1] > 1 else np.zeros(sample.n_points)
    t = _scale01(np.asarray(sample.time, dtype=float))
    source_mid = np.exp(-0.5 * ((x - 0.5) ** 2 + (y - 0.5) ** 2) / (0.22**2))
    source_wide = np.exp(-0.5 * ((x - 0.5) ** 2 + (y - 0.5) ** 2) / (0.38**2))
    moving_source = np.exp(-0.5 * ((x - (0.2 + 0.6 * t)) ** 2 + (y - 0.5) ** 2) / (0.24**2))
    features = np.column_stack(
        [
            np.ones(sample.n_points),
            source_mid,
            source_wide,
            moving_source,
            x,
            y,
            t,
            x * y,
            x * t,
            y * t,
        ]
    )
    feature_names = [
        "ambient",
        "source_mid_amp",
        "source_wide_amp",
        "moving_source_amp",
        "x_background",
        "y_background",
        "time_drift",
        "xy_interaction",
        "xt_interaction",
        "yt_interaction",
    ]
    row_metadata = sample.metadata.get("row_metadata") or {}
    rows = _metadata_float_array(row_metadata, "row_index", fallback=np.arange(sample.n_points, dtype=float))
    cols = _metadata_float_array(row_metadata, "col_index", fallback=np.zeros(sample.n_points, dtype=float))
    frames = _metadata_float_array(row_metadata, "frame_index", fallback=np.zeros(sample.n_points, dtype=float))
    splits = {
        name: np.asarray(split_indices(split_payload, name), dtype=int)
        for name in split_payload["splits"]
    }
    return ProbeData(
        label=table.stem,
        features=features,
        target=target_values,
        feature_names=feature_names,
        splits=splits,
        rows=rows,
        cols=cols,
        frames=frames,
        source_prior_score=np.maximum.reduce([source_mid, moving_source]),
    )


def _metadata_float_array(
    row_metadata: dict[str, list[Any]],
    name: str,
    *,
    fallback: np.ndarray,
) -> np.ndarray:
    values = row_metadata.get(name)
    if values is None:
        return fallback.astype(float)
    try:
        return np.asarray([float(value) for value in values], dtype=float)
    except (TypeError, ValueError):
        return fallback.astype(float)


def fit_bayesian_linear(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    prior_variance: float,
    noise_variance: float | None,
    noise_floor: float,
) -> Posterior:
    if x_train.ndim != 2:
        raise ValueError("x_train must be a 2D matrix")
    if len(x_train) != len(y_train):
        raise ValueError("x_train and y_train length mismatch")
    n_features = x_train.shape[1]
    if noise_variance is None:
        ridge = 1e-8
        lhs = x_train.T @ x_train + ridge * np.eye(n_features)
        coef = np.linalg.solve(lhs, x_train.T @ y_train)
        residual = y_train - x_train @ coef
        dof = max(1, len(y_train) - n_features)
        estimated = float(np.dot(residual, residual) / dof)
        variance_floor = max(noise_floor**2, float(np.var(y_train)) * 1e-8, 1e-8)
        noise_variance = max(estimated, variance_floor)
    precision = (x_train.T @ x_train) / noise_variance + np.eye(n_features) / prior_variance
    covariance = np.linalg.inv(precision)
    mean = covariance @ (x_train.T @ y_train) / noise_variance
    return Posterior(mean=mean, covariance=covariance, noise_variance=float(noise_variance))


def posterior_predict(posterior: Posterior, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x @ posterior.mean
    epistemic = np.einsum("ij,jk,ik->i", x, posterior.covariance, x)
    variance = np.maximum(epistemic + posterior.noise_variance, 1e-12)
    return mean, np.sqrt(variance)


def _source_gradient_score(data: ProbeData) -> np.ndarray:
    return _scale01(_gradient_scores(data, data.source_prior_score))


def _select_top_indices(
    candidates: np.ndarray,
    score_by_index: np.ndarray,
    count: int,
    selected: set[int],
) -> list[int]:
    if count <= 0:
        return []
    available = [int(index) for index in candidates if int(index) not in selected]
    if not available:
        return []
    ordered = sorted(available, key=lambda index: float(score_by_index[index]), reverse=True)
    return ordered[:count]


def _warm_posterior_scores(
    data: ProbeData,
    initial: np.ndarray,
    remaining: np.ndarray,
    *,
    prior_variance: float,
    noise_variance: float | None,
    noise_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    warm = fit_bayesian_linear(
        data.features[initial],
        data.target[initial],
        prior_variance=prior_variance,
        noise_variance=noise_variance,
        noise_floor=noise_floor,
    )
    _, std = posterior_predict(warm, data.features[remaining])
    score = np.zeros(len(data.target), dtype=float)
    score[remaining] = _scale01(std)
    source = _scale01(data.source_prior_score)
    source_gradient = _source_gradient_score(data)
    return score, source, source_gradient


def _acquire_indices(
    data: ProbeData,
    *,
    strategy: str,
    rng: np.random.Generator,
    initial: np.ndarray,
    remaining: np.ndarray,
    acquisition_size: int,
    prior_variance: float,
    noise_variance: float | None,
    noise_floor: float,
) -> np.ndarray:
    if acquisition_size <= 0:
        return np.asarray([], dtype=int)
    if strategy == "random":
        return np.sort(rng.choice(remaining, size=acquisition_size, replace=False))

    uncertainty, source, source_gradient = _warm_posterior_scores(
        data,
        initial,
        remaining,
        prior_variance=prior_variance,
        noise_variance=noise_variance,
        noise_floor=noise_floor,
    )
    if strategy == "uncertainty":
        score = uncertainty
        return np.asarray(_select_top_indices(remaining, score, acquisition_size, set()), dtype=int)
    if strategy == "uncertainty_source":
        score = uncertainty * (1.0 + source)
        return np.asarray(_select_top_indices(remaining, score, acquisition_size, set()), dtype=int)
    if strategy == "pareto_source_gradient":
        score = uncertainty + source + source_gradient
        return np.asarray(_select_top_indices(remaining, score, acquisition_size, set()), dtype=int)
    if strategy == "region_quota_uncertainty":
        selected: set[int] = set()
        hot_quota = int(round(0.25 * acquisition_size))
        gradient_quota = int(round(0.25 * acquisition_size))
        source_threshold = float(np.quantile(source[remaining], 0.8))
        gradient_threshold = float(np.quantile(source_gradient[remaining], 0.8))
        source_region = remaining[source[remaining] >= source_threshold]
        gradient_region = remaining[source_gradient[remaining] >= gradient_threshold]
        source_score = uncertainty * (1.0 + source)
        gradient_score = uncertainty * (1.0 + source_gradient)
        for index in _select_top_indices(source_region, source_score, hot_quota, selected):
            selected.add(index)
        for index in _select_top_indices(gradient_region, gradient_score, gradient_quota, selected):
            selected.add(index)
        fill_score = uncertainty + 0.5 * source + 0.5 * source_gradient
        fill_count = acquisition_size - len(selected)
        for index in _select_top_indices(remaining, fill_score, fill_count, selected):
            selected.add(index)
        return np.asarray(sorted(selected), dtype=int)
    raise ValueError(f"Unsupported acquisition strategy: {strategy}")


def _validation_objective(data: ProbeData, posterior: Posterior) -> float:
    mean, _ = posterior_predict(posterior, data.features)
    val_indices = np.asarray(data.splits.get("val", []), dtype=int)
    if len(val_indices) == 0:
        return float("inf")
    split_metrics = _split_metrics(data, val_indices, mean, np.ones_like(mean))
    metrics = split_metrics["metrics"]
    regions = split_metrics.get("region_metrics", {})
    hot = regions.get("hot_q90", {}).get("metrics", {}).get("rmse", metrics["rmse"])
    gradient = regions.get("gradient_q90", {}).get("metrics", {}).get("rmse", metrics["rmse"])
    return float(metrics["rmse"] + 0.5 * hot + 0.5 * gradient)


def _select_validation_policy(
    data: ProbeData,
    *,
    rng: np.random.Generator,
    initial: np.ndarray,
    remaining: np.ndarray,
    acquisition_size: int,
    prior_variance: float,
    noise_variance: float | None,
    noise_floor: float,
) -> tuple[np.ndarray, str, dict[str, float]]:
    candidates = ["uncertainty_source", "region_quota_uncertainty", "pareto_source_gradient"]
    objectives: dict[str, float] = {}
    acquired_by_policy: dict[str, np.ndarray] = {}
    for candidate in candidates:
        acquired = _acquire_indices(
            data,
            strategy=candidate,
            rng=rng,
            initial=initial,
            remaining=remaining,
            acquisition_size=acquisition_size,
            prior_variance=prior_variance,
            noise_variance=noise_variance,
            noise_floor=noise_floor,
        )
        selected = np.sort(np.concatenate([initial, acquired]))
        posterior = fit_bayesian_linear(
            data.features[selected],
            data.target[selected],
            prior_variance=prior_variance,
            noise_variance=noise_variance,
            noise_floor=noise_floor,
        )
        objectives[candidate] = _validation_objective(data, posterior)
        acquired_by_policy[candidate] = acquired
    selected_policy = min(objectives, key=objectives.get)
    return acquired_by_policy[selected_policy], selected_policy, objectives


def _conformal_std(
    data: ProbeData,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if mode == "none":
        return std, {"mode": "none", "scale": 1.0}
    if mode != "conformal90":
        raise ValueError(f"Unsupported calibration mode: {mode}")
    val_indices = np.asarray(data.splits.get("val", []), dtype=int)
    if len(val_indices) == 0:
        return std, {"mode": mode, "scale": 1.0, "reason": "missing validation split"}
    safe_std = np.maximum(std[val_indices], 1e-12)
    nonconformity = np.abs(data.target[val_indices] - mean[val_indices]) / safe_std
    quantile = float(np.quantile(nonconformity, 0.9))
    scale = max(1.0, quantile / Z90)
    return std * scale, {
        "mode": mode,
        "scale": float(scale),
        "nonconformity_q90": quantile,
        "validation_points": int(len(val_indices)),
    }


def run_sampling_strategy(
    data: ProbeData,
    *,
    strategy: str,
    seed: int,
    initial_size: int,
    acquisition_size: int,
    prior_variance: float,
    noise_variance: float | None,
    noise_floor: float,
    calibration_mode: str,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train_pool = np.asarray(data.splits["train"], dtype=int)
    if initial_size + acquisition_size > len(train_pool):
        raise ValueError("initial-size + acquisition-size cannot exceed train split size")
    initial = np.sort(rng.choice(train_pool, size=initial_size, replace=False))
    remaining = np.setdiff1d(train_pool, initial, assume_unique=False)
    selected_policy = None
    validation_objectives = None
    if acquisition_size:
        if strategy == "validation_selected_region_policy":
            acquired, selected_policy, validation_objectives = _select_validation_policy(
                data,
                rng=rng,
                initial=initial,
                remaining=remaining,
                acquisition_size=acquisition_size,
                prior_variance=prior_variance,
                noise_variance=noise_variance,
                noise_floor=noise_floor,
            )
        else:
            acquired = _acquire_indices(
                data,
                strategy=strategy,
                rng=rng,
                initial=initial,
                remaining=remaining,
                acquisition_size=acquisition_size,
                prior_variance=prior_variance,
                noise_variance=noise_variance,
                noise_floor=noise_floor,
            )
    else:
        acquired = np.asarray([], dtype=int)
    selected = np.sort(np.concatenate([initial, acquired]))
    posterior = fit_bayesian_linear(
        data.features[selected],
        data.target[selected],
        prior_variance=prior_variance,
        noise_variance=noise_variance,
        noise_floor=noise_floor,
    )
    all_mean, all_std = posterior_predict(posterior, data.features)
    all_std, calibration_payload = _conformal_std(data, all_mean, all_std, mode=calibration_mode)
    split_payload = {
        split: _split_metrics(data, indices, all_mean, all_std)
        for split, indices in data.splits.items()
    }
    payload: dict[str, Any] = {
        "strategy": strategy,
        "seed": seed,
        "initial_size": initial_size,
        "acquisition_size": acquisition_size,
        "selected_size": int(len(selected)),
        "noise_variance": posterior.noise_variance,
        "calibration": calibration_payload,
        "splits": split_payload,
    }
    if selected_policy is not None:
        payload["selected_policy"] = selected_policy
        payload["validation_objectives"] = validation_objectives
    if data.true_theta is not None:
        payload["parameter_recovery"] = _parameter_recovery_payload(
            data.feature_names,
            posterior,
            data.true_theta,
        )
    return payload


def _split_metrics(
    data: ProbeData,
    indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, Any]:
    idx = np.asarray(indices, dtype=int)
    y_true = data.target[idx].tolist()
    y_pred = mean[idx].tolist()
    metrics: dict[str, Any] = {
        "n_points": int(len(idx)),
        "metrics": regression_metric_table(y_true, y_pred),
        "coverage_90": _coverage(data.target[idx], mean[idx], std[idx], Z90),
        "mean_predictive_std": float(np.mean(std[idx])) if len(idx) else 0.0,
    }
    regions = _region_metrics(data, idx, mean)
    if regions:
        metrics["region_metrics"] = regions
    return metrics


def _coverage(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, z_value: float) -> float:
    lower = mean - z_value * std
    upper = mean + z_value * std
    return float(np.mean((y_true >= lower) & (y_true <= upper))) if len(y_true) else 0.0


def _region_metrics(data: ProbeData, indices: np.ndarray, mean: np.ndarray) -> dict[str, Any]:
    if len(indices) == 0:
        return {}
    y_true = data.target
    output: dict[str, Any] = {}
    hot_indices = _quantile_indices(y_true, indices, 0.9)
    output["hot_q90"] = {
        "n_points": int(len(hot_indices)),
        "metrics": regression_metric_table(
            y_true[hot_indices].tolist(),
            mean[hot_indices].tolist(),
        )
        if len(hot_indices)
        else {},
    }
    gradient_scores = _gradient_scores(data, y_true)
    grad_indices = _quantile_indices(gradient_scores, indices, 0.9)
    output["gradient_q90"] = {
        "n_points": int(len(grad_indices)),
        "metrics": regression_metric_table(
            y_true[grad_indices].tolist(),
            mean[grad_indices].tolist(),
        )
        if len(grad_indices)
        else {},
    }
    return output


def _quantile_indices(values: np.ndarray, indices: np.ndarray, quantile: float) -> np.ndarray:
    if len(indices) == 0:
        return np.asarray([], dtype=int)
    threshold = float(np.quantile(values[indices], quantile))
    return indices[values[indices] >= threshold]


def _gradient_scores(data: ProbeData, values: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(values), dtype=float)
    groups: dict[float, list[int]] = {}
    for index, frame in enumerate(data.frames):
        groups.setdefault(float(frame), []).append(index)
    for group_indices in groups.values():
        by_position = {
            (float(data.rows[index]), float(data.cols[index])): index
            for index in group_indices
        }
        row_values = sorted({float(data.rows[index]) for index in group_indices})
        col_values = sorted({float(data.cols[index]) for index in group_indices})
        row_neighbors = _axis_neighbors(row_values)
        col_neighbors = _axis_neighbors(col_values)
        for index in group_indices:
            row = float(data.rows[index])
            col = float(data.cols[index])
            local: list[float] = []
            for neighbor_row in row_neighbors.get(row, []):
                neighbor = by_position.get((neighbor_row, col))
                if neighbor is not None:
                    local.append(abs(float(values[index] - values[neighbor])) / (abs(neighbor_row - row) or 1.0))
            for neighbor_col in col_neighbors.get(col, []):
                neighbor = by_position.get((row, neighbor_col))
                if neighbor is not None:
                    local.append(abs(float(values[index] - values[neighbor])) / (abs(neighbor_col - col) or 1.0))
            if local:
                scores[index] = max(local)
    return scores


def _axis_neighbors(values: list[float]) -> dict[float, list[float]]:
    output: dict[float, list[float]] = {}
    for position, value in enumerate(values):
        neighbors: list[float] = []
        if position > 0:
            neighbors.append(values[position - 1])
        if position + 1 < len(values):
            neighbors.append(values[position + 1])
        output[value] = neighbors
    return output


def _parameter_recovery_payload(
    feature_names: list[str],
    posterior: Posterior,
    true_theta: np.ndarray,
) -> dict[str, Any]:
    std = np.sqrt(np.maximum(np.diag(posterior.covariance), 0.0))
    rows: list[dict[str, Any]] = []
    covered = []
    for name, true_value, mean_value, std_value in zip(feature_names, true_theta, posterior.mean, std):
        lower = float(mean_value - Z90 * std_value)
        upper = float(mean_value + Z90 * std_value)
        is_covered = lower <= float(true_value) <= upper
        covered.append(is_covered)
        rows.append(
            {
                "name": name,
                "true": float(true_value),
                "posterior_mean": float(mean_value),
                "posterior_std": float(std_value),
                "ci90": [lower, upper],
                "covered_by_ci90": bool(is_covered),
                "absolute_error": float(abs(mean_value - true_value)),
            }
        )
    source_rows = [
        row
        for row in rows
        if row["name"] in {"source_core_amp", "source_tail_amp"}
    ]
    source_covered = [row["covered_by_ci90"] for row in source_rows]
    source_mean_abs_error = float(np.mean([row["absolute_error"] for row in source_rows])) if source_rows else 0.0
    return {
        "parameters": rows,
        "all_ci90_coverage": float(np.mean(covered)) if covered else 0.0,
        "source_parameter_ci90_coverage": float(np.mean(source_covered)) if source_covered else 0.0,
        "source_parameter_mean_abs_error": source_mean_abs_error,
        "source_recovery_pass": bool(source_covered and all(source_covered) and source_mean_abs_error <= 25.0),
    }


def _strategy_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_strategy.setdefault(result["strategy"], []).append(result)
    output: dict[str, Any] = {}
    for strategy, rows in by_strategy.items():
        output[strategy] = {
            "n_repeats": len(rows),
            "test_rmse_mean": _mean_metric(rows, "test", "rmse"),
            "test_hot_q90_rmse_mean": _mean_region_metric(rows, "test", "hot_q90", "rmse"),
            "test_gradient_q90_rmse_mean": _mean_region_metric(rows, "test", "gradient_q90", "rmse"),
            "test_coverage90_mean": float(np.mean([row["splits"]["test"]["coverage_90"] for row in rows])),
        }
        recovery_values = [
            row.get("parameter_recovery", {}).get("source_recovery_pass")
            for row in rows
            if "parameter_recovery" in row
        ]
        if recovery_values:
            output[strategy]["source_recovery_pass_rate"] = float(np.mean([bool(value) for value in recovery_values]))
            output[strategy]["source_parameter_mean_abs_error_mean"] = float(
                np.mean([
                    row["parameter_recovery"]["source_parameter_mean_abs_error"]
                    for row in rows
                    if "parameter_recovery" in row
                ])
            )
        selected_policies = [
            row.get("selected_policy")
            for row in rows
            if row.get("selected_policy") is not None
        ]
        if selected_policies:
            policy_counts = {
                policy: selected_policies.count(policy)
                for policy in sorted(set(selected_policies))
            }
            output[strategy]["selected_policy_counts"] = policy_counts
        calibration_scales = [
            row.get("calibration", {}).get("scale")
            for row in rows
            if "calibration" in row
        ]
        if calibration_scales:
            output[strategy]["calibration_scale_mean"] = float(np.mean(calibration_scales))
    return output


def _mean_metric(rows: list[dict[str, Any]], split: str, metric: str) -> float:
    return float(np.mean([row["splits"][split]["metrics"][metric] for row in rows]))


def _mean_region_metric(rows: list[dict[str, Any]], split: str, region: str, metric: str) -> float:
    values = [
        row["splits"][split]["region_metrics"][region]["metrics"][metric]
        for row in rows
        if row["splits"][split]["region_metrics"][region]["metrics"]
    ]
    return float(np.mean(values)) if values else 0.0


def run_probe(data: ProbeData, args: argparse.Namespace, *, known_noise_variance: float | None = None) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        seed = args.seed + repeat
        for strategy in args.strategy:
            results.append(
                run_sampling_strategy(
                    data,
                    strategy=strategy,
                    seed=seed,
                    initial_size=args.initial_size,
                    acquisition_size=args.acquisition_size,
                    prior_variance=args.prior_variance,
                    noise_variance=known_noise_variance,
                    noise_floor=args.noise_floor,
                    calibration_mode=args.calibration_mode,
                )
            )
    summary = _strategy_summary(results)
    decision = _decision_payload(
        summary,
        has_true_theta=data.true_theta is not None,
        active_strategy=args.active_strategy,
        require_region_preservation=args.require_region_preservation,
    )
    return {
        "label": data.label,
        "feature_names": data.feature_names,
        "n_points": int(len(data.target)),
        "split_sizes": {name: int(len(indices)) for name, indices in data.splits.items()},
        "initial_size": args.initial_size,
        "acquisition_size": args.acquisition_size,
        "repeats": args.repeats,
        "strategies": args.strategy,
        "active_strategy": args.active_strategy,
        "feature_mode": args.feature_mode,
        "calibration_mode": args.calibration_mode,
        "summary": summary,
        "decision": decision,
        "runs": results,
    }


def _decision_payload(
    summary: dict[str, Any],
    *,
    has_true_theta: bool,
    active_strategy: str,
    require_region_preservation: bool,
) -> dict[str, Any]:
    random_summary = summary.get("random")
    active_summary = summary.get(active_strategy)
    if not random_summary or not active_summary:
        return {
            "status": "inconclusive",
            "reason": f"random and active strategy '{active_strategy}' are required for a gate",
        }
    rmse_gain = random_summary["test_rmse_mean"] - active_summary["test_rmse_mean"]
    hot_gain = random_summary["test_hot_q90_rmse_mean"] - active_summary["test_hot_q90_rmse_mean"]
    gradient_gain = random_summary["test_gradient_q90_rmse_mean"] - active_summary["test_gradient_q90_rmse_mean"]
    calibration_ok = 0.75 <= active_summary["test_coverage90_mean"] <= 1.0
    source_ok = True
    if has_true_theta:
        source_ok = active_summary.get("source_recovery_pass_rate", 0.0) >= 0.8
    region_ok = hot_gain >= 0.0 and gradient_gain >= 0.0
    region_gate = region_ok if require_region_preservation else (hot_gain >= 0.0 or gradient_gain >= 0.0)
    effective = source_ok and calibration_ok and rmse_gain >= 0.0 and region_gate
    return {
        "status": "positive" if effective else "negative",
        "active_strategy": active_strategy,
        "source_recovery_ok": bool(source_ok),
        "calibration_ok": bool(calibration_ok),
        "region_preservation_ok": bool(region_ok),
        "require_region_preservation": bool(require_region_preservation),
        "rmse_gain_vs_random": float(rmse_gain),
        "hot_q90_rmse_gain_vs_random": float(hot_gain),
        "gradient_q90_rmse_gain_vs_random": float(gradient_gain),
        "interpretation": (
            "Bayesian low-dimensional inverse closure shows enough region-preserving signal for expansion."
            if effective
            else "Bayesian low-dimensional inverse closure is not yet region-preserving enough for AM-Bench/A100 expansion."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "synthetic":
        data = make_synthetic_data(
            n_grid=args.synthetic_grid,
            n_frames=args.synthetic_frames,
            seed=args.seed,
            noise_std=args.synthetic_noise_std,
        )
        if args.feature_mode == "physics_attention":
            data = with_physics_guided_attention_features(data)
        elif args.feature_mode == "heat_kernel":
            data = with_heat_kernel_features(data)
        return run_probe(data, args, known_noise_variance=args.synthetic_noise_std**2)
    if args.mode == "table":
        if args.table is None or args.split_manifest is None or args.target is None:
            raise ValueError("--table, --target, and --split-manifest are required for --mode table")
        data = make_table_data(
            table=args.table,
            target=args.target,
            split_manifest=args.split_manifest,
        )
        if args.feature_mode == "physics_attention":
            data = with_physics_guided_attention_features(data)
        elif args.feature_mode == "heat_kernel":
            data = with_heat_kernel_features(data)
        return run_probe(data, args)
    raise ValueError(f"Unsupported mode: {args.mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "table"], default="synthetic")
    parser.add_argument("--table", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--feature-mode", choices=["base", "physics_attention", "heat_kernel"], default="base")
    parser.add_argument("--strategy", action="append", default=["random", "uncertainty_source"])
    parser.add_argument("--active-strategy", default="uncertainty_source")
    parser.add_argument("--calibration-mode", choices=["none", "conformal90"], default="none")
    parser.add_argument("--require-region-preservation", action="store_true")
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--initial-size", type=int, default=32)
    parser.add_argument("--acquisition-size", type=int, default=96)
    parser.add_argument("--prior-variance", type=float, default=1e6)
    parser.add_argument("--noise-floor", type=float, default=1.0)
    parser.add_argument("--synthetic-grid", type=int, default=16)
    parser.add_argument("--synthetic-frames", type=int, default=8)
    parser.add_argument("--synthetic-noise-std", type=float, default=8.0)
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")
        print(f"Wrote: {args.json_output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
