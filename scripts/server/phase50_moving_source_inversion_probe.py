"""Phase 50 explicit nonlinear moving-source inversion probe.

This is a local feasibility gate. It fits interpretable moving heat-source
parameters with train/validation data only, then evaluates sparse prediction on
held-out test points.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gnnpinn.eval.baselines import regression_metric_table


Z90 = 1.6448536269514722


@dataclass(frozen=True)
class SourceParams:
    start_x: float
    span_x: float
    center_y: float
    sine_y_amp: float
    core_width: float
    tail_width: float
    tail_decay: float


@dataclass(frozen=True)
class FitResult:
    params: SourceParams
    theta: np.ndarray
    validation_objective: float
    validation_metrics: dict[str, Any]
    train_residual_std: float


def _load_phase46_module():
    module_path = Path(__file__).with_name("phase46_bayesian_inverse_closure_probe.py")
    module_spec = importlib.util.spec_from_file_location("phase46_probe", module_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def _scale01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = float(values.max() - values.min()) if values.size else 0.0
    if span <= 0.0:
        return np.zeros_like(values)
    return (values - float(values.min())) / span


def _moving_features(data: Any, params: SourceParams) -> np.ndarray:
    x = _scale01(data.cols)
    y = _scale01(data.rows)
    t = _scale01(data.frames)
    source_x = params.start_x + params.span_x * t
    source_y = params.center_y + params.sine_y_amp * np.sin(2.0 * np.pi * t)
    radius2 = (x - source_x) ** 2 + (y - source_y) ** 2
    core = np.exp(-0.5 * radius2 / max(params.core_width**2, 1e-8))
    tail = np.exp(-0.5 * radius2 / max(params.tail_width**2, 1e-8)) * np.exp(-t / max(params.tail_decay, 1e-8))
    return np.column_stack(
        [
            np.ones_like(x),
            core,
            tail,
            x,
            y,
            t,
        ]
    )


def _parameter_grid(mode: str) -> list[SourceParams]:
    if mode == "fast":
        starts = [0.20, 0.25, 0.30]
        spans = [0.45, 0.50, 0.60]
        centers = [0.48, 0.52, 0.56]
        sine_amps = [0.0, 0.05]
        core_widths = [0.08, 0.11, 0.14]
        tail_widths = [0.18, 0.24, 0.32]
        decays = [0.7, 1.5]
    elif mode == "tiny":
        starts = [0.20, 0.25]
        spans = [0.50, 0.60]
        centers = [0.50, 0.55]
        sine_amps = [0.0, 0.05]
        core_widths = [0.10, 0.14]
        tail_widths = [0.22, 0.32]
        decays = [1.0]
    else:
        raise ValueError(f"Unsupported grid mode: {mode}")
    return [
        SourceParams(start, span, center, sine_amp, core_width, tail_width, decay)
        for start in starts
        for span in spans
        for center in centers
        for sine_amp in sine_amps
        for core_width in core_widths
        for tail_width in tail_widths
        for decay in decays
    ]


def _fit_theta(features: np.ndarray, target: np.ndarray, indices: np.ndarray, ridge: float) -> np.ndarray:
    x_train = features[indices]
    y_train = target[indices]
    lhs = x_train.T @ x_train + ridge * np.eye(x_train.shape[1])
    rhs = x_train.T @ y_train
    return np.linalg.solve(lhs, rhs)


def _gradient_scores(data: Any, values: np.ndarray) -> np.ndarray:
    phase46 = _load_phase46_module()
    return phase46._gradient_scores(data, values)


def _quantile_indices(values: np.ndarray, indices: np.ndarray, quantile: float) -> np.ndarray:
    if len(indices) == 0:
        return np.asarray([], dtype=int)
    threshold = float(np.quantile(values[indices], quantile))
    return indices[values[indices] >= threshold]


def _split_metrics(data: Any, indices: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict[str, Any]:
    idx = np.asarray(indices, dtype=int)
    y_true = data.target[idx]
    payload: dict[str, Any] = {
        "n_points": int(len(idx)),
        "metrics": regression_metric_table(y_true.tolist(), mean[idx].tolist()),
        "coverage_90": _coverage(y_true, mean[idx], std[idx]),
        "mean_predictive_std": float(np.mean(std[idx])) if len(idx) else 0.0,
    }
    hot_indices = _quantile_indices(data.target, idx, 0.9)
    gradient_scores = _gradient_scores(data, data.target)
    grad_indices = _quantile_indices(gradient_scores, idx, 0.9)
    payload["region_metrics"] = {
        "hot_q90": {
            "n_points": int(len(hot_indices)),
            "metrics": regression_metric_table(data.target[hot_indices].tolist(), mean[hot_indices].tolist())
            if len(hot_indices)
            else {},
        },
        "gradient_q90": {
            "n_points": int(len(grad_indices)),
            "metrics": regression_metric_table(data.target[grad_indices].tolist(), mean[grad_indices].tolist())
            if len(grad_indices)
            else {},
        },
    }
    return payload


def _coverage(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    lower = mean - Z90 * std
    upper = mean + Z90 * std
    return float(np.mean((y_true >= lower) & (y_true <= upper))) if len(y_true) else 0.0


def _validation_objective(metrics: dict[str, Any]) -> float:
    base = metrics["metrics"]["rmse"]
    regions = metrics["region_metrics"]
    hot = regions["hot_q90"]["metrics"].get("rmse", base)
    gradient = regions["gradient_q90"]["metrics"].get("rmse", base)
    return float(base + 0.5 * hot + 0.5 * gradient)


def _selected_train_indices(data: Any, *, seed: int, initial_size: int, acquisition_size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    train_pool = np.asarray(data.splits["train"], dtype=int)
    selected_size = initial_size + acquisition_size
    if selected_size > len(train_pool):
        raise ValueError("initial-size + acquisition-size cannot exceed train split size")
    return np.sort(rng.choice(train_pool, size=selected_size, replace=False))


def fit_moving_source(
    data: Any,
    *,
    train_indices: np.ndarray,
    grid_mode: str,
    ridge: float,
) -> FitResult:
    val_indices = np.asarray(data.splits["val"], dtype=int)
    best: FitResult | None = None
    for params in _parameter_grid(grid_mode):
        features = _moving_features(data, params)
        theta = _fit_theta(features, data.target, train_indices, ridge)
        mean = features @ theta
        train_residual = data.target[train_indices] - mean[train_indices]
        train_std = float(max(np.std(train_residual), 1e-6))
        std = np.full_like(mean, train_std, dtype=float)
        val_metrics = _split_metrics(data, val_indices, mean, std)
        objective = _validation_objective(val_metrics)
        if best is None or objective < best.validation_objective:
            best = FitResult(
                params=params,
                theta=theta,
                validation_objective=objective,
                validation_metrics=val_metrics,
                train_residual_std=train_std,
            )
    if best is None:
        raise RuntimeError("No moving-source candidates were evaluated")
    return best


def _conformal_scale(data: Any, mean: np.ndarray, base_std: float) -> dict[str, Any]:
    val_indices = np.asarray(data.splits["val"], dtype=int)
    safe_std = max(base_std, 1e-6)
    nonconformity = np.abs(data.target[val_indices] - mean[val_indices]) / safe_std
    q90 = float(np.quantile(nonconformity, 0.9)) if len(nonconformity) else Z90
    return {
        "mode": "conformal90",
        "scale": float(max(1.0, q90 / Z90)),
        "nonconformity_q90": q90,
        "validation_points": int(len(val_indices)),
    }


def run_probe(data: Any, args: argparse.Namespace) -> dict[str, Any]:
    runs = []
    for repeat in range(args.repeats):
        train_indices = _selected_train_indices(
            data,
            seed=args.seed + repeat,
            initial_size=args.initial_size,
            acquisition_size=args.acquisition_size,
        )
        fit = fit_moving_source(
            data,
            train_indices=train_indices,
            grid_mode=args.grid_mode,
            ridge=args.ridge,
        )
        features = _moving_features(data, fit.params)
        mean = features @ fit.theta
        calibration = _conformal_scale(data, mean, fit.train_residual_std)
        std = np.full_like(mean, fit.train_residual_std * calibration["scale"], dtype=float)
        split_payload = {
            split: _split_metrics(data, np.asarray(indices, dtype=int), mean, std)
            for split, indices in data.splits.items()
        }
        runs.append(
            {
                "repeat": repeat,
                "seed": args.seed + repeat,
                "selected_size": int(len(train_indices)),
                "params": fit.params.__dict__,
                "theta": [float(value) for value in fit.theta],
                "validation_objective": fit.validation_objective,
                "validation_metrics": fit.validation_metrics,
                "train_residual_std": fit.train_residual_std,
                "calibration": calibration,
                "splits": split_payload,
                "parameter_recovery": _parameter_recovery(data, fit.params),
            }
        )
    summary = _summary(runs)
    decision = _decision(summary, has_true_params=data.true_theta is not None)
    return {
        "label": data.label,
        "mode": "moving_source_inversion",
        "n_points": int(len(data.target)),
        "split_sizes": {name: int(len(indices)) for name, indices in data.splits.items()},
        "initial_size": args.initial_size,
        "acquisition_size": args.acquisition_size,
        "repeats": args.repeats,
        "grid_mode": args.grid_mode,
        "summary": summary,
        "decision": decision,
        "runs": runs,
    }


def _parameter_recovery(data: Any, params: SourceParams) -> dict[str, Any] | None:
    if data.true_theta is None:
        return None
    expected = {
        "start_x": 0.25,
        "span_x": 0.50,
        "center_y": 0.52,
        "sine_y_amp": 0.05,
        "core_width": 0.105,
    }
    errors = {
        name: abs(float(getattr(params, name)) - expected_value)
        for name, expected_value in expected.items()
    }
    return {
        "expected": expected,
        "absolute_errors": errors,
        "max_key_parameter_error": float(max(errors.values())),
        "pass": bool(
            errors["start_x"] <= 0.06
            and errors["span_x"] <= 0.10
            and errors["center_y"] <= 0.06
            and errors["sine_y_amp"] <= 0.06
            and errors["core_width"] <= 0.05
        ),
    }


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    output = {
        "test_rmse_mean": _mean_metric(runs, "test", "rmse"),
        "test_hot_q90_rmse_mean": _mean_region_metric(runs, "test", "hot_q90", "rmse"),
        "test_gradient_q90_rmse_mean": _mean_region_metric(runs, "test", "gradient_q90", "rmse"),
        "test_coverage90_mean": float(np.mean([run["splits"]["test"]["coverage_90"] for run in runs])),
        "calibration_scale_mean": float(np.mean([run["calibration"]["scale"] for run in runs])),
        "validation_objective_mean": float(np.mean([run["validation_objective"] for run in runs])),
    }
    recovery = [
        run.get("parameter_recovery", {}).get("pass")
        for run in runs
        if run.get("parameter_recovery") is not None
    ]
    if recovery:
        output["parameter_recovery_pass_rate"] = float(np.mean([bool(value) for value in recovery]))
    return output


def _mean_metric(runs: list[dict[str, Any]], split: str, metric: str) -> float:
    return float(np.mean([run["splits"][split]["metrics"][metric] for run in runs]))


def _mean_region_metric(runs: list[dict[str, Any]], split: str, region: str, metric: str) -> float:
    values = [
        run["splits"][split]["region_metrics"][region]["metrics"][metric]
        for run in runs
        if run["splits"][split]["region_metrics"][region]["metrics"]
    ]
    return float(np.mean(values)) if values else 0.0


def _decision(summary: dict[str, Any], *, has_true_params: bool) -> dict[str, Any]:
    coverage_ok = 0.75 <= summary["test_coverage90_mean"] <= 1.0
    recovery_ok = True
    if has_true_params:
        recovery_ok = summary.get("parameter_recovery_pass_rate", 0.0) >= 0.8
    return {
        "status": "positive" if coverage_ok and recovery_ok else "negative",
        "coverage_ok": bool(coverage_ok),
        "parameter_recovery_ok": bool(recovery_ok),
        "interpretation": (
            "Moving-source parameter inversion is locally identifiable enough for expansion."
            if coverage_ok and recovery_ok
            else "Moving-source parameter inversion is not yet strong enough for expansion."
        ),
    }


def make_data(args: argparse.Namespace) -> Any:
    phase46 = _load_phase46_module()
    if args.mode == "synthetic":
        return phase46.make_synthetic_data(
            n_grid=args.synthetic_grid,
            n_frames=args.synthetic_frames,
            seed=args.seed,
            noise_std=args.synthetic_noise_std,
        )
    if args.mode == "table":
        if args.table is None or args.target is None or args.split_manifest is None:
            raise ValueError("--table, --target, and --split-manifest are required for --mode table")
        return phase46.make_table_data(
            table=args.table,
            target=args.target,
            split_manifest=args.split_manifest,
        )
    raise ValueError(f"Unsupported mode: {args.mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "table"], default="synthetic")
    parser.add_argument("--table", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--grid-mode", choices=["tiny", "fast"], default="fast")
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--initial-size", type=int, default=32)
    parser.add_argument("--acquisition-size", type=int, default=96)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--synthetic-grid", type=int, default=16)
    parser.add_argument("--synthetic-frames", type=int, default=8)
    parser.add_argument("--synthetic-noise-std", type=float, default=8.0)
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = make_data(args)
    payload = run_probe(data, args)
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
