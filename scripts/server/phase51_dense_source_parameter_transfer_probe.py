"""Phase 51 dense-to-sparse moving-source parameter transfer probe.

This local gate tests whether Phase 50 failed because the sparse observation
set could not identify moving-source parameters, or because the current
moving-source parameterization is not aligned with the AM-Bench field.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _load_phase50_module():
    module_path = Path(__file__).with_name("phase50_moving_source_inversion_probe.py")
    module_spec = importlib.util.spec_from_file_location("phase50_probe", module_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def _train_indices(data: Any, *, seed: int, size: int) -> np.ndarray:
    train_pool = np.asarray(data.splits["train"], dtype=int)
    if size <= 0 or size >= len(train_pool):
        return np.sort(train_pool)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(train_pool, size=size, replace=False))


def _fit_fixed_params(
    phase50: Any,
    data: Any,
    *,
    params: Any,
    train_indices: np.ndarray,
    ridge: float,
) -> dict[str, Any]:
    features = phase50._moving_features(data, params)
    theta = phase50._fit_theta(features, data.target, train_indices, ridge)
    mean = features @ theta
    train_residual = data.target[train_indices] - mean[train_indices]
    train_std = float(max(np.std(train_residual), 1e-6))
    calibration = phase50._conformal_scale(data, mean, train_std)
    std = np.full_like(mean, train_std * calibration["scale"], dtype=float)
    splits = {
        split: phase50._split_metrics(data, np.asarray(indices, dtype=int), mean, std)
        for split, indices in data.splits.items()
    }
    return {
        "params": params.__dict__,
        "theta": [float(value) for value in theta],
        "train_size": int(len(train_indices)),
        "train_residual_std": train_std,
        "calibration": calibration,
        "validation_objective": phase50._validation_objective(splits["val"]),
        "splits": splits,
        "parameter_recovery": phase50._parameter_recovery(data, params),
    }


def _parameter_delta(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {
        name: float(abs(float(left[name]) - float(right[name])))
        for name in sorted(left)
        if name in right
    }


def _run_once(phase50: Any, data: Any, args: argparse.Namespace, *, repeat: int) -> dict[str, Any]:
    seed = args.seed + repeat
    sparse_indices = _train_indices(data, seed=seed, size=args.sparse_fit_size)
    dense_indices = _train_indices(data, seed=seed + 100_000, size=args.dense_fit_size)

    sparse_fit = phase50.fit_moving_source(
        data,
        train_indices=sparse_indices,
        grid_mode=args.grid_mode,
        ridge=args.ridge,
    )
    dense_fit = phase50.fit_moving_source(
        data,
        train_indices=dense_indices,
        grid_mode=args.grid_mode,
        ridge=args.ridge,
    )

    sparse_search = _fit_fixed_params(
        phase50,
        data,
        params=sparse_fit.params,
        train_indices=sparse_indices,
        ridge=args.ridge,
    )
    dense_params_sparse_theta = _fit_fixed_params(
        phase50,
        data,
        params=dense_fit.params,
        train_indices=sparse_indices,
        ridge=args.ridge,
    )
    dense_upper_bound = _fit_fixed_params(
        phase50,
        data,
        params=dense_fit.params,
        train_indices=dense_indices,
        ridge=args.ridge,
    )
    return {
        "repeat": repeat,
        "seed": seed,
        "sparse_fit_size": int(len(sparse_indices)),
        "dense_fit_size": int(len(dense_indices)),
        "methods": {
            "sparse_search": sparse_search,
            "dense_params_sparse_theta": dense_params_sparse_theta,
            "dense_upper_bound": dense_upper_bound,
        },
        "parameter_delta_dense_vs_sparse": _parameter_delta(
            dense_upper_bound["params"],
            sparse_search["params"],
        ),
    }


def run_probe(data: Any, args: argparse.Namespace) -> dict[str, Any]:
    phase50 = _load_phase50_module()
    runs = [_run_once(phase50, data, args, repeat=repeat) for repeat in range(args.repeats)]
    summary = _summary(runs)
    return {
        "label": data.label,
        "mode": "dense_to_sparse_moving_source_transfer",
        "n_points": int(len(data.target)),
        "split_sizes": {name: int(len(indices)) for name, indices in data.splits.items()},
        "sparse_fit_size": args.sparse_fit_size,
        "dense_fit_size": args.dense_fit_size,
        "repeats": args.repeats,
        "grid_mode": args.grid_mode,
        "summary": summary,
        "decision": _decision(summary, has_true_params=data.true_theta is not None),
        "runs": runs,
    }


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    methods = ["sparse_search", "dense_params_sparse_theta", "dense_upper_bound"]
    output: dict[str, Any] = {}
    for method in methods:
        rows = [run["methods"][method] for run in runs]
        output[method] = {
            "test_rmse_mean": _mean_metric(rows, "test", "rmse"),
            "test_hot_q90_rmse_mean": _mean_region_metric(rows, "test", "hot_q90", "rmse"),
            "test_gradient_q90_rmse_mean": _mean_region_metric(rows, "test", "gradient_q90", "rmse"),
            "test_coverage90_mean": float(np.mean([row["splits"]["test"]["coverage_90"] for row in rows])),
            "validation_objective_mean": float(np.mean([row["validation_objective"] for row in rows])),
            "calibration_scale_mean": float(np.mean([row["calibration"]["scale"] for row in rows])),
        }
        recovery = [
            row.get("parameter_recovery", {}).get("pass")
            for row in rows
            if row.get("parameter_recovery") is not None
        ]
        if recovery:
            output[method]["parameter_recovery_pass_rate"] = float(np.mean([bool(value) for value in recovery]))
    output["gains_vs_sparse_search"] = {
        "dense_params_sparse_theta": _gain_payload(output["sparse_search"], output["dense_params_sparse_theta"]),
        "dense_upper_bound": _gain_payload(output["sparse_search"], output["dense_upper_bound"]),
    }
    output["parameter_delta_dense_vs_sparse_mean"] = _mean_parameter_delta(runs)
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


def _gain_payload(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    return {
        "test_rmse_gain": float(baseline["test_rmse_mean"] - candidate["test_rmse_mean"]),
        "test_hot_q90_rmse_gain": float(
            baseline["test_hot_q90_rmse_mean"] - candidate["test_hot_q90_rmse_mean"]
        ),
        "test_gradient_q90_rmse_gain": float(
            baseline["test_gradient_q90_rmse_mean"] - candidate["test_gradient_q90_rmse_mean"]
        ),
        "test_coverage90_gain": float(candidate["test_coverage90_mean"] - baseline["test_coverage90_mean"]),
    }


def _mean_parameter_delta(runs: list[dict[str, Any]]) -> dict[str, float]:
    names = sorted(runs[0]["parameter_delta_dense_vs_sparse"]) if runs else []
    return {
        name: float(np.mean([run["parameter_delta_dense_vs_sparse"][name] for run in runs]))
        for name in names
    }


def _decision(summary: dict[str, Any], *, has_true_params: bool) -> dict[str, Any]:
    transfer = summary["dense_params_sparse_theta"]
    upper = summary["dense_upper_bound"]
    transfer_gain = summary["gains_vs_sparse_search"]["dense_params_sparse_theta"]
    upper_gain = summary["gains_vs_sparse_search"]["dense_upper_bound"]
    transfer_region_ok = (
        transfer_gain["test_rmse_gain"] >= 0.0
        and transfer_gain["test_hot_q90_rmse_gain"] >= 0.0
        and transfer_gain["test_gradient_q90_rmse_gain"] >= 0.0
    )
    upper_region_ok = (
        upper_gain["test_rmse_gain"] >= 0.0
        and upper_gain["test_hot_q90_rmse_gain"] >= 0.0
        and upper_gain["test_gradient_q90_rmse_gain"] >= 0.0
    )
    coverage_ok = 0.75 <= transfer["test_coverage90_mean"] <= 1.0
    recovery_ok = True
    if has_true_params:
        recovery_ok = transfer.get("parameter_recovery_pass_rate", 0.0) >= 0.8
    positive = transfer_region_ok and upper_region_ok and coverage_ok and recovery_ok
    return {
        "status": "positive" if positive else "negative",
        "transfer_region_ok": bool(transfer_region_ok),
        "dense_upper_bound_region_ok": bool(upper_region_ok),
        "coverage_ok": bool(coverage_ok),
        "parameter_recovery_ok": bool(recovery_ok),
        "interpretation": (
            "Dense source-parameter fitting transfers to sparse refits without hurting global/hot/gradient metrics."
            if positive
            else "Dense source-parameter fitting does not yet transfer cleanly to the sparse AM-Bench gate."
        ),
    }


def make_data(args: argparse.Namespace) -> Any:
    phase50 = _load_phase50_module()
    return phase50.make_data(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "table"], default="synthetic")
    parser.add_argument("--table", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--grid-mode", choices=["tiny", "fast"], default="fast")
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--sparse-fit-size",
        type=int,
        default=256,
        help="Number of train rows used by the sparse search/refit path.",
    )
    parser.add_argument(
        "--dense-fit-size",
        type=int,
        default=0,
        help="Number of train rows used for dense parameter fitting; <=0 means all train rows.",
    )
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
