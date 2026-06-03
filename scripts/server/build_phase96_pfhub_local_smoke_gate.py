#!/usr/bin/env python3
"""Build the Phase 96 PFHub-style local smoke gate.

Phase 96 executes the Phase 95 local-only design contract on a deterministic
analytic heat-diffusion target. It is a small CPU smoke gate: it can allow a
future transfer-design phase, but it never opens A100 training by itself.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ALPHA = 0.04
SOURCE_BETA = 3.5
SOURCE_X0 = 0.28
SOURCE_VELOCITY = 0.42
SOURCE_SIGMA2 = 0.018
ADAPTIVE_BUDGET = 72
ENSEMBLE_SIZE = 9

METRIC_FIELDS = (
    "row_id",
    "method_id",
    "method_family",
    "mechanism",
    "selection_role",
    "gate_comparator",
    "train_points",
    "validation_rmse",
    "validation_pde_residual_rmse",
    "validation_hot_q90_rmse",
    "validation_gradient_q90_rmse",
    "test_rmse",
    "test_pde_residual_rmse",
    "test_hot_q90_rmse",
    "test_gradient_q90_rmse",
    "coverage_95",
    "global_delta_vs_vanilla",
    "hot_delta_vs_vanilla",
    "gradient_delta_vs_vanilla",
    "residual_delta_vs_vanilla",
    "validation_gate_pass",
    "test_audit_pass",
    "notes",
)

MECHANISM_FIELDS = (
    "mechanism_id",
    "mechanism",
    "comparator",
    "selected_by_validation",
    "validation_gate_pass",
    "test_audit_pass",
    "transfer_design_signal",
    "next_action",
    "reason",
)


@dataclass(frozen=True)
class Grid:
    x: np.ndarray
    t: np.ndarray
    x_values: np.ndarray
    t_values: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.x_values), len(self.t_values))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.9f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _default_paths(root: Path) -> dict[str, Path]:
    phase95 = root / "docs/results/phase95_pfhub_physics_design_gate"
    return {
        "phase95_gate": phase95 / "phase95_pfhub_physics_design_gate.json",
        "phase95_candidate_design": phase95 / "phase95_candidate_design.json",
    }


def make_grid(nx: int, nt: int) -> Grid:
    x_values = np.linspace(0.0, 1.0, nx)
    t_values = np.linspace(0.0, 1.0, nt)
    x_mesh, t_mesh = np.meshgrid(x_values, t_values, indexing="ij")
    return Grid(x=x_mesh.ravel(), t=t_mesh.ravel(), x_values=x_values, t_values=t_values)


def heat_kernel_source(
    x: np.ndarray,
    t: np.ndarray,
    *,
    offset: float = 0.0,
    width_scale: float = 1.0,
) -> np.ndarray:
    spread = SOURCE_SIGMA2 * width_scale + 2.0 * ALPHA * (t + 0.03)
    center = SOURCE_X0 + SOURCE_VELOCITY * t + offset
    amplitude = 1.0 - np.exp(-SOURCE_BETA * t)
    return amplitude * np.exp(-((x - center) ** 2) / (2.0 * spread)) / np.sqrt(spread)


def target_solution(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    smooth_component = 0.55 * np.sin(np.pi * x) * np.exp(-ALPHA * np.pi * np.pi * t)
    primary_source = 0.24 * heat_kernel_source(x, t, offset=0.0, width_scale=1.0)
    secondary_source = 0.07 * heat_kernel_source(x, t, offset=-0.08, width_scale=1.8)
    return smooth_component + primary_source + secondary_source


def low_order_features(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.ones_like(x),
            x,
            t,
            x * t,
            x**2,
            t**2,
            x**3,
            t**3,
        ],
        axis=1,
    )


def vanilla_features(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    pieces = [low_order_features(x, t)]
    trig: list[np.ndarray] = []
    for mode in (1, 2, 3):
        sx = np.sin(mode * np.pi * x)
        cx = np.cos(mode * np.pi * x)
        trig.extend([sx, cx, t * sx, t * cx])
    pieces.append(np.stack(trig, axis=1))
    return np.concatenate(pieces, axis=1)


def green_features(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    kernels: list[np.ndarray] = []
    for offset in (-0.10, -0.05, 0.0, 0.05, 0.10):
        for width_scale in (0.75, 1.25, 2.0):
            kernels.append(heat_kernel_source(x, t, offset=offset, width_scale=width_scale))
    primary = heat_kernel_source(x, t, offset=0.0, width_scale=1.0)
    kernels.extend([primary * x, primary * t])
    return np.concatenate([vanilla_features(x, t), np.stack(kernels, axis=1)], axis=1)


def fit_ridge(features: np.ndarray, target: np.ndarray, ridge: float = 1.0e-8) -> np.ndarray:
    lhs = features.T @ features + ridge * np.eye(features.shape[1])
    rhs = features.T @ target
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def predict(feature_fn: Any, weights: np.ndarray, grid: Grid) -> np.ndarray:
    return feature_fn(grid.x, grid.t) @ weights


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def _quantile_mask(values: np.ndarray, q: float) -> np.ndarray:
    threshold = float(np.quantile(values, q))
    threshold = min(max(threshold, float(values.min())), float(values.max()))
    mask = values >= threshold
    if not np.any(mask):
        mask[int(np.argmax(values))] = True
    return mask


def pde_residual(values: np.ndarray, grid: Grid) -> np.ndarray:
    field = values.reshape(grid.shape)
    dx = float(grid.x_values[1] - grid.x_values[0])
    dt = float(grid.t_values[1] - grid.t_values[0])
    field_t = np.gradient(field, dt, axis=1, edge_order=2)
    field_x = np.gradient(field, dx, axis=0, edge_order=2)
    field_xx = np.gradient(field_x, dx, axis=0, edge_order=2)
    return (field_t - ALPHA * field_xx).ravel()


def gradient_magnitude(values: np.ndarray, grid: Grid) -> np.ndarray:
    field = values.reshape(grid.shape)
    dx = float(grid.x_values[1] - grid.x_values[0])
    field_x = np.gradient(field, dx, axis=0, edge_order=2)
    return np.abs(field_x).ravel()


def metric_bundle(actual: np.ndarray, predicted: np.ndarray, grid: Grid) -> dict[str, float]:
    actual_residual = pde_residual(actual, grid)
    predicted_residual = pde_residual(predicted, grid)
    hot_mask = _quantile_mask(actual, 0.9)
    gradient_mask = _quantile_mask(gradient_magnitude(actual, grid), 0.9)
    return {
        "rmse": _rmse(actual, predicted),
        "pde_residual_rmse": _rmse(actual_residual, predicted_residual),
        "hot_q90_rmse": _rmse(actual[hot_mask], predicted[hot_mask]),
        "gradient_q90_rmse": _rmse(actual[gradient_mask], predicted[gradient_mask]),
    }


def adaptive_indices(train_grid: Grid, budget: int = ADAPTIVE_BUDGET) -> np.ndarray:
    source_score = np.abs(heat_kernel_source(train_grid.x, train_grid.t))
    top_count = budget // 2
    top_idx = np.argsort(source_score)[-top_count:]
    remaining = np.setdiff1d(np.arange(len(train_grid.x)), top_idx)
    order = np.argsort(train_grid.t[remaining] + 0.37 * train_grid.x[remaining])
    diverse = remaining[order]
    diverse_idx = diverse[np.linspace(0, len(diverse) - 1, budget - top_count, dtype=int)]
    selected = np.unique(np.concatenate([top_idx, diverse_idx]))
    cursor = 0
    while len(selected) < budget:
        candidate = diverse[cursor % len(diverse)]
        if candidate not in selected:
            selected = np.append(selected, candidate)
        cursor += 1
    return np.sort(selected[:budget])


def random_indices(train_grid: Grid, budget: int = ADAPTIVE_BUDGET) -> np.ndarray:
    rng = np.random.default_rng(960)
    return np.sort(rng.choice(len(train_grid.x), size=budget, replace=False))


def ensemble_predict(
    train_grid: Grid,
    train_target: np.ndarray,
    selected: np.ndarray,
    eval_grid: Grid,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = 0.35 + np.abs(heat_kernel_source(train_grid.x[selected], train_grid.t[selected]))
    weights = weights / weights.sum()
    predictions: list[np.ndarray] = []
    for _ in range(ENSEMBLE_SIZE):
        boot = rng.choice(selected, size=len(selected), replace=True, p=weights)
        model = fit_ridge(green_features(train_grid.x[boot], train_grid.t[boot]), train_target[boot], 1.0e-7)
        predictions.append(predict(green_features, model, eval_grid))
    stacked = np.stack(predictions, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0, ddof=1)


def conformal_coverage(
    train_grid: Grid,
    train_target: np.ndarray,
    selected: np.ndarray,
    validation_grid: Grid,
    validation_target: np.ndarray,
    test_grid: Grid,
    test_target: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, float]:
    validation_mean, validation_std = ensemble_predict(
        train_grid, train_target, selected, validation_grid, seed=seed
    )
    test_mean, test_std = ensemble_predict(train_grid, train_target, selected, test_grid, seed=seed)
    normalized_residual = np.abs(validation_target - validation_mean) / (validation_std + 1.0e-12)
    scale = float(np.quantile(normalized_residual, 0.95))
    interval = scale * (test_std + 1.0e-12)
    coverage = float(np.mean(np.abs(test_target - test_mean) <= interval))
    return test_mean, coverage


def _fit_and_score(
    method_id: str,
    method_family: str,
    mechanism: str,
    selection_role: str,
    gate_comparator: bool,
    feature_fn: Any,
    train_grid: Grid,
    train_target: np.ndarray,
    validation_grid: Grid,
    validation_target: np.ndarray,
    test_grid: Grid,
    test_target: np.ndarray,
    *,
    selected: np.ndarray | None = None,
    coverage: float | None = None,
    test_prediction: np.ndarray | None = None,
) -> dict[str, Any]:
    if selected is None:
        selected = np.arange(len(train_grid.x))
    if test_prediction is None:
        weights = fit_ridge(feature_fn(train_grid.x[selected], train_grid.t[selected]), train_target[selected])
        validation_prediction = predict(feature_fn, weights, validation_grid)
        test_prediction = predict(feature_fn, weights, test_grid)
    else:
        weights = fit_ridge(feature_fn(train_grid.x[selected], train_grid.t[selected]), train_target[selected])
        validation_prediction = predict(feature_fn, weights, validation_grid)

    validation = metric_bundle(validation_target, validation_prediction, validation_grid)
    test = metric_bundle(test_target, test_prediction, test_grid)
    return {
        "method_id": method_id,
        "method_family": method_family,
        "mechanism": mechanism,
        "selection_role": selection_role,
        "gate_comparator": gate_comparator,
        "train_points": int(len(selected)),
        "validation_rmse": validation["rmse"],
        "validation_pde_residual_rmse": validation["pde_residual_rmse"],
        "validation_hot_q90_rmse": validation["hot_q90_rmse"],
        "validation_gradient_q90_rmse": validation["gradient_q90_rmse"],
        "test_rmse": test["rmse"],
        "test_pde_residual_rmse": test["pde_residual_rmse"],
        "test_hot_q90_rmse": test["hot_q90_rmse"],
        "test_gradient_q90_rmse": test["gradient_q90_rmse"],
        "coverage_95": coverage,
        "validation_gate_pass": False,
        "test_audit_pass": False,
        "notes": "",
    }


def run_smoke_experiments() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_grid = make_grid(19, 15)
    validation_grid = make_grid(23, 17)
    test_grid = make_grid(41, 31)
    train_target = target_solution(train_grid.x, train_grid.t)
    validation_target = target_solution(validation_grid.x, validation_grid.t)
    test_target = target_solution(test_grid.x, test_grid.t)

    random_selected = random_indices(train_grid)
    adaptive_selected = adaptive_indices(train_grid)
    adaptive_prediction, adaptive_coverage = conformal_coverage(
        train_grid,
        train_target,
        adaptive_selected,
        validation_grid,
        validation_target,
        test_grid,
        test_target,
        seed=1960,
    )
    random_prediction, random_coverage = conformal_coverage(
        train_grid,
        train_target,
        random_selected,
        validation_grid,
        validation_target,
        test_grid,
        test_target,
        seed=960,
    )

    rows = [
        _fit_and_score(
            "low_order_interpolation",
            "baseline",
            "mean_or_low_order_interpolation",
            "gate_comparator",
            True,
            low_order_features,
            train_grid,
            train_target,
            validation_grid,
            validation_target,
            test_grid,
            test_target,
        ),
        _fit_and_score(
            "vanilla_deterministic_surrogate",
            "baseline",
            "vanilla_pinn_same_budget",
            "primary_comparator",
            True,
            vanilla_features,
            train_grid,
            train_target,
            validation_grid,
            validation_target,
            test_grid,
            test_target,
        ),
        _fit_and_score(
            "fixed_green_function_features",
            "candidate",
            "fixed_green_function_features",
            "validation_selected",
            False,
            green_features,
            train_grid,
            train_target,
            validation_grid,
            validation_target,
            test_grid,
            test_target,
        ),
        _fit_and_score(
            "random_collocation_same_budget",
            "baseline",
            "random_collocation_same_budget",
            "adaptive_comparator",
            True,
            green_features,
            train_grid,
            train_target,
            validation_grid,
            validation_target,
            test_grid,
            test_target,
            selected=random_selected,
            coverage=random_coverage,
            test_prediction=random_prediction,
        ),
        _fit_and_score(
            "bayesian_adaptive_collocation",
            "candidate",
            "bayesian_adaptive_collocation",
            "validation_selected",
            False,
            green_features,
            train_grid,
            train_target,
            validation_grid,
            validation_target,
            test_grid,
            test_target,
            selected=adaptive_selected,
            coverage=adaptive_coverage,
            test_prediction=adaptive_prediction,
        ),
    ]

    vanilla = next(row for row in rows if row["method_id"] == "vanilla_deterministic_surrogate")
    random_row = next(row for row in rows if row["method_id"] == "random_collocation_same_budget")
    for row in rows:
        row["global_delta_vs_vanilla"] = vanilla["test_rmse"] - row["test_rmse"]
        row["hot_delta_vs_vanilla"] = vanilla["test_hot_q90_rmse"] - row["test_hot_q90_rmse"]
        row["gradient_delta_vs_vanilla"] = (
            vanilla["test_gradient_q90_rmse"] - row["test_gradient_q90_rmse"]
        )
        row["residual_delta_vs_vanilla"] = (
            vanilla["test_pde_residual_rmse"] - row["test_pde_residual_rmse"]
        )

    fixed = next(row for row in rows if row["method_id"] == "fixed_green_function_features")
    fixed["validation_gate_pass"] = bool(
        fixed["validation_rmse"] <= vanilla["validation_rmse"]
        and fixed["validation_pde_residual_rmse"] <= vanilla["validation_pde_residual_rmse"]
        and (
            fixed["validation_hot_q90_rmse"] < vanilla["validation_hot_q90_rmse"]
            or fixed["validation_gradient_q90_rmse"] < vanilla["validation_gradient_q90_rmse"]
        )
    )
    fixed["test_audit_pass"] = bool(
        fixed["test_rmse"] <= vanilla["test_rmse"]
        and fixed["test_pde_residual_rmse"] <= vanilla["test_pde_residual_rmse"]
        and fixed["test_hot_q90_rmse"] < vanilla["test_hot_q90_rmse"]
        and fixed["test_gradient_q90_rmse"] < vanilla["test_gradient_q90_rmse"]
    )
    fixed["notes"] = "validation-selected fixed physics features; synthetic/PFHub-style evidence only"

    adaptive = next(row for row in rows if row["method_id"] == "bayesian_adaptive_collocation")
    adaptive["validation_gate_pass"] = bool(
        adaptive["validation_rmse"] <= random_row["validation_rmse"]
        and (
            adaptive["validation_hot_q90_rmse"] < random_row["validation_hot_q90_rmse"]
            or adaptive["validation_gradient_q90_rmse"] < random_row["validation_gradient_q90_rmse"]
        )
    )
    adaptive["test_audit_pass"] = bool(
        adaptive["test_rmse"] <= random_row["test_rmse"]
        and adaptive["test_hot_q90_rmse"] <= random_row["test_hot_q90_rmse"]
        and adaptive["test_gradient_q90_rmse"] <= random_row["test_gradient_q90_rmse"]
        and adaptive["coverage_95"] is not None
        and adaptive["coverage_95"] >= 0.8
    )
    adaptive["notes"] = (
        "region-focused adaptive sampling diagnostic; blocked unless global and coverage guards pass"
    )

    low_order = next(row for row in rows if row["method_id"] == "low_order_interpolation")
    low_order["notes"] = "simple low-order comparator required by Phase 95"
    vanilla["notes"] = "primary deterministic surrogate comparator"
    random_row["notes"] = "same-budget random collocation comparator for adaptive sampling"

    for index, row in enumerate(rows, start=1):
        row["row_id"] = f"P96-MET-{index:03d}"

    target_manifest = {
        "target_id": "phase96_pfhub_style_heat_source_v1",
        "target_type": "manufactured_heat_diffusion_with_moving_source",
        "alpha": ALPHA,
        "source_beta": SOURCE_BETA,
        "source_x0": SOURCE_X0,
        "source_velocity": SOURCE_VELOCITY,
        "source_sigma2": SOURCE_SIGMA2,
        "splits": {
            "train_grid": {"nx": len(train_grid.x_values), "nt": len(train_grid.t_values)},
            "validation_grid": {
                "nx": len(validation_grid.x_values),
                "nt": len(validation_grid.t_values),
            },
            "test_grid": {"nx": len(test_grid.x_values), "nt": len(test_grid.t_values)},
        },
        "adaptive_budget": ADAPTIVE_BUDGET,
        "ensemble_size": ENSEMBLE_SIZE,
        "selection_policy": "validation-only mechanism selection; test metrics are audit fields",
        "not_a_claim": [
            "not AM-Bench evidence",
            "not proof of registered-source transfer",
            "not permission for A100 training",
        ],
    }
    return rows, target_manifest


def build_mechanism_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fixed = next(row for row in metric_rows if row["method_id"] == "fixed_green_function_features")
    adaptive = next(row for row in metric_rows if row["method_id"] == "bayesian_adaptive_collocation")
    random_row = next(row for row in metric_rows if row["method_id"] == "random_collocation_same_budget")
    vanilla = next(row for row in metric_rows if row["method_id"] == "vanilla_deterministic_surrogate")
    return [
        {
            "mechanism_id": "P96-MECH-001",
            "mechanism": "fixed_green_function_features",
            "comparator": "vanilla_deterministic_surrogate",
            "selected_by_validation": fixed["validation_gate_pass"],
            "validation_gate_pass": fixed["validation_gate_pass"],
            "test_audit_pass": fixed["test_audit_pass"],
            "transfer_design_signal": fixed["validation_gate_pass"] and fixed["test_audit_pass"],
            "next_action": (
                "open Phase 97 transfer design gate"
                if fixed["validation_gate_pass"] and fixed["test_audit_pass"]
                else "close as synthetic diagnostic"
            ),
            "reason": (
                f"validation RMSE {fixed['validation_rmse']:.6g} vs "
                f"{vanilla['validation_rmse']:.6g}; test hot/gradient deltas "
                f"{fixed['hot_delta_vs_vanilla']:.6g}/"
                f"{fixed['gradient_delta_vs_vanilla']:.6g}"
            ),
        },
        {
            "mechanism_id": "P96-MECH-002",
            "mechanism": "bayesian_adaptive_collocation",
            "comparator": "random_collocation_same_budget",
            "selected_by_validation": adaptive["validation_gate_pass"],
            "validation_gate_pass": adaptive["validation_gate_pass"],
            "test_audit_pass": adaptive["test_audit_pass"],
            "transfer_design_signal": adaptive["validation_gate_pass"] and adaptive["test_audit_pass"],
            "next_action": (
                "open only if global and coverage guards pass"
                if adaptive["validation_gate_pass"]
                else "keep as diagnostic until global/coverage guards pass"
            ),
            "reason": (
                f"validation RMSE {adaptive['validation_rmse']:.6g} vs random "
                f"{random_row['validation_rmse']:.6g}; coverage "
                f"{adaptive['coverage_95']:.6g}"
            ),
        },
    ]


def build_gate(
    *,
    phase95_gate: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    mechanism_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase95_allows = bool(phase95_gate.get("phase96_local_smoke_allowed"))
    transfer_signals = [row for row in mechanism_rows if row["transfer_design_signal"]]
    if not phase95_allows:
        status = "blocked_by_phase95_design_gate"
        next_action = "repair or reopen Phase 95 before running local smoke"
        phase97_allowed = False
    elif transfer_signals:
        status = "local_smoke_positive_transfer_design_only"
        next_action = "enter Phase 97 AM-Bench/external transfer design gate; do not start A100 training"
        phase97_allowed = True
    else:
        status = "closed_local_smoke_negative"
        next_action = "close as PFHub-style diagnostic and choose the next local/no-training candidate"
        phase97_allowed = False

    return {
        "status": status,
        "source_phase95_status": phase95_gate.get("status"),
        "phase97_transfer_design_allowed": phase97_allowed,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "submission_ready": False,
        "metric_rows": len(metric_rows),
        "mechanism_rows": len(mechanism_rows),
        "validation_selected_mechanisms": sum(
            1 for row in mechanism_rows if row["selected_by_validation"]
        ),
        "transfer_design_signals": len(transfer_signals),
        "positive_mechanisms": [row["mechanism"] for row in transfer_signals],
        "blocked_mechanisms": [
            row["mechanism"] for row in mechanism_rows if not row["transfer_design_signal"]
        ],
        "next_action": next_action,
        "required_before_a100_training": [
            "Phase 97 transfer design gate",
            "Phase 98 AM-Bench or external baseline-first smoke",
            "non-worse global/hot/gradient metrics against strong baselines",
            "no-test-leakage candidate selection",
            "server validation from a pushed commit",
        ],
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_csv_value(row.get(key)).replace("\n", " ") for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def build_markdown(
    gate: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    mechanism_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Phase 96 PFHub-Style Local Smoke Gate",
            "",
            "## Gate Decision",
            "",
            f"Status: `{gate['status']}`.",
            f"Phase 97 transfer design allowed: `{str(gate['phase97_transfer_design_allowed']).lower()}`.",
            f"A100 training allowed now: `{str(gate['a100_training_allowed_now']).lower()}`.",
            f"A100-SXM4-80GB request now: `{str(gate['a100_80gb_request_now']).lower()}`.",
            "",
            "Phase 96 is a local smoke gate only. It can open transfer-design work, but it is not AM-Bench evidence.",
            "",
            "## Mechanism Decisions",
            "",
            _markdown_table(
                mechanism_rows,
                [
                    ("mechanism_id", "Mechanism"),
                    ("mechanism", "Name"),
                    ("comparator", "Comparator"),
                    ("validation_gate_pass", "Validation pass"),
                    ("test_audit_pass", "Test audit"),
                    ("next_action", "Next action"),
                ],
            ),
            "",
            "## Metric Summary",
            "",
            _markdown_table(
                metric_rows,
                [
                    ("method_id", "Method"),
                    ("validation_rmse", "Val RMSE"),
                    ("test_rmse", "Test RMSE"),
                    ("test_pde_residual_rmse", "Residual RMSE"),
                    ("test_hot_q90_rmse", "Hot q90"),
                    ("test_gradient_q90_rmse", "Gradient q90"),
                    ("coverage_95", "Coverage"),
                ],
            ),
            "",
            "## Next Action",
            "",
            gate["next_action"],
            "",
        ]
    )


def build_package(
    root: Path,
    output_dir: Path,
    paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    resolved = _default_paths(root)
    if paths:
        resolved.update(paths)

    phase95_gate = _read_json(resolved["phase95_gate"])
    phase95_design = _read_json(resolved["phase95_candidate_design"])

    if phase95_gate.get("phase96_local_smoke_allowed"):
        metric_rows, target_manifest = run_smoke_experiments()
        mechanism_rows = build_mechanism_rows(metric_rows)
    else:
        metric_rows = []
        mechanism_rows = []
        target_manifest = {
            "target_id": "not_run",
            "reason": "Phase 95 did not allow Phase 96 local smoke",
        }
    gate = build_gate(
        phase95_gate=phase95_gate,
        metric_rows=metric_rows,
        mechanism_rows=mechanism_rows,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_path = output_dir / "phase96_local_smoke_metric_table.csv"
    mechanism_path = output_dir / "phase96_mechanism_decision_table.csv"
    target_path = output_dir / "phase96_pfhub_style_target_manifest.json"
    gate_path = output_dir / "phase96_pfhub_local_smoke_gate.json"
    markdown_path = output_dir / "phase96_pfhub_local_smoke_gate.md"
    manifest_path = output_dir / "phase96_pfhub_local_smoke_gate_manifest.json"

    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_csv(mechanism_path, mechanism_rows, MECHANISM_FIELDS)
    _write_json(target_path, target_manifest)
    _write_json(gate_path, gate)
    markdown_path.write_text(build_markdown(gate, metric_rows, mechanism_rows), encoding="utf-8")

    manifest = {
        "phase": 96,
        "objective": "pfhub_style_local_smoke_gate",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "metric_table": _display_path(metric_path, root),
            "mechanism_decision_table": _display_path(mechanism_path, root),
            "target_manifest": _display_path(target_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "metric_rows": len(metric_rows),
            "mechanism_rows": len(mechanism_rows),
            "transfer_design_signals": gate["transfer_design_signals"],
        },
        "gate": gate,
        "phase95_gate": {
            "status": phase95_gate.get("status"),
            "phase96_local_smoke_allowed": phase95_gate.get("phase96_local_smoke_allowed"),
            "a100_training_allowed_now": phase95_gate.get("a100_training_allowed_now"),
        },
        "phase95_candidate_design": {
            "candidate_id": phase95_design.get("candidate_id"),
            "source_candidate": phase95_design.get("source_candidate"),
            "selected_benchmark_style": phase95_design.get("selected_benchmark_style"),
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase96_pfhub_local_smoke_gate"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    manifest = build_package(root=root, output_dir=output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
