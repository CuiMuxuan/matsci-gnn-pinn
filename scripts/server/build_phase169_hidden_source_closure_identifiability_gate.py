#!/usr/bin/env python3
"""Build Phase 169 hidden-source/closure identifiability gate.

Phase 169 is a no-training numerical identifiability gate. It tests whether a
low-dimensional hidden moving-source and residual-closure parameterization is
recoverable from sparse synthetic sensors before any PINN training is reopened.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.ensemble import ExtraTreesRegressor
except Exception:  # pragma: no cover - optional in minimal environments.
    ExtraTreesRegressor = None  # type: ignore[assignment]


DEFAULT_OUTPUT_DIR = Path(
    "docs/results/phase169_hidden_source_closure_identifiability_gate"
)

PHASE_INPUTS = {
    "phase168_gate": Path(
        "docs/results/phase168_hidden_source_closure_redesign_gate/"
        "phase168_hidden_source_closure_redesign_gate.json"
    ),
    "phase168_design_table": Path(
        "docs/results/phase168_hidden_source_closure_redesign_gate/"
        "phase168_phase169_design_contract_table.csv"
    ),
}

CASE_FIELDS = (
    "case_id",
    "split",
    "center_shift",
    "source_width",
    "closure_coeff",
    "source_amplitude",
    "ambient",
    "sensor_count",
    "noise_std",
)

PREDICTION_FIELDS = (
    "method",
    "case_id",
    "split",
    "true_center_shift",
    "pred_center_shift",
    "center_shift_ci90_low",
    "center_shift_ci90_high",
    "true_source_width",
    "pred_source_width",
    "source_width_ci90_low",
    "source_width_ci90_high",
    "true_closure_coeff",
    "pred_closure_coeff",
    "closure_coeff_ci90_low",
    "closure_coeff_ci90_high",
)

METRIC_FIELDS = (
    "method",
    "method_family",
    "split",
    "case_count",
    "center_shift_rmse",
    "source_width_rmse",
    "closure_coeff_rmse",
    "joint_normalized_rmse",
    "coverage90_mean",
    "calibration_gap",
    "selection_score",
)


@dataclass(frozen=True)
class HiddenSourceCase:
    case_id: str
    split: str
    center_shift: float
    source_width: float
    closure_coeff: float
    source_amplitude: float
    ambient: float
    x: np.ndarray
    t: np.ndarray
    y: np.ndarray
    noise_std: float


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _stable(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 10)
    if isinstance(value, dict):
        return {key: _stable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(_stable(payload), indent=2, sort_keys=True) + "\n")


def _csv_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{round(value, 10):.10g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_stable(value), sort_keys=True)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is None:
        return str(path).replace("\\", "/")
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _kernel(x: np.ndarray, t: np.ndarray, center_shift: float, width: float) -> np.ndarray:
    center = 0.22 + 0.56 * t + 0.030 * np.sin(2.0 * math.pi * t) + center_shift
    variance = width**2 + 0.018 * (t + 0.04)
    return np.exp(-0.5 * (x - center) ** 2 / variance) / np.sqrt(variance)


def _closure_basis(x: np.ndarray, t: np.ndarray, center_shift: float, width: float) -> np.ndarray:
    center = 0.22 + 0.56 * t + center_shift
    local = (x - center) / max(width, 1e-8)
    return np.sin(math.pi * x) * t * (1.0 - 0.35 * t) + 0.20 * local * np.exp(
        -0.5 * local**2
    )


def _design_matrix(
    x: np.ndarray,
    t: np.ndarray,
    center_shift: float,
    width: float,
    *,
    include_closure: bool = True,
) -> np.ndarray:
    columns = [
        np.ones_like(x),
        t,
        x,
        _kernel(x, t, center_shift, width),
    ]
    if include_closure:
        columns.append(_closure_basis(x, t, center_shift, width))
    return np.column_stack(columns)


def _signal(
    x: np.ndarray,
    t: np.ndarray,
    *,
    center_shift: float,
    width: float,
    closure_coeff: float,
    amplitude: float,
    ambient: float,
) -> np.ndarray:
    return (
        ambient
        + 0.07 * t
        + 0.025 * x
        + amplitude * _kernel(x, t, center_shift, width)
        + closure_coeff * _closure_basis(x, t, center_shift, width)
    )


def generate_cases(*, seed: int = 169, noise_std: float = 0.018) -> list[HiddenSourceCase]:
    rng = np.random.default_rng(seed)
    x_values = np.linspace(0.04, 0.96, 12)
    t_values = np.linspace(0.06, 1.0, 9)
    t_grid, x_grid = np.meshgrid(t_values, x_values, indexing="ij")
    x = x_grid.ravel()
    t = t_grid.ravel()
    center_shifts = [-0.045, -0.020, 0.000, 0.026, 0.052]
    widths = [0.036, 0.054, 0.074, 0.096]
    closures = [-0.115, -0.045, 0.035, 0.105]
    cases: list[HiddenSourceCase] = []
    counter = 0
    for i, center_shift in enumerate(center_shifts):
        for j, width in enumerate(widths):
            for k, closure_coeff in enumerate(closures):
                split = "train"
                if (i + 2 * j + k) % 5 == 1:
                    split = "val"
                elif (i + 2 * j + k) % 5 == 3:
                    split = "test"
                amplitude = 0.88 + 0.06 * math.sin(i + 0.5 * j - 0.3 * k)
                ambient = 0.16 + 0.015 * math.cos(0.4 * i - j + 0.2 * k)
                clean = _signal(
                    x,
                    t,
                    center_shift=center_shift,
                    width=width,
                    closure_coeff=closure_coeff,
                    amplitude=amplitude,
                    ambient=ambient,
                )
                y = clean + rng.normal(0.0, noise_std, size=clean.shape)
                cases.append(
                    HiddenSourceCase(
                        case_id=f"P169-CASE-{counter:03d}",
                        split=split,
                        center_shift=center_shift,
                        source_width=width,
                        closure_coeff=closure_coeff,
                        source_amplitude=amplitude,
                        ambient=ambient,
                        x=x,
                        t=t,
                        y=y,
                        noise_std=noise_std,
                    )
                )
                counter += 1
    return cases


def _fit_grid(
    case: HiddenSourceCase,
    center_shift: float,
    width: float,
    *,
    include_closure: bool = True,
) -> tuple[float, np.ndarray]:
    design = _design_matrix(
        case.x,
        case.t,
        center_shift,
        width,
        include_closure=include_closure,
    )
    coef, *_ = np.linalg.lstsq(design, case.y, rcond=None)
    residual = case.y - design @ coef
    return float(np.dot(residual, residual)), coef


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    threshold = quantile * cumulative[-1]
    return float(sorted_values[np.searchsorted(cumulative, threshold, side="left")])


def bayesian_hidden_source_closure_posterior(
    case: HiddenSourceCase,
    *,
    center_grid: np.ndarray,
    width_grid: np.ndarray,
) -> dict[str, Any]:
    rows: list[tuple[float, float, float, float]] = []
    best: tuple[float, float, float, float] | None = None
    for center_shift in center_grid:
        for width in width_grid:
            sse, coef = _fit_grid(
                case,
                float(center_shift),
                float(width),
                include_closure=True,
            )
            closure_coeff = float(coef[-1])
            row = (float(center_shift), float(width), closure_coeff, sse)
            rows.append(row)
            if best is None or sse < best[3]:
                best = row
    assert best is not None
    sse_values = np.asarray([row[3] for row in rows], dtype=float)
    sigma2 = max(case.noise_std**2, float(np.min(sse_values)) / max(1, len(case.y) - 5))
    log_weights = -0.5 * (sse_values - float(np.min(sse_values))) / sigma2
    weights = np.exp(np.clip(log_weights, -700.0, 0.0))
    weights = weights / float(np.sum(weights))
    centers = np.asarray([row[0] for row in rows], dtype=float)
    widths = np.asarray([row[1] for row in rows], dtype=float)
    closures = np.asarray([row[2] for row in rows], dtype=float)
    return {
        "pred_center_shift": float(np.sum(weights * centers)),
        "pred_source_width": float(np.sum(weights * widths)),
        "pred_closure_coeff": float(np.sum(weights * closures)),
        "center_shift_ci90_low": _weighted_quantile(centers, weights, 0.05),
        "center_shift_ci90_high": _weighted_quantile(centers, weights, 0.95),
        "source_width_ci90_low": _weighted_quantile(widths, weights, 0.05),
        "source_width_ci90_high": _weighted_quantile(widths, weights, 0.95),
        "closure_coeff_ci90_low": _weighted_quantile(closures, weights, 0.05),
        "closure_coeff_ci90_high": _weighted_quantile(closures, weights, 0.95),
        "best_grid_center_shift": best[0],
        "best_grid_source_width": best[1],
        "best_grid_closure_coeff": best[2],
        "best_sse": best[3],
    }


def grid_least_squares_control(
    case: HiddenSourceCase,
    *,
    center_grid: np.ndarray,
    width_grid: np.ndarray,
) -> dict[str, Any]:
    posterior = bayesian_hidden_source_closure_posterior(
        case,
        center_grid=center_grid,
        width_grid=width_grid,
    )
    return {
        "pred_center_shift": posterior["best_grid_center_shift"],
        "pred_source_width": posterior["best_grid_source_width"],
        "pred_closure_coeff": posterior["best_grid_closure_coeff"],
        "center_shift_ci90_low": None,
        "center_shift_ci90_high": None,
        "source_width_ci90_low": None,
        "source_width_ci90_high": None,
        "closure_coeff_ci90_low": None,
        "closure_coeff_ci90_high": None,
    }


def no_closure_source_control(
    case: HiddenSourceCase,
    *,
    center_grid: np.ndarray,
    width_grid: np.ndarray,
) -> dict[str, Any]:
    best: tuple[float, float, float] | None = None
    for center_shift in center_grid:
        for width in width_grid:
            sse, _ = _fit_grid(
                case,
                float(center_shift),
                float(width),
                include_closure=False,
            )
            row = (float(center_shift), float(width), sse)
            if best is None or sse < best[2]:
                best = row
    assert best is not None
    return {
        "pred_center_shift": best[0],
        "pred_source_width": best[1],
        "pred_closure_coeff": 0.0,
        "center_shift_ci90_low": None,
        "center_shift_ci90_high": None,
        "source_width_ci90_low": None,
        "source_width_ci90_high": None,
        "closure_coeff_ci90_low": None,
        "closure_coeff_ci90_high": None,
    }


def moment_linearized_control(case: HiddenSourceCase) -> dict[str, Any]:
    centers: list[float] = []
    variances: list[float] = []
    times: list[float] = []
    for time_value in sorted(set(float(value) for value in case.t)):
        mask = np.isclose(case.t, time_value)
        x = case.x[mask]
        y = case.y[mask]
        weights = np.maximum(y - float(np.min(y)) + 1e-6, 1e-6)
        center = float(np.sum(weights * x) / np.sum(weights))
        variance = float(np.sum(weights * (x - center) ** 2) / np.sum(weights))
        centers.append(center)
        variances.append(variance)
        times.append(time_value)
    design = np.column_stack([np.ones(len(times)), np.asarray(times)])
    center_coef, *_ = np.linalg.lstsq(design, np.asarray(centers), rcond=None)
    width = math.sqrt(max(float(np.median(variances)), 1e-8))
    center_shift = float(center_coef[0] - 0.22)
    closure_basis = _closure_basis(case.x, case.t, center_shift, width)
    residual = case.y - _design_matrix(
        case.x,
        case.t,
        center_shift,
        width,
        include_closure=False,
    ) @ np.linalg.lstsq(
        _design_matrix(case.x, case.t, center_shift, width, include_closure=False),
        case.y,
        rcond=None,
    )[0]
    denom = float(np.dot(closure_basis, closure_basis))
    closure_coeff = float(np.dot(residual, closure_basis) / denom) if denom > 0.0 else 0.0
    return {
        "pred_center_shift": float(np.clip(center_shift, -0.08, 0.08)),
        "pred_source_width": float(np.clip(width, 0.025, 0.12)),
        "pred_closure_coeff": float(np.clip(closure_coeff, -0.2, 0.2)),
        "center_shift_ci90_low": None,
        "center_shift_ci90_high": None,
        "source_width_ci90_low": None,
        "source_width_ci90_high": None,
        "closure_coeff_ci90_low": None,
        "closure_coeff_ci90_high": None,
    }


def _sensor_matrix(cases: list[HiddenSourceCase]) -> np.ndarray:
    return np.vstack([case.y for case in cases])


def extra_trees_sensor_control(cases: list[HiddenSourceCase]) -> dict[str, dict[str, Any]]:
    train_cases = [case for case in cases if case.split == "train"]
    x_train = _sensor_matrix(train_cases)
    y_train = np.asarray(
        [[case.center_shift, case.source_width, case.closure_coeff] for case in train_cases]
    )
    predictions: dict[str, dict[str, Any]] = {}
    if ExtraTreesRegressor is None:
        train_mean = np.mean(y_train, axis=0)
        for case in cases:
            predictions[case.case_id] = {
                "pred_center_shift": float(train_mean[0]),
                "pred_source_width": float(train_mean[1]),
                "pred_closure_coeff": float(train_mean[2]),
                "center_shift_ci90_low": None,
                "center_shift_ci90_high": None,
                "source_width_ci90_low": None,
                "source_width_ci90_high": None,
                "closure_coeff_ci90_low": None,
                "closure_coeff_ci90_high": None,
            }
        return predictions
    model = ExtraTreesRegressor(
        n_estimators=64,
        min_samples_leaf=4,
        random_state=169,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(_sensor_matrix(cases))
    for case, pred in zip(cases, y_pred):
        predictions[case.case_id] = {
            "pred_center_shift": float(pred[0]),
            "pred_source_width": float(pred[1]),
            "pred_closure_coeff": float(pred[2]),
            "center_shift_ci90_low": None,
            "center_shift_ci90_high": None,
            "source_width_ci90_low": None,
            "source_width_ci90_high": None,
            "closure_coeff_ci90_low": None,
            "closure_coeff_ci90_high": None,
        }
    return predictions


def calibrated_bayesian_predictions(
    cases: list[HiddenSourceCase],
    raw_predictions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    ratios: list[float] = []
    for case in cases:
        if case.split != "train":
            continue
        pred = raw_predictions[case.case_id]
        for parameter, true_value in (
            ("center_shift", case.center_shift),
            ("source_width", case.source_width),
            ("closure_coeff", case.closure_coeff),
        ):
            mean = float(pred[f"pred_{parameter}"])
            low = float(pred[f"{parameter}_ci90_low"])
            high = float(pred[f"{parameter}_ci90_high"])
            half_width = max((high - low) / 2.0, 1e-12)
            ratios.append(abs(float(true_value) - mean) / half_width)
    scale = max(1.0, float(np.quantile(np.asarray(ratios, dtype=float), 0.9)))
    calibrated: dict[str, dict[str, Any]] = {}
    for case in cases:
        pred = raw_predictions[case.case_id]
        payload = {
            "pred_center_shift": pred["pred_center_shift"],
            "pred_source_width": pred["pred_source_width"],
            "pred_closure_coeff": pred["pred_closure_coeff"],
        }
        for parameter in ("center_shift", "source_width", "closure_coeff"):
            mean = float(pred[f"pred_{parameter}"])
            low = float(pred[f"{parameter}_ci90_low"])
            high = float(pred[f"{parameter}_ci90_high"])
            half_width = max((high - low) / 2.0, 1e-12) * scale
            payload[f"{parameter}_ci90_low"] = mean - half_width
            payload[f"{parameter}_ci90_high"] = mean + half_width
        calibrated[case.case_id] = payload
    return calibrated


def build_predictions(cases: list[HiddenSourceCase]) -> list[dict[str, Any]]:
    center_grid = np.linspace(-0.060, 0.065, 82)
    width_grid = np.linspace(0.030, 0.105, 70)
    extra_tree_predictions = extra_trees_sensor_control(cases)
    raw_bayesian = {
        case.case_id: bayesian_hidden_source_closure_posterior(
            case,
            center_grid=center_grid,
            width_grid=width_grid,
        )
        for case in cases
    }
    calibrated_bayesian = calibrated_bayesian_predictions(cases, raw_bayesian)
    rows: list[dict[str, Any]] = []
    for case in cases:
        method_predictions = {
            "bayesian_hidden_source_closure_posterior": raw_bayesian[case.case_id],
            "calibrated_bayesian_hidden_source_closure_posterior": calibrated_bayesian[
                case.case_id
            ],
            "grid_least_squares_source_closure_control": grid_least_squares_control(
                case,
                center_grid=center_grid,
                width_grid=width_grid,
            ),
            "no_closure_source_control": no_closure_source_control(
                case,
                center_grid=center_grid,
                width_grid=width_grid,
            ),
            "moment_linearized_closure_control": moment_linearized_control(case),
            "extra_trees_sensor_control": extra_tree_predictions[case.case_id],
        }
        for method, pred in method_predictions.items():
            rows.append(
                {
                    "method": method,
                    "case_id": case.case_id,
                    "split": case.split,
                    "true_center_shift": case.center_shift,
                    "pred_center_shift": pred["pred_center_shift"],
                    "center_shift_ci90_low": pred["center_shift_ci90_low"],
                    "center_shift_ci90_high": pred["center_shift_ci90_high"],
                    "true_source_width": case.source_width,
                    "pred_source_width": pred["pred_source_width"],
                    "source_width_ci90_low": pred["source_width_ci90_low"],
                    "source_width_ci90_high": pred["source_width_ci90_high"],
                    "true_closure_coeff": case.closure_coeff,
                    "pred_closure_coeff": pred["pred_closure_coeff"],
                    "closure_coeff_ci90_low": pred["closure_coeff_ci90_low"],
                    "closure_coeff_ci90_high": pred["closure_coeff_ci90_high"],
                }
            )
    return rows


def _rmse(values: list[float]) -> float:
    return float(math.sqrt(float(np.mean(np.square(values))))) if values else 0.0


def _coverage(row: dict[str, Any], parameter: str) -> float | None:
    low = row.get(f"{parameter}_ci90_low")
    high = row.get(f"{parameter}_ci90_high")
    if low in {"", None} or high in {"", None}:
        return None
    return 1.0 if float(low) <= float(row[f"true_{parameter}"]) <= float(high) else 0.0


def build_metric_rows(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scales = {
        "center_shift": 0.052 - (-0.045),
        "source_width": 0.096 - 0.036,
        "closure_coeff": 0.105 - (-0.115),
    }
    methods = sorted({row["method"] for row in prediction_rows})
    rows: list[dict[str, Any]] = []
    for method in methods:
        method_family = (
            "bayesian_candidate"
            if method
            in {
                "bayesian_hidden_source_closure_posterior",
                "calibrated_bayesian_hidden_source_closure_posterior",
            }
            else "control"
        )
        for split in ("train", "val", "test"):
            subset = [
                row
                for row in prediction_rows
                if row["method"] == method and row["split"] == split
            ]
            center_errors = [
                float(row["pred_center_shift"]) - float(row["true_center_shift"])
                for row in subset
            ]
            width_errors = [
                float(row["pred_source_width"]) - float(row["true_source_width"])
                for row in subset
            ]
            closure_errors = [
                float(row["pred_closure_coeff"]) - float(row["true_closure_coeff"])
                for row in subset
            ]
            normalized = [
                (
                    ((float(row["pred_center_shift"]) - float(row["true_center_shift"])) / scales["center_shift"])
                    ** 2
                    + ((float(row["pred_source_width"]) - float(row["true_source_width"])) / scales["source_width"])
                    ** 2
                    + ((float(row["pred_closure_coeff"]) - float(row["true_closure_coeff"])) / scales["closure_coeff"])
                    ** 2
                )
                / 3.0
                for row in subset
            ]
            coverages = [
                coverage
                for row in subset
                for coverage in (
                    _coverage(row, "center_shift"),
                    _coverage(row, "source_width"),
                    _coverage(row, "closure_coeff"),
                )
                if coverage is not None
            ]
            coverage_mean = float(np.mean(coverages)) if coverages else 0.0
            calibration_gap = abs(coverage_mean - 0.9) if coverages else 0.9
            joint_rmse = float(math.sqrt(float(np.mean(normalized)))) if normalized else 0.0
            rows.append(
                {
                    "method": method,
                    "method_family": method_family,
                    "split": split,
                    "case_count": len(subset),
                    "center_shift_rmse": _rmse(center_errors),
                    "source_width_rmse": _rmse(width_errors),
                    "closure_coeff_rmse": _rmse(closure_errors),
                    "joint_normalized_rmse": joint_rmse,
                    "coverage90_mean": coverage_mean,
                    "calibration_gap": calibration_gap,
                    "selection_score": joint_rmse + 0.08 * calibration_gap,
                }
            )
    return rows


def build_case_rows(cases: list[HiddenSourceCase]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case.case_id,
            "split": case.split,
            "center_shift": case.center_shift,
            "source_width": case.source_width,
            "closure_coeff": case.closure_coeff,
            "source_amplitude": case.source_amplitude,
            "ambient": case.ambient,
            "sensor_count": len(case.y),
            "noise_std": case.noise_std,
        }
        for case in cases
    ]


def _metric_lookup(rows: list[dict[str, Any]], method: str, split: str) -> dict[str, Any]:
    for row in rows:
        if row["method"] == method and row["split"] == split:
            return row
    raise KeyError((method, split))


def build_gate(
    *,
    phase168_gate: dict[str, Any],
    phase168_design_rows: list[dict[str, str]],
    metric_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase168_ready = (
        phase168_gate.get("status")
        == "phase168_hidden_source_closure_redesign_ready_phase169_identifiability_gate"
        and _is_true(phase168_gate.get("phase169_hidden_source_closure_identifiability_gate_allowed"))
    )
    design_ready = len(phase168_design_rows) >= 5
    val_rows = [row for row in metric_rows if row["split"] == "val"]
    selected = min(val_rows, key=lambda row: float(row["selection_score"]))
    controls = [row for row in val_rows if row["method_family"] == "control"]
    best_control = min(controls, key=lambda row: float(row["selection_score"]))
    if selected["method_family"] == "bayesian_candidate":
        candidate_val = selected
    else:
        bayesian_rows = [
            row for row in val_rows if row["method_family"] == "bayesian_candidate"
        ]
        candidate_val = min(bayesian_rows, key=lambda row: float(row["selection_score"]))
    candidate_test = _metric_lookup(metric_rows, candidate_val["method"], "test")
    best_control_test = _metric_lookup(metric_rows, best_control["method"], "test")
    validation_gain = float(best_control["selection_score"]) - float(
        candidate_val["selection_score"]
    )
    test_reversal_ratio = float(candidate_test["joint_normalized_rmse"]) / max(
        float(best_control_test["joint_normalized_rmse"]),
        1e-12,
    )
    closure_identifiable = (
        float(candidate_val["closure_coeff_rmse"]) <= 0.020
        and float(candidate_test["closure_coeff_rmse"]) <= 0.025
    )
    coverage_ok = (
        0.65 <= float(candidate_val["coverage90_mean"]) <= 1.0
        and 0.65 <= float(candidate_test["coverage90_mean"]) <= 1.0
    )
    pass_gate = (
        phase168_ready
        and design_ready
        and selected["method_family"] == "bayesian_candidate"
        and candidate_val["method"]
        == "calibrated_bayesian_hidden_source_closure_posterior"
        and validation_gain >= 0.008
        and test_reversal_ratio <= 1.05
        and closure_identifiable
        and coverage_ok
    )
    blockers: list[str] = []
    if not phase168_ready:
        blockers.append("phase168_gate_not_ready")
    if not design_ready:
        blockers.append("phase168_design_contract_missing")
    if selected["method_family"] != "bayesian_candidate":
        blockers.append("validation_selected_non_bayesian_control")
    if candidate_val["method"] != "calibrated_bayesian_hidden_source_closure_posterior":
        blockers.append("calibrated_candidate_not_selected")
    if validation_gain < 0.008:
        blockers.append("validation_gain_vs_best_control")
    if test_reversal_ratio > 1.05:
        blockers.append("test_reversal_vs_best_control")
    if not closure_identifiable:
        blockers.append("closure_coeff_identifiability_guard")
    if not coverage_ok:
        blockers.append("posterior_calibration_guard")
    return {
        "status": (
            "phase169_hidden_source_closure_identifiability_ready_phase170_low_budget_mechanism_design"
            if pass_gate
            else "phase169_hidden_source_closure_identifiability_closed_no_guarded_identifiability"
        ),
        "selected_method": selected["method"],
        "candidate_method": candidate_val["method"],
        "best_control_method": best_control["method"],
        "candidate_validation_score": candidate_val["selection_score"],
        "best_control_validation_score": best_control["selection_score"],
        "validation_score_gain_vs_best_control": validation_gain,
        "candidate_test_joint_normalized_rmse": candidate_test["joint_normalized_rmse"],
        "best_control_test_joint_normalized_rmse": best_control_test["joint_normalized_rmse"],
        "test_reversal_ratio_vs_best_control": test_reversal_ratio,
        "candidate_validation_coverage90_mean": candidate_val["coverage90_mean"],
        "candidate_test_coverage90_mean": candidate_test["coverage90_mean"],
        "candidate_validation_closure_coeff_rmse": candidate_val["closure_coeff_rmse"],
        "candidate_test_closure_coeff_rmse": candidate_test["closure_coeff_rmse"],
        "blocking_audits": blockers,
        "phase170_low_budget_mechanism_design_allowed": bool(pass_gate),
        "phase169_model_mechanism_allowed": False,
        "phase169_model_training_allowed": False,
        "phase170_training_allowed_now": False,
        "bayesian_pinn_training_allowed_now": False,
        "adaptive_sampling_training_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": (
            "enter Phase 170 low-budget mechanism smoke design, still before training"
            if pass_gate
            else "close or redesign hidden-source/closure route before any training"
        ),
    }


def _markdown_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[str]:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(_csv_value(row.get(field, "")) for field in fields)
        + " |"
        for row in rows
    ]
    return [header, sep, *body]


def build_markdown(*, gate: dict[str, Any], metric_rows: list[dict[str, Any]]) -> str:
    val_test_rows = [row for row in metric_rows if row["split"] in {"val", "test"}]
    lines = [
        "# Phase 169 Hidden-Source/Closure Identifiability Gate",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Candidate method: `{gate['candidate_method']}`",
        f"- Best control method: `{gate['best_control_method']}`",
        f"- Validation score gain vs best control: `{_csv_value(gate['validation_score_gain_vs_best_control'])}`",
        f"- Test reversal ratio vs best control: `{_csv_value(gate['test_reversal_ratio_vs_best_control'])}`",
        f"- Phase 170 low-budget mechanism design allowed: `{_csv_value(gate['phase170_low_budget_mechanism_design_allowed'])}`",
        f"- Phase 169 model training allowed: `{_csv_value(gate['phase169_model_training_allowed'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "This gate tests source/closure identifiability only. A positive result "
            "does not train a PINN and does not support AM-Bench or Bayesian neural "
            "claims; it only allows a later low-budget mechanism smoke design."
        ),
        "",
        "## Validation and Test Metrics",
        *_markdown_table(val_test_rows, METRIC_FIELDS),
        "",
    ]
    return "\n".join(lines)


def build_package(*, root: Path, output_dir: Path, phase_inputs: dict[str, Path]) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    resolved = {
        name: path if path.is_absolute() else root / path
        for name, path in phase_inputs.items()
    }
    phase168_gate = _read_json(resolved["phase168_gate"])
    phase168_design_rows = _read_csv(resolved["phase168_design_table"])
    cases = generate_cases()
    case_rows = build_case_rows(cases)
    prediction_rows = build_predictions(cases)
    metric_rows = build_metric_rows(prediction_rows)
    gate = build_gate(
        phase168_gate=phase168_gate,
        phase168_design_rows=phase168_design_rows,
        metric_rows=metric_rows,
    )

    case_path = output_dir / "phase169_hidden_source_case_manifest.csv"
    prediction_path = output_dir / "phase169_hidden_source_prediction_table.csv"
    metric_path = output_dir / "phase169_hidden_source_metric_table.csv"
    gate_path = output_dir / "phase169_hidden_source_closure_identifiability_gate.json"
    markdown_path = output_dir / "phase169_hidden_source_closure_identifiability_gate.md"
    manifest_path = output_dir / "phase169_hidden_source_closure_identifiability_manifest.json"

    _write_csv(case_path, case_rows, CASE_FIELDS)
    _write_csv(prediction_path, prediction_rows, PREDICTION_FIELDS)
    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_json(gate_path, gate)
    with markdown_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(build_markdown(gate=gate, metric_rows=metric_rows))

    manifest = {
        "phase": 169,
        "description": "hidden-source/closure no-training identifiability gate",
        "inputs": {name: _display_path(path, root) for name, path in resolved.items()},
        "outputs": {
            "case_manifest": _display_path(case_path, root),
            "prediction_table": _display_path(prediction_path, root),
            "metric_table": _display_path(metric_path, root),
            "gate": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "case_rows": len(case_rows),
            "prediction_rows": len(prediction_rows),
            "metric_rows": len(metric_rows),
            "phase168_design_rows": len(phase168_design_rows),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    for name, default in PHASE_INPUTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase_inputs = {name: getattr(args, name) for name in PHASE_INPUTS}
    manifest = build_package(root=args.root, output_dir=args.output_dir, phase_inputs=phase_inputs)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
