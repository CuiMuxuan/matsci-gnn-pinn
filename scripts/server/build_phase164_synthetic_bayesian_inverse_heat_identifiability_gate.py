#!/usr/bin/env python3
"""Build Phase 164 synthetic Bayesian inverse-heat identifiability gate.

The phase is intentionally no-training. It creates sparse/noisy synthetic heat
sensor cases with hidden diffusivity and source-width parameters, compares a
grid Bayesian posterior against non-neural inverse controls, and decides whether
the Bayesian route is identifiable enough to continue to a later no-training
adaptive-sampler gate.
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
except Exception:  # pragma: no cover - covered only on minimal environments.
    ExtraTreesRegressor = None  # type: ignore[assignment]


DEFAULT_OUTPUT_DIR = Path(
    "docs/results/phase164_synthetic_bayesian_inverse_heat_identifiability_gate"
)

PHASE_INPUTS = {
    "phase163_gate": Path(
        "docs/results/phase163_pinn_bayesian_hybrid_roadmap/"
        "phase163_pinn_bayesian_hybrid_roadmap_gate.json"
    ),
    "phase163_route_table": Path(
        "docs/results/phase163_pinn_bayesian_hybrid_roadmap/"
        "phase163_route_candidate_table.csv"
    ),
}

CASE_FIELDS = (
    "case_id",
    "split",
    "diffusivity",
    "source_width",
    "source_amplitude",
    "ambient",
    "sensor_count",
    "noise_std",
)

METRIC_FIELDS = (
    "method",
    "method_family",
    "split",
    "case_count",
    "diffusivity_rmse",
    "source_width_rmse",
    "joint_normalized_rmse",
    "coverage90_mean",
    "calibration_gap",
    "selection_score",
)

PREDICTION_FIELDS = (
    "method",
    "case_id",
    "split",
    "true_diffusivity",
    "pred_diffusivity",
    "diffusivity_ci90_low",
    "diffusivity_ci90_high",
    "true_source_width",
    "pred_source_width",
    "source_width_ci90_low",
    "source_width_ci90_high",
)


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    split: str
    diffusivity: float
    source_width: float
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(_stable_json_value(payload), indent=2, sort_keys=True) + "\n")


def _stable_json_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 10)
    if isinstance(value, dict):
        return {key: _stable_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stable_json_value(item) for item in value]
    return value


def _csv_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{round(value, 10):.10g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
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


def _is_false(value: Any) -> bool:
    if value is False or value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "0", "false", "none", "no"}
    return False


def _kernel(x: np.ndarray, t: np.ndarray, diffusivity: float, width: float) -> np.ndarray:
    center = 0.24 + 0.50 * t
    variance = width**2 + 2.0 * diffusivity * (t + 0.03)
    return np.exp(-0.5 * (x - center) ** 2 / variance) / np.sqrt(variance)


def _design_matrix(x: np.ndarray, t: np.ndarray, diffusivity: float, width: float) -> np.ndarray:
    return np.column_stack([np.ones_like(x), t, x, _kernel(x, t, diffusivity, width)])


def _signal(
    x: np.ndarray,
    t: np.ndarray,
    *,
    diffusivity: float,
    width: float,
    amplitude: float,
    ambient: float,
) -> np.ndarray:
    return ambient + 0.08 * t + 0.03 * x + amplitude * _kernel(x, t, diffusivity, width)


def generate_cases(*, seed: int = 164, noise_std: float = 0.025) -> list[SyntheticCase]:
    rng = np.random.default_rng(seed)
    x_values = np.linspace(0.05, 0.95, 13)
    t_values = np.linspace(0.08, 1.0, 9)
    t_grid, x_grid = np.meshgrid(t_values, x_values, indexing="ij")
    x = x_grid.ravel()
    t = t_grid.ravel()
    diffusivities = [0.018, 0.028, 0.040, 0.056, 0.075, 0.095]
    widths = [0.035, 0.052, 0.070, 0.090]
    cases: list[SyntheticCase] = []
    counter = 0
    for i, diffusivity in enumerate(diffusivities):
        for j, width in enumerate(widths):
            split = "train"
            if (i + 2 * j) % 5 == 1:
                split = "val"
            elif (i + 2 * j) % 5 == 3:
                split = "test"
            amplitude = 1.00 + 0.08 * math.sin(i + 0.7 * j)
            ambient = 0.20 + 0.02 * math.cos(0.5 * i - j)
            clean = _signal(
                x,
                t,
                diffusivity=diffusivity,
                width=width,
                amplitude=amplitude,
                ambient=ambient,
            )
            y = clean + rng.normal(0.0, noise_std, size=clean.shape)
            cases.append(
                SyntheticCase(
                    case_id=f"P164-CASE-{counter:03d}",
                    split=split,
                    diffusivity=diffusivity,
                    source_width=width,
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


def _fit_for_grid(case: SyntheticCase, diffusivity: float, width: float) -> tuple[float, np.ndarray]:
    design = _design_matrix(case.x, case.t, diffusivity, width)
    coef, *_ = np.linalg.lstsq(design, case.y, rcond=None)
    residual = case.y - design @ coef
    sse = float(np.dot(residual, residual))
    return sse, coef


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    threshold = quantile * cumulative[-1]
    return float(sorted_values[np.searchsorted(cumulative, threshold, side="left")])


def bayesian_grid_posterior(
    case: SyntheticCase,
    *,
    diffusivity_grid: np.ndarray,
    width_grid: np.ndarray,
) -> dict[str, Any]:
    rows: list[tuple[float, float, float]] = []
    best: tuple[float, float, float] | None = None
    for diffusivity in diffusivity_grid:
        for width in width_grid:
            sse, _ = _fit_for_grid(case, float(diffusivity), float(width))
            rows.append((float(diffusivity), float(width), sse))
            if best is None or sse < best[2]:
                best = (float(diffusivity), float(width), sse)
    assert best is not None
    sse_values = np.asarray([row[2] for row in rows], dtype=float)
    sigma2 = max(case.noise_std**2, float(np.min(sse_values)) / max(1, len(case.y) - 4))
    log_weights = -0.5 * (sse_values - float(np.min(sse_values))) / sigma2
    weights = np.exp(np.clip(log_weights, -700.0, 0.0))
    weights = weights / float(np.sum(weights))
    diffusivities = np.asarray([row[0] for row in rows], dtype=float)
    widths = np.asarray([row[1] for row in rows], dtype=float)
    return {
        "pred_diffusivity": float(np.sum(weights * diffusivities)),
        "pred_source_width": float(np.sum(weights * widths)),
        "diffusivity_ci90_low": _weighted_quantile(diffusivities, weights, 0.05),
        "diffusivity_ci90_high": _weighted_quantile(diffusivities, weights, 0.95),
        "source_width_ci90_low": _weighted_quantile(widths, weights, 0.05),
        "source_width_ci90_high": _weighted_quantile(widths, weights, 0.95),
        "best_grid_diffusivity": best[0],
        "best_grid_source_width": best[1],
        "best_sse": best[2],
    }


def grid_least_squares_control(
    case: SyntheticCase,
    *,
    diffusivity_grid: np.ndarray,
    width_grid: np.ndarray,
) -> dict[str, Any]:
    posterior = bayesian_grid_posterior(
        case,
        diffusivity_grid=diffusivity_grid,
        width_grid=width_grid,
    )
    return {
        "pred_diffusivity": posterior["best_grid_diffusivity"],
        "pred_source_width": posterior["best_grid_source_width"],
        "diffusivity_ci90_low": None,
        "diffusivity_ci90_high": None,
        "source_width_ci90_low": None,
        "source_width_ci90_high": None,
    }


def moment_linearized_control(case: SyntheticCase) -> dict[str, Any]:
    variances = []
    times = []
    for time_value in sorted(set(float(value) for value in case.t)):
        mask = np.isclose(case.t, time_value)
        x = case.x[mask]
        y = case.y[mask]
        weights = np.maximum(y - float(np.min(y)) + 1e-6, 1e-6)
        mean_x = float(np.sum(weights * x) / np.sum(weights))
        variance = float(np.sum(weights * (x - mean_x) ** 2) / np.sum(weights))
        variances.append(variance)
        times.append(time_value)
    design = np.column_stack([np.ones(len(times)), np.asarray(times)])
    coef, *_ = np.linalg.lstsq(design, np.asarray(variances), rcond=None)
    slope = max(float(coef[1]), 1e-8)
    diffusivity = slope / 2.0
    width2 = max(float(coef[0]) - 2.0 * diffusivity * 0.03, 1e-8)
    return {
        "pred_diffusivity": float(np.clip(diffusivity, 0.001, 0.2)),
        "pred_source_width": float(np.clip(math.sqrt(width2), 0.005, 0.2)),
        "diffusivity_ci90_low": None,
        "diffusivity_ci90_high": None,
        "source_width_ci90_low": None,
        "source_width_ci90_high": None,
    }


def _sensor_matrix(cases: list[SyntheticCase]) -> np.ndarray:
    return np.vstack([case.y for case in cases])


def extra_trees_sensor_control(cases: list[SyntheticCase]) -> dict[str, dict[str, Any]]:
    train_cases = [case for case in cases if case.split == "train"]
    x_train = _sensor_matrix(train_cases)
    y_train = np.asarray([[case.diffusivity, case.source_width] for case in train_cases])
    predictions: dict[str, dict[str, Any]] = {}
    if ExtraTreesRegressor is None:
        train_mean = np.mean(y_train, axis=0)
        for case in cases:
            predictions[case.case_id] = {
                "pred_diffusivity": float(train_mean[0]),
                "pred_source_width": float(train_mean[1]),
                "diffusivity_ci90_low": None,
                "diffusivity_ci90_high": None,
                "source_width_ci90_low": None,
                "source_width_ci90_high": None,
            }
        return predictions
    model = ExtraTreesRegressor(
        n_estimators=96,
        min_samples_leaf=2,
        random_state=164,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(_sensor_matrix(cases))
    for case, pred in zip(cases, y_pred):
        predictions[case.case_id] = {
            "pred_diffusivity": float(pred[0]),
            "pred_source_width": float(pred[1]),
            "diffusivity_ci90_low": None,
            "diffusivity_ci90_high": None,
            "source_width_ci90_low": None,
            "source_width_ci90_high": None,
        }
    return predictions


def calibrated_bayesian_predictions(
    cases: list[SyntheticCase],
    raw_predictions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    train_cases = [case for case in cases if case.split == "train"]
    ratios: list[float] = []
    for case in train_cases:
        pred = raw_predictions[case.case_id]
        for parameter, true_value in (
            ("diffusivity", case.diffusivity),
            ("source_width", case.source_width),
        ):
            mean = float(pred[f"pred_{parameter}"])
            low = float(pred[f"{parameter}_ci90_low"])
            high = float(pred[f"{parameter}_ci90_high"])
            half_width = max((high - low) / 2.0, 1e-12)
            ratios.append(abs(float(true_value) - mean) / half_width)
    # Train-split conformal-style scaling for nominal 90% parameter intervals.
    scale = max(1.0, float(np.quantile(np.asarray(ratios, dtype=float), 0.9)))
    calibrated: dict[str, dict[str, Any]] = {}
    for case in cases:
        pred = raw_predictions[case.case_id]
        payload = {
            "pred_diffusivity": pred["pred_diffusivity"],
            "pred_source_width": pred["pred_source_width"],
        }
        for parameter in ("diffusivity", "source_width"):
            mean = float(pred[f"pred_{parameter}"])
            low = float(pred[f"{parameter}_ci90_low"])
            high = float(pred[f"{parameter}_ci90_high"])
            half_width = max((high - low) / 2.0, 1e-12) * scale
            payload[f"{parameter}_ci90_low"] = mean - half_width
            payload[f"{parameter}_ci90_high"] = mean + half_width
        calibrated[case.case_id] = payload
    return calibrated


def build_predictions(cases: list[SyntheticCase]) -> list[dict[str, Any]]:
    diffusivity_grid = np.linspace(0.012, 0.110, 70)
    width_grid = np.linspace(0.025, 0.105, 64)
    extra_tree_predictions = extra_trees_sensor_control(cases)
    raw_bayesian = {
        case.case_id: bayesian_grid_posterior(
            case,
            diffusivity_grid=diffusivity_grid,
            width_grid=width_grid,
        )
        for case in cases
    }
    calibrated_bayesian = calibrated_bayesian_predictions(cases, raw_bayesian)
    rows: list[dict[str, Any]] = []
    for case in cases:
        method_predictions = {
            "bayesian_grid_posterior": raw_bayesian[case.case_id],
            "calibrated_bayesian_grid_posterior": calibrated_bayesian[case.case_id],
            "grid_least_squares_control": grid_least_squares_control(
                case,
                diffusivity_grid=diffusivity_grid,
                width_grid=width_grid,
            ),
            "moment_linearized_control": moment_linearized_control(case),
            "extra_trees_sensor_control": extra_tree_predictions[case.case_id],
        }
        for method, pred in method_predictions.items():
            rows.append(
                {
                    "method": method,
                    "case_id": case.case_id,
                    "split": case.split,
                    "true_diffusivity": case.diffusivity,
                    "pred_diffusivity": pred["pred_diffusivity"],
                    "diffusivity_ci90_low": pred["diffusivity_ci90_low"],
                    "diffusivity_ci90_high": pred["diffusivity_ci90_high"],
                    "true_source_width": case.source_width,
                    "pred_source_width": pred["pred_source_width"],
                    "source_width_ci90_low": pred["source_width_ci90_low"],
                    "source_width_ci90_high": pred["source_width_ci90_high"],
                }
            )
    return rows


def _rmse(values: list[float]) -> float:
    return float(math.sqrt(float(np.mean(np.square(values))))) if values else 0.0


def _coverage(row: dict[str, Any], parameter: str) -> float | None:
    true_key = f"true_{parameter}"
    low_key = f"{parameter}_ci90_low"
    high_key = f"{parameter}_ci90_high"
    low = row.get(low_key)
    high = row.get(high_key)
    if low in {"", None} or high in {"", None}:
        return None
    return 1.0 if float(low) <= float(row[true_key]) <= float(high) else 0.0


def build_metric_rows(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alpha_scale = 0.095 - 0.018
    width_scale = 0.090 - 0.035
    methods = sorted({row["method"] for row in prediction_rows})
    rows: list[dict[str, Any]] = []
    for method in methods:
        method_family = (
            "bayesian_candidate"
            if method in {"bayesian_grid_posterior", "calibrated_bayesian_grid_posterior"}
            else "control"
        )
        for split in ("train", "val", "test"):
            subset = [
                row
                for row in prediction_rows
                if row["method"] == method and row["split"] == split
            ]
            alpha_errors = [
                float(row["pred_diffusivity"]) - float(row["true_diffusivity"])
                for row in subset
            ]
            width_errors = [
                float(row["pred_source_width"]) - float(row["true_source_width"])
                for row in subset
            ]
            normalized = [
                0.5
                * (
                    ((float(row["pred_diffusivity"]) - float(row["true_diffusivity"])) / alpha_scale)
                    ** 2
                    + ((float(row["pred_source_width"]) - float(row["true_source_width"])) / width_scale)
                    ** 2
                )
                for row in subset
            ]
            coverages = [
                coverage
                for row in subset
                for coverage in (
                    _coverage(row, "diffusivity"),
                    _coverage(row, "source_width"),
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
                    "diffusivity_rmse": _rmse(alpha_errors),
                    "source_width_rmse": _rmse(width_errors),
                    "joint_normalized_rmse": joint_rmse,
                    "coverage90_mean": coverage_mean,
                    "calibration_gap": calibration_gap,
                    "selection_score": joint_rmse + 0.10 * calibration_gap,
                }
            )
    return rows


def build_case_rows(cases: list[SyntheticCase]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case.case_id,
            "split": case.split,
            "diffusivity": case.diffusivity,
            "source_width": case.source_width,
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
    phase163_gate: dict[str, Any],
    route_rows: list[dict[str, str]],
    metric_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase163_ready = (
        phase163_gate.get("status")
        == "phase163_pinn_bayesian_hybrid_roadmap_ready_phase164_synthetic_inverse_gate"
        and _is_true(phase163_gate.get("phase164_no_training_design_allowed"))
    )
    matching_routes = [
        row
        for row in route_rows
        if row.get("route_name") == "bayesian_inverse_heat_parameter_synthetic_gate"
    ]
    val_rows = [row for row in metric_rows if row["split"] == "val"]
    selected = min(val_rows, key=lambda row: float(row["selection_score"]))
    controls = [row for row in val_rows if row["method_family"] == "control"]
    best_control = min(controls, key=lambda row: float(row["selection_score"]))
    if selected["method_family"] == "bayesian_candidate":
        candidate_val = selected
    else:
        bayesian_rows = [row for row in val_rows if row["method_family"] == "bayesian_candidate"]
        candidate_val = min(bayesian_rows, key=lambda row: float(row["selection_score"]))
    candidate_test = _metric_lookup(metric_rows, candidate_val["method"], "test")
    best_control_test = _metric_lookup(metric_rows, best_control["method"], "test")
    validation_gain = float(best_control["selection_score"]) - float(candidate_val["selection_score"])
    test_reversal_ratio = (
        float(candidate_test["joint_normalized_rmse"])
        / max(float(best_control_test["joint_normalized_rmse"]), 1e-12)
    )
    coverage_ok = (
        0.65 <= float(candidate_val["coverage90_mean"]) <= 1.0
        and 0.65 <= float(candidate_test["coverage90_mean"]) <= 1.0
    )
    pass_gate = (
        phase163_ready
        and bool(matching_routes)
        and selected["method_family"] == "bayesian_candidate"
        and validation_gain >= 0.005
        and test_reversal_ratio <= 1.05
        and coverage_ok
    )
    blockers: list[str] = []
    if not phase163_ready:
        blockers.append("phase163_gate_not_ready")
    if not matching_routes:
        blockers.append("phase163_recommended_route_missing")
    if selected["method_family"] != "bayesian_candidate":
        blockers.append("validation_selected_non_bayesian_control")
    if validation_gain < 0.005:
        blockers.append("validation_gain_vs_best_control")
    if test_reversal_ratio > 1.05:
        blockers.append("test_reversal_vs_best_control")
    if not coverage_ok:
        blockers.append("posterior_calibration_guard")
    return {
        "status": (
            "phase164_synthetic_bayesian_inverse_heat_identifiability_ready_phase165_sampler_gate"
            if pass_gate
            else "phase164_synthetic_bayesian_inverse_heat_identifiability_closed_no_guarded_gain"
        ),
        "selected_method": selected["method"],
        "best_control_method": best_control["method"],
        "candidate_validation_score": candidate_val["selection_score"],
        "best_control_validation_score": best_control["selection_score"],
        "validation_score_gain_vs_best_control": validation_gain,
        "candidate_test_joint_normalized_rmse": candidate_test["joint_normalized_rmse"],
        "best_control_test_joint_normalized_rmse": best_control_test["joint_normalized_rmse"],
        "test_reversal_ratio_vs_best_control": test_reversal_ratio,
        "candidate_validation_coverage90_mean": candidate_val["coverage90_mean"],
        "candidate_test_coverage90_mean": candidate_test["coverage90_mean"],
        "blocking_audits": blockers,
        "phase165_adaptive_sampler_gate_allowed": bool(pass_gate),
        "phase164_low_capacity_training_allowed": False,
        "phase164_model_mechanism_allowed": False,
        "phase164_model_training_allowed": False,
        "bayesian_pinn_training_allowed_now": False,
        "adaptive_sampling_training_allowed_now": False,
        "gcn_pinn_training_allowed_now": False,
        "cnn_pinn_training_allowed_now": False,
        "operator_training_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": (
            "enter Phase 165 as a no-training adaptive residual sampler gate"
            if pass_gate
            else "close or redesign the synthetic Bayesian inverse route before any training"
        ),
    }


def _markdown_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[str]:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(_csv_value(row.get(field, "")) for field in fields) + " |"
        for row in rows
    ]
    return [header, sep, *body]


def build_markdown(
    *,
    gate: dict[str, Any],
    metric_rows: list[dict[str, Any]],
) -> str:
    val_test_rows = [
        row for row in metric_rows if row["split"] in {"val", "test"}
    ]
    lines = [
        "# Phase 164 Synthetic Bayesian Inverse-Heat Identifiability Gate",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Selected method: `{gate['selected_method']}`",
        f"- Best control method: `{gate['best_control_method']}`",
        f"- Validation score gain vs best control: `{_csv_value(gate['validation_score_gain_vs_best_control'])}`",
        f"- Test reversal ratio vs best control: `{_csv_value(gate['test_reversal_ratio_vs_best_control'])}`",
        f"- Phase 165 adaptive sampler gate allowed: `{_csv_value(gate['phase165_adaptive_sampler_gate_allowed'])}`",
        f"- Phase 164 model training allowed: `{_csv_value(gate['phase164_model_training_allowed'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "The synthetic inverse task is intentionally physics-matched and local. A "
            "positive gate means hidden diffusivity/source-width parameters are "
            "identifiable against strong non-neural controls on this controlled task. "
            "It does not permit AM-Bench Bayesian PINN training."
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
        name: path if path.is_absolute() else root / path for name, path in phase_inputs.items()
    }
    phase163_gate = _read_json(resolved["phase163_gate"])
    route_rows = _read_csv(resolved["phase163_route_table"])

    cases = generate_cases()
    case_rows = build_case_rows(cases)
    prediction_rows = build_predictions(cases)
    metric_rows = build_metric_rows(prediction_rows)
    gate = build_gate(
        phase163_gate=phase163_gate,
        route_rows=route_rows,
        metric_rows=metric_rows,
    )

    case_path = output_dir / "phase164_synthetic_case_manifest.csv"
    prediction_path = output_dir / "phase164_inverse_prediction_table.csv"
    metric_path = output_dir / "phase164_inverse_metric_table.csv"
    gate_path = output_dir / "phase164_synthetic_bayesian_inverse_heat_identifiability_gate.json"
    markdown_path = output_dir / "phase164_synthetic_bayesian_inverse_heat_identifiability_gate.md"
    manifest_path = output_dir / "phase164_synthetic_bayesian_inverse_heat_identifiability_manifest.json"

    _write_csv(case_path, case_rows, CASE_FIELDS)
    _write_csv(prediction_path, prediction_rows, PREDICTION_FIELDS)
    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_json(gate_path, gate)
    with markdown_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(build_markdown(gate=gate, metric_rows=metric_rows))

    manifest = {
        "phase": 164,
        "description": "synthetic Bayesian inverse heat hidden-parameter identifiability gate",
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
            "route_rows": len(route_rows),
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
