#!/usr/bin/env python3
"""Build Phase 173 bounded trainable hidden-closure low-budget smoke.

Phase 173 executes the tiny synthetic smoke opened by the Phase 172 design
gate. It uses NumPy coordinate-search latents for source center/width plus a
train-split calibrated closure head. It does not read AM-Bench/NIST raw data,
does not train a neural PINN, and does not justify A100-SXM4-80GB.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np


DEFAULT_OUTPUT_DIR = Path(
    "docs/results/phase173_trainable_hidden_closure_low_budget_smoke"
)

PHASE_INPUTS = {
    "phase172_gate": Path(
        "docs/results/phase172_trainable_hidden_closure_smoke_design_gate/"
        "phase172_trainable_hidden_closure_smoke_design_gate.json"
    ),
    "phase172_control_table": Path(
        "docs/results/phase172_trainable_hidden_closure_smoke_design_gate/"
        "phase172_control_table.csv"
    ),
    "phase171_gate": Path(
        "docs/results/phase171_hidden_closure_low_budget_smoke/"
        "phase171_hidden_closure_low_budget_smoke_gate.json"
    ),
}

VARIANT_FIELDS = (
    "variant_id",
    "family",
    "source_estimator",
    "closure_estimator",
    "executed",
    "is_control",
    "trainable",
    "description",
)

CALIBRATION_FIELDS = (
    "seed",
    "variant_id",
    "coefficient_intercept",
    "coefficient_optimized_closure",
    "coefficient_posterior_closure",
    "coefficient_grid_closure",
    "coefficient_source_width",
    "coefficient_center_shift",
    "train_case_count",
)

TRAINING_AUDIT_FIELDS = (
    "seed",
    "case_id",
    "split",
    "variant_id",
    "start_count",
    "max_rounds_per_start",
    "executed_rounds_total",
    "function_evaluations_total",
    "train_objective",
    "learned_center_shift",
    "learned_source_width",
    "raw_closure_coeff",
    "calibrated_closure_coeff",
)

CASE_METRIC_FIELDS = (
    "seed",
    "case_id",
    "split",
    "variant_id",
    "family",
    "field_rmse",
    "hot_q90_rmse",
    "gradient_q90_rmse",
    "closure_abs_error",
    "coverage90_mean",
    "selection_score",
)

SUMMARY_FIELDS = (
    "variant_id",
    "family",
    "split",
    "seed_count",
    "case_count",
    "field_rmse_mean",
    "hot_q90_rmse_mean",
    "gradient_q90_rmse_mean",
    "closure_abs_error_mean",
    "coverage90_mean",
    "selection_score_mean",
    "selection_score_std",
)

SEED_SUMMARY_FIELDS = (
    "seed",
    "variant_id",
    "split",
    "case_count",
    "field_rmse_mean",
    "closure_abs_error_mean",
    "selection_score_mean",
)


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


def _load_phase169_module() -> Any:
    script = Path(__file__).with_name(
        "build_phase169_hidden_source_closure_identifiability_gate.py"
    )
    spec = importlib.util.spec_from_file_location("phase169_identifiability_source", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Phase 169 module from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_phase171_module() -> Any:
    script = Path(__file__).with_name("build_phase171_hidden_closure_low_budget_smoke.py")
    spec = importlib.util.spec_from_file_location("phase171_hidden_closure_smoke", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Phase 171 module from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_variant_rows() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "tiny_explicit_latent_hidden_closure_smoke",
            "family": "mechanism_candidate",
            "source_estimator": "continuous_center_width_coordinate_search",
            "closure_estimator": "train_split_calibrated_optimized_closure_head",
            "executed": True,
            "is_control": False,
            "trainable": True,
            "description": "bounded trainable explicit source/closure latent smoke",
        },
        {
            "variant_id": "phase171_numpy_closure_head_control",
            "family": "control",
            "source_estimator": "phase169_calibrated_posterior_center_width",
            "closure_estimator": "phase171_train_split_linear_closure_head",
            "executed": True,
            "is_control": True,
            "trainable": False,
            "description": "required Phase 171 non-trainable closure-head control",
        },
        {
            "variant_id": "posterior_only_calibrated_bayesian_no_neural",
            "family": "control",
            "source_estimator": "phase169_calibrated_posterior_center_width",
            "closure_estimator": "posterior_point_estimate",
            "executed": True,
            "is_control": True,
            "trainable": False,
            "description": "strong no-neural calibrated posterior control",
        },
        {
            "variant_id": "grid_least_squares_source_closure_control",
            "family": "control",
            "source_estimator": "coarse_grid_best_sse_center_width",
            "closure_estimator": "least_squares_closure_coefficient",
            "executed": True,
            "is_control": True,
            "trainable": False,
            "description": "required non-Bayesian inverse grid control",
        },
        {
            "variant_id": "no_closure_source_control",
            "family": "control",
            "source_estimator": "grid_search_without_closure_term",
            "closure_estimator": "zero_closure",
            "executed": True,
            "is_control": True,
            "trainable": False,
            "description": "tests whether the hidden closure remains necessary",
        },
        {
            "variant_id": "wrong_source_prior_control",
            "family": "control",
            "source_estimator": "deliberately_shifted_source_prior",
            "closure_estimator": "least_squares_under_wrong_source",
            "executed": True,
            "is_control": True,
            "trainable": False,
            "description": "tests whether a wrong source prior can solve the task",
        },
        {
            "variant_id": "data_only_tiny_control",
            "family": "control",
            "source_estimator": "polynomial_sensor_regression_no_physics",
            "closure_estimator": "zero_closure",
            "executed": True,
            "is_control": True,
            "trainable": False,
            "description": "NumPy proxy for the registered data-only tiny control",
        },
        {
            "variant_id": "uniform_grid_latent_trainable_control",
            "family": "control",
            "source_estimator": "single_nominal_start_coordinate_search",
            "closure_estimator": "train_split_calibrated_optimized_closure_head",
            "executed": True,
            "is_control": True,
            "trainable": True,
            "description": "uniform/nominal-start trainable latent control",
        },
        {
            "variant_id": "failure_sampler_retrain_block",
            "family": "blocked_control",
            "source_estimator": "not_executed",
            "closure_estimator": "not_executed",
            "executed": False,
            "is_control": True,
            "trainable": False,
            "description": "registered block against repeating the Phase 167 sampler route",
        },
    ]


def _unique_starts(starts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique: list[tuple[float, float]] = []
    for center, width in starts:
        clipped = (
            float(np.clip(center, -0.070, 0.075)),
            float(np.clip(width, 0.025, 0.120)),
        )
        if not any(abs(clipped[0] - old[0]) < 1e-12 and abs(clipped[1] - old[1]) < 1e-12 for old in unique):
            unique.append(clipped)
    return unique


def _latent_objective(
    p169: Any,
    case: Any,
    center_shift: float,
    width: float,
) -> tuple[float, np.ndarray]:
    sse, coef = p169._fit_grid(
        case,
        float(center_shift),
        float(width),
        include_closure=True,
    )
    return float(sse / max(1, len(case.y))), coef


def _coordinate_search_latents(
    p169: Any,
    case: Any,
    starts: list[tuple[float, float]],
    *,
    max_rounds: int = 48,
    center_step: float = 0.012,
    width_step: float = 0.010,
) -> dict[str, Any]:
    """Optimize two explicit source latents with a bounded deterministic search."""

    best: dict[str, Any] | None = None
    total_rounds = 0
    total_evaluations = 0
    clean_starts = _unique_starts(starts)
    for start_index, (start_center, start_width) in enumerate(clean_starts):
        center = start_center
        width = start_width
        objective, coef = _latent_objective(p169, case, center, width)
        local_evaluations = 1
        local_rounds = 0
        delta_center = center_step
        delta_width = width_step
        for _ in range(max_rounds):
            local_rounds += 1
            improved = False
            candidates = [
                (center + delta_center, width),
                (center - delta_center, width),
                (center, width + delta_width),
                (center, width - delta_width),
                (center + 0.5 * delta_center, width + 0.5 * delta_width),
                (center - 0.5 * delta_center, width - 0.5 * delta_width),
                (center + 0.5 * delta_center, width - 0.5 * delta_width),
                (center - 0.5 * delta_center, width + 0.5 * delta_width),
            ]
            for next_center, next_width in candidates:
                next_center = float(np.clip(next_center, -0.070, 0.075))
                next_width = float(np.clip(next_width, 0.025, 0.120))
                next_objective, next_coef = _latent_objective(
                    p169,
                    case,
                    next_center,
                    next_width,
                )
                local_evaluations += 1
                if next_objective < objective - 1e-12:
                    center = next_center
                    width = next_width
                    objective = next_objective
                    coef = next_coef
                    improved = True
            if not improved:
                delta_center *= 0.55
                delta_width *= 0.55
                if max(delta_center, delta_width) < 1e-5:
                    break
        total_rounds += local_rounds
        total_evaluations += local_evaluations
        row = {
            "start_index": start_index,
            "start_count": len(clean_starts),
            "max_rounds_per_start": max_rounds,
            "executed_rounds_for_best_start": local_rounds,
            "function_evaluations_for_best_start": local_evaluations,
            "train_objective": objective,
            "learned_center_shift": center,
            "learned_source_width": width,
            "raw_closure_coeff": float(coef[-1]),
            "coefficients": coef,
        }
        if best is None or float(row["train_objective"]) < float(best["train_objective"]):
            best = row
    assert best is not None
    best["executed_rounds_total"] = total_rounds
    best["function_evaluations_total"] = total_evaluations
    return best


def _candidate_feature(result: dict[str, Any], posterior: dict[str, Any], raw: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            1.0,
            float(result["raw_closure_coeff"]),
            float(posterior["pred_closure_coeff"]),
            float(raw["best_grid_closure_coeff"]),
            float(result["learned_source_width"]),
            float(result["learned_center_shift"]),
        ],
        dtype=float,
    )


def _phase171_feature(posterior: dict[str, Any], raw: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            1.0,
            float(posterior["pred_closure_coeff"]),
            float(raw["best_grid_closure_coeff"]),
            float(posterior["pred_source_width"]),
            float(posterior["pred_center_shift"]),
        ],
        dtype=float,
    )


def _uniform_feature(result: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            1.0,
            float(result["raw_closure_coeff"]),
            float(result["learned_source_width"]),
            float(result["learned_center_shift"]),
        ],
        dtype=float,
    )


def _fit_linear_head(features: list[np.ndarray], targets: list[float]) -> np.ndarray:
    return np.linalg.lstsq(
        np.asarray(features, dtype=float),
        np.asarray(targets, dtype=float),
        rcond=None,
    )[0]


def _summary_rows(case_rows: list[dict[str, Any]], variant_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    family = {row["variant_id"]: row["family"] for row in variant_rows}
    rows: list[dict[str, Any]] = []
    for variant_id in sorted({row["variant_id"] for row in case_rows}):
        for split in ("val", "test"):
            subset = [
                row
                for row in case_rows
                if row["variant_id"] == variant_id and row["split"] == split
            ]
            if not subset:
                continue
            seeds = sorted({int(row["seed"]) for row in subset})
            scores = [float(row["selection_score"]) for row in subset]
            rows.append(
                {
                    "variant_id": variant_id,
                    "family": family.get(variant_id, "unknown"),
                    "split": split,
                    "seed_count": len(seeds),
                    "case_count": len(subset),
                    "field_rmse_mean": mean(float(row["field_rmse"]) for row in subset),
                    "hot_q90_rmse_mean": mean(float(row["hot_q90_rmse"]) for row in subset),
                    "gradient_q90_rmse_mean": mean(float(row["gradient_q90_rmse"]) for row in subset),
                    "closure_abs_error_mean": mean(float(row["closure_abs_error"]) for row in subset),
                    "coverage90_mean": mean(float(row["coverage90_mean"]) for row in subset),
                    "selection_score_mean": mean(scores),
                    "selection_score_std": pstdev(scores),
                }
            )
    return rows


def _seed_summary_rows(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in sorted({int(row["seed"]) for row in case_rows}):
        for variant_id in sorted({row["variant_id"] for row in case_rows}):
            for split in ("val", "test"):
                subset = [
                    row
                    for row in case_rows
                    if int(row["seed"]) == seed
                    and row["variant_id"] == variant_id
                    and row["split"] == split
                ]
                if not subset:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "variant_id": variant_id,
                        "split": split,
                        "case_count": len(subset),
                        "field_rmse_mean": mean(float(row["field_rmse"]) for row in subset),
                        "closure_abs_error_mean": mean(float(row["closure_abs_error"]) for row in subset),
                        "selection_score_mean": mean(float(row["selection_score"]) for row in subset),
                    }
                )
    return rows


def run_smoke(
    *,
    seeds: tuple[int, ...] = (171, 172, 173),
    noise_std: float = 0.025,
    center_grid_size: int = 38,
    width_grid_size: int = 34,
    max_latent_rounds: int = 48,
) -> dict[str, list[dict[str, Any]]]:
    p169 = _load_phase169_module()
    p171 = _load_phase171_module()
    center_grid = np.linspace(-0.060, 0.065, center_grid_size)
    width_grid = np.linspace(0.030, 0.105, width_grid_size)
    variant_rows = build_variant_rows()
    family = {row["variant_id"]: row["family"] for row in variant_rows}
    calibration_rows: list[dict[str, Any]] = []
    training_audit_rows: list[dict[str, Any]] = []
    case_metric_rows: list[dict[str, Any]] = []
    for seed in seeds:
        cases = p169.generate_cases(seed=seed, noise_std=noise_std)
        raw_posteriors = {
            case.case_id: p169.bayesian_hidden_source_closure_posterior(
                case,
                center_grid=center_grid,
                width_grid=width_grid,
            )
            for case in cases
        }
        calibrated_posteriors = p169.calibrated_bayesian_predictions(cases, raw_posteriors)
        no_closure_grid = {
            case.case_id: p171._best_no_closure_grid(p169, case, center_grid, width_grid)
            for case in cases
        }
        trainable_results: dict[str, dict[str, Any]] = {}
        uniform_results: dict[str, dict[str, Any]] = {}
        for case in cases:
            posterior = calibrated_posteriors[case.case_id]
            raw = raw_posteriors[case.case_id]
            candidate_starts = [
                (float(posterior["pred_center_shift"]), float(posterior["pred_source_width"])),
                (float(raw["best_grid_center_shift"]), float(raw["best_grid_source_width"])),
                (0.0, 0.064),
            ]
            trainable_results[case.case_id] = _coordinate_search_latents(
                p169,
                case,
                candidate_starts,
                max_rounds=max_latent_rounds,
            )
            uniform_results[case.case_id] = _coordinate_search_latents(
                p169,
                case,
                [(0.0, 0.064)],
                max_rounds=max_latent_rounds,
            )
        candidate_features: list[np.ndarray] = []
        candidate_targets: list[float] = []
        phase171_features: list[np.ndarray] = []
        uniform_features: list[np.ndarray] = []
        for case in cases:
            if case.split != "train":
                continue
            posterior = calibrated_posteriors[case.case_id]
            raw = raw_posteriors[case.case_id]
            candidate_features.append(
                _candidate_feature(trainable_results[case.case_id], posterior, raw)
            )
            phase171_features.append(_phase171_feature(posterior, raw))
            uniform_features.append(_uniform_feature(uniform_results[case.case_id]))
            candidate_targets.append(float(case.closure_coeff))
        candidate_coef = _fit_linear_head(candidate_features, candidate_targets)
        phase171_coef = _fit_linear_head(phase171_features, candidate_targets)
        uniform_coef = _fit_linear_head(uniform_features, candidate_targets)
        calibration_rows.extend(
            [
                {
                    "seed": seed,
                    "variant_id": "tiny_explicit_latent_hidden_closure_smoke",
                    "coefficient_intercept": float(candidate_coef[0]),
                    "coefficient_optimized_closure": float(candidate_coef[1]),
                    "coefficient_posterior_closure": float(candidate_coef[2]),
                    "coefficient_grid_closure": float(candidate_coef[3]),
                    "coefficient_source_width": float(candidate_coef[4]),
                    "coefficient_center_shift": float(candidate_coef[5]),
                    "train_case_count": len(candidate_targets),
                },
                {
                    "seed": seed,
                    "variant_id": "phase171_numpy_closure_head_control",
                    "coefficient_intercept": float(phase171_coef[0]),
                    "coefficient_optimized_closure": 0.0,
                    "coefficient_posterior_closure": float(phase171_coef[1]),
                    "coefficient_grid_closure": float(phase171_coef[2]),
                    "coefficient_source_width": float(phase171_coef[3]),
                    "coefficient_center_shift": float(phase171_coef[4]),
                    "train_case_count": len(candidate_targets),
                },
                {
                    "seed": seed,
                    "variant_id": "uniform_grid_latent_trainable_control",
                    "coefficient_intercept": float(uniform_coef[0]),
                    "coefficient_optimized_closure": float(uniform_coef[1]),
                    "coefficient_posterior_closure": 0.0,
                    "coefficient_grid_closure": 0.0,
                    "coefficient_source_width": float(uniform_coef[2]),
                    "coefficient_center_shift": float(uniform_coef[3]),
                    "train_case_count": len(candidate_targets),
                },
            ]
        )
        for case in cases:
            posterior = calibrated_posteriors[case.case_id]
            raw = raw_posteriors[case.case_id]
            trainable_result = trainable_results[case.case_id]
            uniform_result = uniform_results[case.case_id]
            trainable_closure = float(
                _candidate_feature(trainable_result, posterior, raw) @ candidate_coef
            )
            uniform_closure = float(_uniform_feature(uniform_result) @ uniform_coef)
            for variant_id, result, closure_value in (
                (
                    "tiny_explicit_latent_hidden_closure_smoke",
                    trainable_result,
                    trainable_closure,
                ),
                (
                    "uniform_grid_latent_trainable_control",
                    uniform_result,
                    uniform_closure,
                ),
            ):
                training_audit_rows.append(
                    {
                        "seed": seed,
                        "case_id": case.case_id,
                        "split": case.split,
                        "variant_id": variant_id,
                        "start_count": result["start_count"],
                        "max_rounds_per_start": result["max_rounds_per_start"],
                        "executed_rounds_total": result["executed_rounds_total"],
                        "function_evaluations_total": result["function_evaluations_total"],
                        "train_objective": result["train_objective"],
                        "learned_center_shift": result["learned_center_shift"],
                        "learned_source_width": result["learned_source_width"],
                        "raw_closure_coeff": result["raw_closure_coeff"],
                        "calibrated_closure_coeff": closure_value,
                    }
                )
        for case in cases:
            if case.split not in {"val", "test"}:
                continue
            posterior = calibrated_posteriors[case.case_id]
            raw = raw_posteriors[case.case_id]
            trainable_result = trainable_results[case.case_id]
            trainable_pred, true, gradient, _ = p171._field_prediction(
                p169,
                case,
                center_shift=float(trainable_result["learned_center_shift"]),
                width=float(trainable_result["learned_source_width"]),
                include_closure=True,
                split=case.split,
            )
            trainable_closure = float(
                _candidate_feature(trainable_result, posterior, raw) @ candidate_coef
            )
            phase171_pred, _, _, posterior_closure_for_field = p171._field_prediction(
                p169,
                case,
                center_shift=float(posterior["pred_center_shift"]),
                width=float(posterior["pred_source_width"]),
                include_closure=True,
                split=case.split,
            )
            phase171_closure = float(_phase171_feature(posterior, raw) @ phase171_coef)
            grid_pred, _, grid_gradient, grid_closure = p171._field_prediction(
                p169,
                case,
                center_shift=float(raw["best_grid_center_shift"]),
                width=float(raw["best_grid_source_width"]),
                include_closure=True,
                split=case.split,
            )
            no_closure = no_closure_grid[case.case_id]
            no_closure_pred, _, no_closure_gradient, no_closure_value = p171._field_prediction(
                p169,
                case,
                center_shift=float(no_closure[1]),
                width=float(no_closure[2]),
                include_closure=False,
                split=case.split,
            )
            wrong_pred, _, wrong_gradient, wrong_closure = p171._field_prediction(
                p169,
                case,
                center_shift=float(raw["best_grid_center_shift"]) + 0.035,
                width=float(raw["best_grid_source_width"]) * 1.25,
                include_closure=True,
                split=case.split,
            )
            data_pred, data_true, data_gradient = p171._data_only_prediction(
                p169,
                case,
                case.split,
            )
            uniform_result = uniform_results[case.case_id]
            uniform_pred, _, uniform_gradient, _ = p171._field_prediction(
                p169,
                case,
                center_shift=float(uniform_result["learned_center_shift"]),
                width=float(uniform_result["learned_source_width"]),
                include_closure=True,
                split=case.split,
            )
            uniform_closure = float(_uniform_feature(uniform_result) @ uniform_coef)
            coverage = p171._coverage(posterior, case)
            variant_payloads = {
                "tiny_explicit_latent_hidden_closure_smoke": (
                    trainable_pred,
                    true,
                    gradient,
                    trainable_closure,
                    coverage,
                ),
                "phase171_numpy_closure_head_control": (
                    phase171_pred,
                    true,
                    gradient,
                    phase171_closure,
                    coverage,
                ),
                "posterior_only_calibrated_bayesian_no_neural": (
                    phase171_pred,
                    true,
                    gradient,
                    posterior_closure_for_field,
                    coverage,
                ),
                "grid_least_squares_source_closure_control": (
                    grid_pred,
                    true,
                    grid_gradient,
                    grid_closure,
                    0.0,
                ),
                "no_closure_source_control": (
                    no_closure_pred,
                    true,
                    no_closure_gradient,
                    no_closure_value,
                    0.0,
                ),
                "wrong_source_prior_control": (
                    wrong_pred,
                    true,
                    wrong_gradient,
                    wrong_closure,
                    0.0,
                ),
                "data_only_tiny_control": (
                    data_pred,
                    data_true,
                    data_gradient,
                    0.0,
                    0.0,
                ),
                "uniform_grid_latent_trainable_control": (
                    uniform_pred,
                    true,
                    uniform_gradient,
                    uniform_closure,
                    0.0,
                ),
            }
            for variant_id, (pred, truth, grad, closure_pred, coverage90) in variant_payloads.items():
                metrics = p171._metric_payload(
                    pred=pred,
                    true=truth,
                    gradient=grad,
                    closure_pred=float(closure_pred),
                    coverage90_mean=float(coverage90),
                    case=case,
                )
                case_metric_rows.append(
                    {
                        "seed": seed,
                        "case_id": case.case_id,
                        "split": case.split,
                        "variant_id": variant_id,
                        "family": family[variant_id],
                        **metrics,
                    }
                )
    summary_rows = _summary_rows(case_metric_rows, variant_rows)
    seed_summary_rows = _seed_summary_rows(case_metric_rows)
    return {
        "variant_rows": variant_rows,
        "calibration_rows": calibration_rows,
        "training_audit_rows": training_audit_rows,
        "case_metric_rows": case_metric_rows,
        "summary_rows": summary_rows,
        "seed_summary_rows": seed_summary_rows,
    }


def _summary_lookup(rows: list[dict[str, Any]], variant_id: str, split: str) -> dict[str, Any]:
    for row in rows:
        if row["variant_id"] == variant_id and row["split"] == split:
            return row
    raise KeyError((variant_id, split))


def _seed_lookup(rows: list[dict[str, Any]], seed: int, variant_id: str, split: str) -> dict[str, Any]:
    for row in rows:
        if int(row["seed"]) == seed and row["variant_id"] == variant_id and row["split"] == split:
            return row
    raise KeyError((seed, variant_id, split))


def build_gate(
    *,
    phase172_gate: dict[str, Any],
    phase172_control_rows: list[dict[str, str]],
    phase171_gate: dict[str, Any],
    variant_rows: list[dict[str, Any]],
    training_audit_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    seed_summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase172_ready = (
        phase172_gate.get("status")
        == "phase172_trainable_hidden_closure_smoke_design_ready_phase173_low_budget_trainable_smoke"
        and _is_true(phase172_gate.get("phase173_low_budget_trainable_smoke_allowed"))
        and not _is_true(phase172_gate.get("phase172_model_training_allowed"))
    )
    phase171_ready = (
        phase171_gate.get("status")
        == "phase171_hidden_closure_low_budget_smoke_ready_phase172_trainable_design"
        and _is_true(phase171_gate.get("phase172_trainable_hidden_closure_design_allowed"))
    )
    required_controls = {
        "phase171_numpy_closure_head",
        "posterior_only_calibrated_bayesian_no_neural",
        "grid_least_squares_source_closure_control",
        "no_closure_source_control",
        "data_only_tiny_control",
        "wrong_source_prior_control",
        "uniform_grid_pinn_control",
        "failure_sampler_retrain_block",
        "seed_stability_control",
    }
    present_controls = {row.get("control_name", "") for row in phase172_control_rows}
    control_contract_ready = required_controls.issubset(present_controls)
    blocked_sampler_row = next(
        (row for row in variant_rows if row["variant_id"] == "failure_sampler_retrain_block"),
        None,
    )
    sampler_retrain_blocked = blocked_sampler_row is not None and not _is_true(
        blocked_sampler_row.get("executed")
    )
    candidate_id = "tiny_explicit_latent_hidden_closure_smoke"
    phase171_id = "phase171_numpy_closure_head_control"
    posterior_id = "posterior_only_calibrated_bayesian_no_neural"
    candidate_val = _summary_lookup(summary_rows, candidate_id, "val")
    candidate_test = _summary_lookup(summary_rows, candidate_id, "test")
    phase171_val = _summary_lookup(summary_rows, phase171_id, "val")
    phase171_test = _summary_lookup(summary_rows, phase171_id, "test")
    posterior_val = _summary_lookup(summary_rows, posterior_id, "val")
    control_val_rows = [
        row
        for row in summary_rows
        if row["split"] == "val" and row["family"] == "control"
    ]
    best_control_val = min(control_val_rows, key=lambda row: float(row["selection_score_mean"]))
    best_control_test = _summary_lookup(summary_rows, best_control_val["variant_id"], "test")
    selected_val = min(
        [row for row in summary_rows if row["split"] == "val"],
        key=lambda row: float(row["selection_score_mean"]),
    )
    validation_gain = float(best_control_val["selection_score_mean"]) - float(
        candidate_val["selection_score_mean"]
    )
    phase171_validation_score_gain = float(phase171_val["selection_score_mean"]) - float(
        candidate_val["selection_score_mean"]
    )
    phase171_test_score_gain = float(phase171_test["selection_score_mean"]) - float(
        candidate_test["selection_score_mean"]
    )
    phase171_validation_closure_gain = float(phase171_val["closure_abs_error_mean"]) - float(
        candidate_val["closure_abs_error_mean"]
    )
    phase171_test_closure_gain = float(phase171_test["closure_abs_error_mean"]) - float(
        candidate_test["closure_abs_error_mean"]
    )
    posterior_validation_score_gain = float(posterior_val["selection_score_mean"]) - float(
        candidate_val["selection_score_mean"]
    )
    test_reversal_ratio = float(candidate_test["selection_score_mean"]) / max(
        float(best_control_test["selection_score_mean"]),
        1e-12,
    )
    seeds = sorted({int(row["seed"]) for row in seed_summary_rows})
    stable_seed_count = 0
    for seed in seeds:
        candidate_seed = _seed_lookup(seed_summary_rows, seed, candidate_id, "val")
        phase171_seed = _seed_lookup(seed_summary_rows, seed, phase171_id, "val")
        posterior_seed = _seed_lookup(seed_summary_rows, seed, posterior_id, "val")
        control_seed_rows = [
            row
            for row in seed_summary_rows
            if int(row["seed"]) == seed
            and row["split"] == "val"
            and row["variant_id"] != candidate_id
        ]
        best_seed_control = min(
            control_seed_rows,
            key=lambda row: float(row["selection_score_mean"]),
        )
        if (
            float(candidate_seed["selection_score_mean"])
            < float(phase171_seed["selection_score_mean"])
            and float(candidate_seed["selection_score_mean"])
            < float(posterior_seed["selection_score_mean"])
            and float(candidate_seed["selection_score_mean"])
            < float(best_seed_control["selection_score_mean"])
        ):
            stable_seed_count += 1
    seed_pass_rate = stable_seed_count / max(1, len(seeds))
    trainable_audits = [
        row
        for row in training_audit_rows
        if row["variant_id"] == candidate_id
    ]
    max_start_count = max(int(row["start_count"]) for row in trainable_audits)
    max_rounds_per_start = max(int(row["max_rounds_per_start"]) for row in trainable_audits)
    max_executed_rounds = max(int(row["executed_rounds_total"]) for row in trainable_audits)
    max_function_evaluations = max(int(row["function_evaluations_total"]) for row in trainable_audits)
    budget_ok = (
        len(seeds) <= 3
        and max_start_count <= 3
        and max_rounds_per_start <= 48
        and max_executed_rounds <= 144
        and max_function_evaluations <= 1200
    )
    pass_gate = (
        phase172_ready
        and phase171_ready
        and control_contract_ready
        and sampler_retrain_blocked
        and selected_val["variant_id"] == candidate_id
        and validation_gain >= 0.005
        and phase171_validation_score_gain >= 0.010
        and phase171_test_score_gain >= 0.010
        and phase171_validation_closure_gain >= 0.005
        and phase171_test_closure_gain >= 0.002
        and posterior_validation_score_gain >= 0.010
        and test_reversal_ratio <= 0.98
        and 0.65 <= float(candidate_val["coverage90_mean"]) <= 1.0
        and 0.65 <= float(candidate_test["coverage90_mean"]) <= 1.0
        and seed_pass_rate >= 1.0
        and budget_ok
    )
    blockers: list[str] = []
    if not phase172_ready:
        blockers.append("phase172_gate_not_ready")
    if not phase171_ready:
        blockers.append("phase171_gate_not_ready")
    if not control_contract_ready:
        blockers.append("phase172_control_contract_missing")
    if not sampler_retrain_blocked:
        blockers.append("failure_sampler_retrain_block_missing")
    if selected_val["variant_id"] != candidate_id:
        blockers.append("validation_selected_control_variant")
    if validation_gain < 0.005:
        blockers.append("validation_gain_vs_best_control")
    if phase171_validation_score_gain < 0.010:
        blockers.append("phase171_validation_score_gain_guard")
    if phase171_test_score_gain < 0.010:
        blockers.append("phase171_test_score_gain_guard")
    if phase171_validation_closure_gain < 0.005:
        blockers.append("phase171_validation_closure_gain_guard")
    if phase171_test_closure_gain < 0.002:
        blockers.append("phase171_test_closure_gain_guard")
    if posterior_validation_score_gain < 0.010:
        blockers.append("posterior_validation_score_gain_guard")
    if test_reversal_ratio > 0.98:
        blockers.append("test_reversal_vs_best_control")
    if not (0.65 <= float(candidate_val["coverage90_mean"]) <= 1.0) or not (
        0.65 <= float(candidate_test["coverage90_mean"]) <= 1.0
    ):
        blockers.append("coverage_guard")
    if seed_pass_rate < 1.0:
        blockers.append("seed_stability_guard")
    if not budget_ok:
        blockers.append("compute_budget_guard")
    return {
        "status": (
            "phase173_trainable_hidden_closure_low_budget_smoke_ready_phase174_low_capacity_hidden_closure_design"
            if pass_gate
            else "phase173_trainable_hidden_closure_low_budget_smoke_closed_no_stable_trainable_gain"
        ),
        "selected_variant": selected_val["variant_id"],
        "candidate_variant": candidate_id,
        "best_control_variant": best_control_val["variant_id"],
        "candidate_validation_selection_score": candidate_val["selection_score_mean"],
        "best_control_validation_selection_score": best_control_val["selection_score_mean"],
        "validation_score_gain_vs_best_control": validation_gain,
        "candidate_test_selection_score": candidate_test["selection_score_mean"],
        "best_control_test_selection_score": best_control_test["selection_score_mean"],
        "test_reversal_ratio_vs_best_control": test_reversal_ratio,
        "phase171_validation_score_gain": phase171_validation_score_gain,
        "phase171_test_score_gain": phase171_test_score_gain,
        "phase171_validation_closure_gain": phase171_validation_closure_gain,
        "phase171_test_closure_gain": phase171_test_closure_gain,
        "posterior_validation_score_gain": posterior_validation_score_gain,
        "candidate_validation_field_rmse": candidate_val["field_rmse_mean"],
        "candidate_test_field_rmse": candidate_test["field_rmse_mean"],
        "candidate_validation_closure_abs_error": candidate_val["closure_abs_error_mean"],
        "candidate_test_closure_abs_error": candidate_test["closure_abs_error_mean"],
        "candidate_validation_coverage90_mean": candidate_val["coverage90_mean"],
        "candidate_test_coverage90_mean": candidate_test["coverage90_mean"],
        "seed_stability_pass_rate": seed_pass_rate,
        "seed_count": len(seeds),
        "max_trainable_start_count": max_start_count,
        "max_rounds_per_start": max_rounds_per_start,
        "max_executed_rounds_total": max_executed_rounds,
        "max_function_evaluations_total": max_function_evaluations,
        "blocking_audits": blockers,
        "phase174_low_capacity_hidden_closure_design_allowed": bool(pass_gate),
        "phase173_model_mechanism_allowed": False,
        "phase173_model_training_allowed": False,
        "phase174_training_allowed_now": False,
        "bayesian_pinn_training_allowed_now": False,
        "adaptive_sampling_training_allowed_now": False,
        "gcn_pinn_training_allowed_now": False,
        "cnn_operator_training_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": (
            "enter Phase 174 low-capacity hidden-closure design before any PINN training"
            if pass_gate
            else "close or redesign trainable hidden-closure route before training"
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
    variant_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    seed_summary_rows: list[dict[str, Any]],
) -> str:
    val_test_summary = [row for row in summary_rows if row["split"] in {"val", "test"}]
    lines = [
        "# Phase 173 Trainable Hidden-Closure Low-Budget Smoke",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Selected variant: `{gate['selected_variant']}`",
        f"- Best control variant: `{gate['best_control_variant']}`",
        f"- Validation score gain vs best control: `{_csv_value(gate['validation_score_gain_vs_best_control'])}`",
        f"- Phase 171 validation score gain: `{_csv_value(gate['phase171_validation_score_gain'])}`",
        f"- Phase 171 test closure gain: `{_csv_value(gate['phase171_test_closure_gain'])}`",
        f"- Phase 174 low-capacity hidden-closure design allowed: `{_csv_value(gate['phase174_low_capacity_hidden_closure_design_allowed'])}`",
        f"- Phase 173 model training allowed: `{_csv_value(gate['phase173_model_training_allowed'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "This is a tiny synthetic trainable-latent smoke. A positive gate means "
            "continuous explicit source/closure latents beat the Phase 171 closure "
            "head and strong posterior/grid controls under validation-only selection. "
            "It is not AM-Bench evidence, not Bayesian PINN training, not a GCN/CNN/"
            "operator route, and not an A100-80GB justification."
        ),
        "",
        "## Variants",
        *_markdown_table(variant_rows, VARIANT_FIELDS),
        "",
        "## Summary Metrics",
        *_markdown_table(val_test_summary, SUMMARY_FIELDS),
        "",
        "## Seed Summary",
        *_markdown_table(seed_summary_rows, SEED_SUMMARY_FIELDS),
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
    phase172_gate = _read_json(resolved["phase172_gate"])
    phase172_control_rows = _read_csv(resolved["phase172_control_table"])
    phase171_gate = _read_json(resolved["phase171_gate"])
    smoke = run_smoke()
    variant_rows = smoke["variant_rows"]
    calibration_rows = smoke["calibration_rows"]
    training_audit_rows = smoke["training_audit_rows"]
    case_metric_rows = smoke["case_metric_rows"]
    summary_rows = smoke["summary_rows"]
    seed_summary_rows = smoke["seed_summary_rows"]
    gate = build_gate(
        phase172_gate=phase172_gate,
        phase172_control_rows=phase172_control_rows,
        phase171_gate=phase171_gate,
        variant_rows=variant_rows,
        training_audit_rows=training_audit_rows,
        summary_rows=summary_rows,
        seed_summary_rows=seed_summary_rows,
    )

    variant_path = output_dir / "phase173_variant_table.csv"
    calibration_path = output_dir / "phase173_calibration_table.csv"
    training_audit_path = output_dir / "phase173_training_audit_table.csv"
    case_metric_path = output_dir / "phase173_case_metric_table.csv"
    summary_path = output_dir / "phase173_variant_summary_table.csv"
    seed_summary_path = output_dir / "phase173_seed_summary_table.csv"
    gate_path = output_dir / "phase173_trainable_hidden_closure_low_budget_smoke_gate.json"
    markdown_path = output_dir / "phase173_trainable_hidden_closure_low_budget_smoke.md"
    manifest_path = output_dir / "phase173_trainable_hidden_closure_low_budget_smoke_manifest.json"

    _write_csv(variant_path, variant_rows, VARIANT_FIELDS)
    _write_csv(calibration_path, calibration_rows, CALIBRATION_FIELDS)
    _write_csv(training_audit_path, training_audit_rows, TRAINING_AUDIT_FIELDS)
    _write_csv(case_metric_path, case_metric_rows, CASE_METRIC_FIELDS)
    _write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    _write_csv(seed_summary_path, seed_summary_rows, SEED_SUMMARY_FIELDS)
    _write_json(gate_path, gate)
    with markdown_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            build_markdown(
                gate=gate,
                variant_rows=variant_rows,
                summary_rows=summary_rows,
                seed_summary_rows=seed_summary_rows,
            )
        )

    manifest = {
        "phase": 173,
        "description": "bounded trainable hidden-source/closure low-budget synthetic smoke",
        "inputs": {name: _display_path(path, root) for name, path in resolved.items()},
        "outputs": {
            "variant_table": _display_path(variant_path, root),
            "calibration_table": _display_path(calibration_path, root),
            "training_audit_table": _display_path(training_audit_path, root),
            "case_metric_table": _display_path(case_metric_path, root),
            "variant_summary_table": _display_path(summary_path, root),
            "seed_summary_table": _display_path(seed_summary_path, root),
            "gate": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "variant_rows": len(variant_rows),
            "calibration_rows": len(calibration_rows),
            "training_audit_rows": len(training_audit_rows),
            "case_metric_rows": len(case_metric_rows),
            "summary_rows": len(summary_rows),
            "seed_summary_rows": len(seed_summary_rows),
            "phase172_control_rows": len(phase172_control_rows),
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
