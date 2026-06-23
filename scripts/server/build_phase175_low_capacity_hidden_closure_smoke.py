#!/usr/bin/env python3
"""Build Phase 175 bounded low-capacity hidden-closure smoke.

Phase 175 executes the tiny synthetic smoke opened by the Phase 174 design
gate. It tests whether a low-capacity explicit-latent head improves beyond the
Phase 173 trainable-latent control. It does not read AM-Bench/NIST raw data,
does not train a neural PINN, and does not justify A100-SXM4-80GB.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np


DEFAULT_OUTPUT_DIR = Path("docs/results/phase175_low_capacity_hidden_closure_smoke")

PHASE_INPUTS = {
    "phase174_gate": Path(
        "docs/results/phase174_low_capacity_hidden_closure_design_gate/"
        "phase174_low_capacity_hidden_closure_design_gate.json"
    ),
    "phase174_control_table": Path(
        "docs/results/phase174_low_capacity_hidden_closure_design_gate/"
        "phase174_control_table.csv"
    ),
    "phase173_gate": Path(
        "docs/results/phase173_trainable_hidden_closure_low_budget_smoke/"
        "phase173_trainable_hidden_closure_low_budget_smoke_gate.json"
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
    "head_target",
    "ridge_alpha",
    "coefficient_intercept",
    "coefficient_count",
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
    "ridge_alpha",
    "learned_center_shift",
    "learned_source_width",
    "raw_closure_coeff",
    "head_center_shift",
    "head_source_width",
    "head_closure_coeff",
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


def _load_phase173_module() -> Any:
    script = Path(__file__).with_name(
        "build_phase173_trainable_hidden_closure_low_budget_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("phase173_trainable_hidden_closure", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Phase 173 module from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_variant_rows() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "low_capacity_explicit_latent_hidden_closure_head",
            "family": "mechanism_candidate",
            "source_estimator": "ridge_head_over_phase173_latent_posterior_grid_features",
            "closure_estimator": "ridge_head_alpha_1e-6",
            "executed": True,
            "is_control": False,
            "trainable": True,
            "description": "low-capacity head predicting source center, width, and closure",
        },
        {
            "variant_id": "phase173_tiny_explicit_latent_hidden_closure_smoke",
            "family": "control",
            "source_estimator": "phase173_continuous_center_width_coordinate_search",
            "closure_estimator": "phase173_calibrated_optimized_closure_head",
            "executed": True,
            "is_control": True,
            "trainable": True,
            "description": "required Phase 173 trainable-latent control",
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


def _low_capacity_feature(result: dict[str, Any], posterior: dict[str, Any], raw: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            1.0,
            float(result["learned_center_shift"]),
            float(result["learned_source_width"]),
            float(result["raw_closure_coeff"]),
            float(posterior["pred_center_shift"]),
            float(posterior["pred_source_width"]),
            float(posterior["pred_closure_coeff"]),
            float(raw["best_grid_center_shift"]),
            float(raw["best_grid_source_width"]),
            float(raw["best_grid_closure_coeff"]),
        ],
        dtype=float,
    )


def _ridge_head(features: list[np.ndarray], targets: list[float], alpha: float) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    regularizer = alpha * np.eye(x.shape[1])
    regularizer[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + regularizer, x.T @ y)


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
    ridge_alpha: float = 1e-6,
) -> dict[str, list[dict[str, Any]]]:
    p169 = _load_phase169_module()
    p171 = _load_phase171_module()
    p173 = _load_phase173_module()
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
            starts = [
                (float(posterior["pred_center_shift"]), float(posterior["pred_source_width"])),
                (float(raw["best_grid_center_shift"]), float(raw["best_grid_source_width"])),
                (0.0, 0.064),
            ]
            trainable_results[case.case_id] = p173._coordinate_search_latents(
                p169,
                case,
                starts,
                max_rounds=max_latent_rounds,
            )
            uniform_results[case.case_id] = p173._coordinate_search_latents(
                p169,
                case,
                [(0.0, 0.064)],
                max_rounds=max_latent_rounds,
            )
        lowcap_features: list[np.ndarray] = []
        candidate_features: list[np.ndarray] = []
        phase171_features: list[np.ndarray] = []
        uniform_features: list[np.ndarray] = []
        center_targets: list[float] = []
        width_targets: list[float] = []
        closure_targets: list[float] = []
        for case in cases:
            if case.split != "train":
                continue
            posterior = calibrated_posteriors[case.case_id]
            raw = raw_posteriors[case.case_id]
            result = trainable_results[case.case_id]
            lowcap_features.append(_low_capacity_feature(result, posterior, raw))
            candidate_features.append(p173._candidate_feature(result, posterior, raw))
            phase171_features.append(p173._phase171_feature(posterior, raw))
            uniform_features.append(p173._uniform_feature(uniform_results[case.case_id]))
            center_targets.append(float(case.center_shift))
            width_targets.append(float(case.source_width))
            closure_targets.append(float(case.closure_coeff))
        lowcap_center_coef = _ridge_head(lowcap_features, center_targets, ridge_alpha)
        lowcap_width_coef = _ridge_head(lowcap_features, width_targets, ridge_alpha)
        lowcap_closure_coef = _ridge_head(lowcap_features, closure_targets, ridge_alpha)
        phase173_closure_coef = _fit_linear_head(candidate_features, closure_targets)
        phase171_closure_coef = _fit_linear_head(phase171_features, closure_targets)
        uniform_closure_coef = _fit_linear_head(uniform_features, closure_targets)
        calibration_rows.extend(
            [
                {
                    "seed": seed,
                    "variant_id": "low_capacity_explicit_latent_hidden_closure_head",
                    "head_target": "center_shift",
                    "ridge_alpha": ridge_alpha,
                    "coefficient_intercept": float(lowcap_center_coef[0]),
                    "coefficient_count": len(lowcap_center_coef),
                    "train_case_count": len(center_targets),
                },
                {
                    "seed": seed,
                    "variant_id": "low_capacity_explicit_latent_hidden_closure_head",
                    "head_target": "source_width",
                    "ridge_alpha": ridge_alpha,
                    "coefficient_intercept": float(lowcap_width_coef[0]),
                    "coefficient_count": len(lowcap_width_coef),
                    "train_case_count": len(width_targets),
                },
                {
                    "seed": seed,
                    "variant_id": "low_capacity_explicit_latent_hidden_closure_head",
                    "head_target": "closure_coeff",
                    "ridge_alpha": ridge_alpha,
                    "coefficient_intercept": float(lowcap_closure_coef[0]),
                    "coefficient_count": len(lowcap_closure_coef),
                    "train_case_count": len(closure_targets),
                },
            ]
        )
        for case in cases:
            posterior = calibrated_posteriors[case.case_id]
            raw = raw_posteriors[case.case_id]
            result = trainable_results[case.case_id]
            lowcap_feature = _low_capacity_feature(result, posterior, raw)
            lowcap_center = float(np.clip(lowcap_feature @ lowcap_center_coef, -0.070, 0.075))
            lowcap_width = float(np.clip(lowcap_feature @ lowcap_width_coef, 0.025, 0.120))
            lowcap_closure = float(lowcap_feature @ lowcap_closure_coef)
            training_audit_rows.append(
                {
                    "seed": seed,
                    "case_id": case.case_id,
                    "split": case.split,
                    "variant_id": "low_capacity_explicit_latent_hidden_closure_head",
                    "start_count": result["start_count"],
                    "max_rounds_per_start": result["max_rounds_per_start"],
                    "executed_rounds_total": result["executed_rounds_total"],
                    "function_evaluations_total": result["function_evaluations_total"],
                    "ridge_alpha": ridge_alpha,
                    "learned_center_shift": result["learned_center_shift"],
                    "learned_source_width": result["learned_source_width"],
                    "raw_closure_coeff": result["raw_closure_coeff"],
                    "head_center_shift": lowcap_center,
                    "head_source_width": lowcap_width,
                    "head_closure_coeff": lowcap_closure,
                }
            )
        for case in cases:
            if case.split not in {"val", "test"}:
                continue
            posterior = calibrated_posteriors[case.case_id]
            raw = raw_posteriors[case.case_id]
            result = trainable_results[case.case_id]
            lowcap_feature = _low_capacity_feature(result, posterior, raw)
            lowcap_center = float(np.clip(lowcap_feature @ lowcap_center_coef, -0.070, 0.075))
            lowcap_width = float(np.clip(lowcap_feature @ lowcap_width_coef, 0.025, 0.120))
            lowcap_closure = float(lowcap_feature @ lowcap_closure_coef)
            lowcap_pred, true, gradient, _ = p171._field_prediction(
                p169,
                case,
                center_shift=lowcap_center,
                width=lowcap_width,
                include_closure=True,
                split=case.split,
            )
            phase173_pred, _, phase173_gradient, _ = p171._field_prediction(
                p169,
                case,
                center_shift=float(result["learned_center_shift"]),
                width=float(result["learned_source_width"]),
                include_closure=True,
                split=case.split,
            )
            phase173_closure = float(
                p173._candidate_feature(result, posterior, raw) @ phase173_closure_coef
            )
            phase171_pred, _, _, posterior_closure_for_field = p171._field_prediction(
                p169,
                case,
                center_shift=float(posterior["pred_center_shift"]),
                width=float(posterior["pred_source_width"]),
                include_closure=True,
                split=case.split,
            )
            phase171_closure = float(
                p173._phase171_feature(posterior, raw) @ phase171_closure_coef
            )
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
            uniform_closure = float(
                p173._uniform_feature(uniform_result) @ uniform_closure_coef
            )
            coverage = p171._coverage(posterior, case)
            variant_payloads = {
                "low_capacity_explicit_latent_hidden_closure_head": (
                    lowcap_pred,
                    true,
                    gradient,
                    lowcap_closure,
                    coverage,
                ),
                "phase173_tiny_explicit_latent_hidden_closure_smoke": (
                    phase173_pred,
                    true,
                    phase173_gradient,
                    phase173_closure,
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
    phase174_gate: dict[str, Any],
    phase174_control_rows: list[dict[str, str]],
    phase173_gate: dict[str, Any],
    variant_rows: list[dict[str, Any]],
    training_audit_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    seed_summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase174_ready = (
        phase174_gate.get("status")
        == "phase174_low_capacity_hidden_closure_design_ready_phase175_low_capacity_smoke"
        and _is_true(phase174_gate.get("phase175_low_capacity_smoke_allowed"))
        and not _is_true(phase174_gate.get("phase174_model_training_allowed"))
    )
    phase173_ready = (
        phase173_gate.get("status")
        == "phase173_trainable_hidden_closure_low_budget_smoke_ready_phase174_low_capacity_hidden_closure_design"
        and _is_true(phase173_gate.get("phase174_low_capacity_hidden_closure_design_allowed"))
    )
    required_controls = {
        "phase173_tiny_explicit_latent_hidden_closure_smoke",
        "phase171_numpy_closure_head_control",
        "posterior_only_calibrated_bayesian_no_neural",
        "grid_least_squares_source_closure_control",
        "no_closure_source_control",
        "wrong_source_prior_control",
        "data_only_tiny_control",
        "uniform_grid_latent_trainable_control",
        "failure_sampler_retrain_block",
        "seed_stability_control",
    }
    present_controls = {row.get("control_name", "") for row in phase174_control_rows}
    control_contract_ready = required_controls.issubset(present_controls)
    blocked_sampler_row = next(
        (row for row in variant_rows if row["variant_id"] == "failure_sampler_retrain_block"),
        None,
    )
    sampler_retrain_blocked = blocked_sampler_row is not None and not _is_true(
        blocked_sampler_row.get("executed")
    )
    candidate_id = "low_capacity_explicit_latent_hidden_closure_head"
    phase173_id = "phase173_tiny_explicit_latent_hidden_closure_smoke"
    candidate_val = _summary_lookup(summary_rows, candidate_id, "val")
    candidate_test = _summary_lookup(summary_rows, candidate_id, "test")
    phase173_val = _summary_lookup(summary_rows, phase173_id, "val")
    phase173_test = _summary_lookup(summary_rows, phase173_id, "test")
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
    phase173_validation_score_gain = float(phase173_val["selection_score_mean"]) - float(
        candidate_val["selection_score_mean"]
    )
    phase173_test_score_gain = float(phase173_test["selection_score_mean"]) - float(
        candidate_test["selection_score_mean"]
    )
    phase173_validation_closure_gain = float(phase173_val["closure_abs_error_mean"]) - float(
        candidate_val["closure_abs_error_mean"]
    )
    phase173_test_closure_gain = float(phase173_test["closure_abs_error_mean"]) - float(
        candidate_test["closure_abs_error_mean"]
    )
    test_reversal_ratio = float(candidate_test["selection_score_mean"]) / max(
        float(best_control_test["selection_score_mean"]),
        1e-12,
    )
    seeds = sorted({int(row["seed"]) for row in seed_summary_rows})
    stable_seed_count = 0
    for seed in seeds:
        candidate_seed = _seed_lookup(seed_summary_rows, seed, candidate_id, "val")
        phase173_seed = _seed_lookup(seed_summary_rows, seed, phase173_id, "val")
        if float(candidate_seed["selection_score_mean"]) < float(
            phase173_seed["selection_score_mean"]
        ):
            stable_seed_count += 1
    seed_pass_rate = stable_seed_count / max(1, len(seeds))
    lowcap_audits = [
        row
        for row in training_audit_rows
        if row["variant_id"] == candidate_id
    ]
    max_start_count = max(int(row["start_count"]) for row in lowcap_audits)
    max_rounds_per_start = max(int(row["max_rounds_per_start"]) for row in lowcap_audits)
    max_executed_rounds = max(int(row["executed_rounds_total"]) for row in lowcap_audits)
    max_function_evaluations = max(int(row["function_evaluations_total"]) for row in lowcap_audits)
    budget_ok = (
        len(seeds) <= 3
        and max_start_count <= 3
        and max_rounds_per_start <= 48
        and max_executed_rounds <= 144
        and max_function_evaluations <= 1200
    )
    pass_gate = (
        phase174_ready
        and phase173_ready
        and control_contract_ready
        and sampler_retrain_blocked
        and selected_val["variant_id"] == candidate_id
        and validation_gain >= 0.002
        and phase173_validation_score_gain >= 0.002
        and phase173_test_score_gain >= 0.0
        and phase173_validation_closure_gain >= 0.0
        and phase173_test_closure_gain >= 0.0
        and test_reversal_ratio <= 1.02
        and 0.65 <= float(candidate_val["coverage90_mean"]) <= 1.0
        and 0.65 <= float(candidate_test["coverage90_mean"]) <= 1.0
        and seed_pass_rate >= 1.0
        and budget_ok
    )
    blockers: list[str] = []
    if not phase174_ready:
        blockers.append("phase174_gate_not_ready")
    if not phase173_ready:
        blockers.append("phase173_gate_not_ready")
    if not control_contract_ready:
        blockers.append("phase174_control_contract_missing")
    if not sampler_retrain_blocked:
        blockers.append("failure_sampler_retrain_block_missing")
    if selected_val["variant_id"] != candidate_id:
        blockers.append("validation_selected_control_variant")
    if validation_gain < 0.002:
        blockers.append("validation_gain_vs_best_control")
    if phase173_validation_score_gain < 0.002:
        blockers.append("phase173_validation_score_gain_guard")
    if phase173_test_score_gain < 0.0:
        blockers.append("phase173_test_score_gain_guard")
    if phase173_validation_closure_gain < 0.0:
        blockers.append("phase173_validation_closure_gain_guard")
    if phase173_test_closure_gain < 0.0:
        blockers.append("phase173_test_closure_gain_guard")
    if test_reversal_ratio > 1.02:
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
            "phase175_low_capacity_hidden_closure_smoke_ready_phase176_focused_review"
            if pass_gate
            else "phase175_low_capacity_hidden_closure_smoke_closed_no_incremental_gain"
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
        "phase173_validation_score_gain": phase173_validation_score_gain,
        "phase173_test_score_gain": phase173_test_score_gain,
        "phase173_validation_closure_gain": phase173_validation_closure_gain,
        "phase173_test_closure_gain": phase173_test_closure_gain,
        "candidate_validation_field_rmse": candidate_val["field_rmse_mean"],
        "candidate_test_field_rmse": candidate_test["field_rmse_mean"],
        "candidate_validation_closure_abs_error": candidate_val["closure_abs_error_mean"],
        "candidate_test_closure_abs_error": candidate_test["closure_abs_error_mean"],
        "candidate_validation_coverage90_mean": candidate_val["coverage90_mean"],
        "candidate_test_coverage90_mean": candidate_test["coverage90_mean"],
        "seed_stability_pass_rate": seed_pass_rate,
        "seed_count": len(seeds),
        "max_lowcap_start_count": max_start_count,
        "max_rounds_per_start": max_rounds_per_start,
        "max_executed_rounds_total": max_executed_rounds,
        "max_function_evaluations_total": max_function_evaluations,
        "blocking_audits": blockers,
        "phase176_focused_review_allowed": bool(pass_gate),
        "phase175_model_mechanism_allowed": False,
        "phase175_model_training_allowed": False,
        "phase176_training_allowed_now": False,
        "bayesian_pinn_training_allowed_now": False,
        "adaptive_sampling_training_allowed_now": False,
        "gcn_pinn_training_allowed_now": False,
        "cnn_operator_training_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": (
            "enter Phase 176 focused review before any AM or PINN training"
            if pass_gate
            else "close low-capacity expansion and redesign with a different mechanism or refresh evidence"
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
        "# Phase 175 Low-Capacity Hidden-Closure Smoke",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Selected variant: `{gate['selected_variant']}`",
        f"- Candidate variant: `{gate['candidate_variant']}`",
        f"- Best control variant: `{gate['best_control_variant']}`",
        f"- Validation score gain vs best control: `{_csv_value(gate['validation_score_gain_vs_best_control'])}`",
        f"- Phase 173 validation score gain: `{_csv_value(gate['phase173_validation_score_gain'])}`",
        f"- Phase 176 focused review allowed: `{_csv_value(gate['phase176_focused_review_allowed'])}`",
        f"- Phase 175 model training allowed: `{_csv_value(gate['phase175_model_training_allowed'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "This is a tiny synthetic low-capacity smoke. A closed gate means the "
            "extra low-capacity head did not beat the simpler Phase 173 explicit "
            "latent control under validation-only selection. It is not AM-Bench "
            "evidence, not Bayesian PINN training, not a GCN/CNN/operator route, "
            "and not an A100-80GB justification."
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
    phase174_gate = _read_json(resolved["phase174_gate"])
    phase174_control_rows = _read_csv(resolved["phase174_control_table"])
    phase173_gate = _read_json(resolved["phase173_gate"])
    smoke = run_smoke()
    variant_rows = smoke["variant_rows"]
    calibration_rows = smoke["calibration_rows"]
    training_audit_rows = smoke["training_audit_rows"]
    case_metric_rows = smoke["case_metric_rows"]
    summary_rows = smoke["summary_rows"]
    seed_summary_rows = smoke["seed_summary_rows"]
    gate = build_gate(
        phase174_gate=phase174_gate,
        phase174_control_rows=phase174_control_rows,
        phase173_gate=phase173_gate,
        variant_rows=variant_rows,
        training_audit_rows=training_audit_rows,
        summary_rows=summary_rows,
        seed_summary_rows=seed_summary_rows,
    )

    variant_path = output_dir / "phase175_variant_table.csv"
    calibration_path = output_dir / "phase175_calibration_table.csv"
    training_audit_path = output_dir / "phase175_training_audit_table.csv"
    case_metric_path = output_dir / "phase175_case_metric_table.csv"
    summary_path = output_dir / "phase175_variant_summary_table.csv"
    seed_summary_path = output_dir / "phase175_seed_summary_table.csv"
    gate_path = output_dir / "phase175_low_capacity_hidden_closure_smoke_gate.json"
    markdown_path = output_dir / "phase175_low_capacity_hidden_closure_smoke.md"
    manifest_path = output_dir / "phase175_low_capacity_hidden_closure_smoke_manifest.json"

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
        "phase": 175,
        "description": "bounded low-capacity hidden-closure synthetic smoke",
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
            "phase174_control_rows": len(phase174_control_rows),
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
