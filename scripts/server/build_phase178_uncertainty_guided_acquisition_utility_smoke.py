#!/usr/bin/env python3
"""Build Phase 178 no-training uncertainty-guided acquisition utility smoke.

Phase 178 executes the no-training smoke opened by Phase 177. It evaluates
whether posterior/ensemble uncertainty can choose sparse synthetic observations
that reduce hidden source/closure uncertainty better than uniform, random, and
no-new-observation controls. It does not train a PINN, read raw data, or request
A100-SXM4-80GB.
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
    "docs/results/phase178_uncertainty_guided_acquisition_utility_smoke"
)

PHASE_INPUTS = {
    "phase177_gate": Path(
        "docs/results/phase177_uncertainty_guided_latent_acquisition_design_gate/"
        "phase177_uncertainty_guided_latent_acquisition_design_gate.json"
    ),
    "phase177_acquisition_table": Path(
        "docs/results/phase177_uncertainty_guided_latent_acquisition_design_gate/"
        "phase177_acquisition_policy_table.csv"
    ),
    "phase177_control_table": Path(
        "docs/results/phase177_uncertainty_guided_latent_acquisition_design_gate/"
        "phase177_control_table.csv"
    ),
}

POLICY_FIELDS = (
    "policy_id",
    "family",
    "executed",
    "is_control",
    "description",
)

CASE_METRIC_FIELDS = (
    "seed",
    "case_id",
    "split",
    "policy_id",
    "family",
    "selected_count",
    "posterior_trace_before",
    "posterior_trace_after",
    "posterior_trace_contraction",
    "parameter_error_before",
    "parameter_error_after",
    "parameter_error_gain",
    "closure_abs_error_before",
    "closure_abs_error_after",
    "closure_abs_error_gain",
    "duplicate_fraction",
    "boundary_fraction",
    "utility_score",
)

SUMMARY_FIELDS = (
    "policy_id",
    "family",
    "split",
    "seed_count",
    "case_count",
    "posterior_trace_contraction_mean",
    "parameter_error_gain_mean",
    "closure_abs_error_gain_mean",
    "parameter_error_after_mean",
    "closure_abs_error_after_mean",
    "duplicate_fraction_mean",
    "boundary_fraction_mean",
    "utility_score_mean",
    "utility_score_std",
)

SEED_SUMMARY_FIELDS = (
    "seed",
    "policy_id",
    "split",
    "case_count",
    "posterior_trace_contraction_mean",
    "parameter_error_gain_mean",
    "closure_abs_error_gain_mean",
    "utility_score_mean",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


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
    path.write_text(json.dumps(_stable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
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
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


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


def _load_phase169_module() -> Any:
    script = Path(__file__).with_name(
        "build_phase169_hidden_source_closure_identifiability_gate.py"
    )
    spec = importlib.util.spec_from_file_location("phase169_for_phase178", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Phase 169 module from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_policy_rows() -> list[dict[str, Any]]:
    return [
        {
            "policy_id": "posterior_entropy_reduction_candidate",
            "family": "acquisition_candidate",
            "executed": True,
            "is_control": False,
            "description": "selects points with high posterior predictive variance",
        },
        {
            "policy_id": "latent_ensemble_disagreement_candidate",
            "family": "acquisition_candidate",
            "executed": True,
            "is_control": False,
            "description": "selects points with high top-latent ensemble disagreement",
        },
        {
            "policy_id": "hybrid_uncertainty_hot_gradient_candidate",
            "family": "acquisition_candidate",
            "executed": True,
            "is_control": False,
            "description": "combines posterior variance with a fixed hot-gradient quota proxy",
        },
        {
            "policy_id": "uniform_budget_control",
            "family": "control",
            "executed": True,
            "is_control": True,
            "description": "same-budget uniform candidate-pool acquisition",
        },
        {
            "policy_id": "random_budget_control",
            "family": "control",
            "executed": True,
            "is_control": True,
            "description": "same-budget deterministic random acquisition",
        },
        {
            "policy_id": "no_new_observation_control",
            "family": "control",
            "executed": True,
            "is_control": True,
            "description": "posterior update without adding observations",
        },
    ]


def _initial_indices() -> np.ndarray:
    pairs = [(ti, xi) for ti in (0, 2, 5, 8) for xi in (1, 4, 7, 10)]
    return np.asarray(sorted(ti * 12 + xi for ti, xi in pairs), dtype=int)


def _case_view(p169: Any, case: Any, indices: np.ndarray) -> Any:
    return p169.HiddenSourceCase(
        case_id=case.case_id,
        split=case.split,
        center_shift=case.center_shift,
        source_width=case.source_width,
        closure_coeff=case.closure_coeff,
        source_amplitude=case.source_amplitude,
        ambient=case.ambient,
        x=case.x[indices],
        t=case.t[indices],
        y=case.y[indices],
        noise_std=case.noise_std,
    )


def _posterior_particles(
    p169: Any,
    case_view: Any,
    *,
    center_grid: np.ndarray,
    width_grid: np.ndarray,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for center_shift in center_grid:
        for width in width_grid:
            sse, coef = p169._fit_grid(
                case_view,
                float(center_shift),
                float(width),
                include_closure=True,
            )
            rows.append(
                {
                    "center_shift": float(center_shift),
                    "source_width": float(width),
                    "closure_coeff": float(coef[-1]),
                    "sse": float(sse),
                    "coef": coef,
                }
            )
    sse_values = np.asarray([row["sse"] for row in rows], dtype=float)
    sigma2 = max(
        case_view.noise_std**2,
        float(np.min(sse_values)) / max(1, len(case_view.y) - 5),
    )
    log_weights = -0.5 * (sse_values - float(np.min(sse_values))) / sigma2
    weights = np.exp(np.clip(log_weights, -700.0, 0.0))
    weights = weights / float(np.sum(weights))
    params = np.asarray(
        [[row["center_shift"], row["source_width"], row["closure_coeff"]] for row in rows],
        dtype=float,
    )
    mean_params = weights @ params
    scales = np.asarray([0.097, 0.060, 0.220], dtype=float)
    trace = float(weights @ np.sum(((params - mean_params) / scales) ** 2, axis=1))
    return {
        "rows": rows,
        "weights": weights,
        "mean_params": mean_params,
        "trace": trace,
    }


def _predictive_variance(
    p169: Any,
    case: Any,
    candidate_indices: np.ndarray,
    posterior: dict[str, Any],
    *,
    top_equal_weight: bool,
) -> np.ndarray:
    rows = posterior["rows"]
    if top_equal_weight:
        order = np.argsort([row["sse"] for row in rows])[:16]
        active_rows = [rows[int(index)] for index in order]
        weights = np.full(len(active_rows), 1.0 / len(active_rows), dtype=float)
    else:
        active_rows = rows
        weights = posterior["weights"]
    predictions: list[np.ndarray] = []
    for row in active_rows:
        design = p169._design_matrix(
            case.x[candidate_indices],
            case.t[candidate_indices],
            row["center_shift"],
            row["source_width"],
            include_closure=True,
        )
        predictions.append(design @ row["coef"])
    matrix = np.vstack(predictions)
    mean_prediction = weights @ matrix
    return weights @ ((matrix - mean_prediction) ** 2)


def _hot_gradient_proxy(p169: Any, case: Any, candidate_indices: np.ndarray, mean_params: np.ndarray) -> np.ndarray:
    center_shift, width, closure_coeff = [float(value) for value in mean_params]
    x = case.x[candidate_indices]
    t = case.t[candidate_indices]
    center = 0.22 + 0.56 * t + center_shift
    hot = np.exp(-0.5 * ((x - center) / max(width, 1e-8)) ** 2)
    closure = np.abs(
        p169._closure_basis(
            x,
            t,
            center_shift,
            max(width, 1e-8),
        )
        * closure_coeff
    )
    score = hot + 0.35 * closure
    return (score - float(np.min(score))) / (float(np.ptp(score)) + 1e-12)


def _select_indices(
    *,
    p169: Any,
    case: Any,
    candidate_indices: np.ndarray,
    posterior: dict[str, Any],
    policy_id: str,
    seed: int,
    budget: int,
) -> np.ndarray:
    if policy_id == "no_new_observation_control":
        return np.asarray([], dtype=int)
    if policy_id == "uniform_budget_control":
        return candidate_indices[np.linspace(0, len(candidate_indices) - 1, budget, dtype=int)]
    if policy_id == "random_budget_control":
        case_number = int(str(case.case_id).split("-")[-1])
        rng = np.random.default_rng(seed + case_number)
        return np.sort(rng.choice(candidate_indices, size=budget, replace=False))
    if policy_id == "posterior_entropy_reduction_candidate":
        score = _predictive_variance(
            p169,
            case,
            candidate_indices,
            posterior,
            top_equal_weight=False,
        )
    elif policy_id == "latent_ensemble_disagreement_candidate":
        score = _predictive_variance(
            p169,
            case,
            candidate_indices,
            posterior,
            top_equal_weight=True,
        )
    elif policy_id == "hybrid_uncertainty_hot_gradient_candidate":
        variance = _predictive_variance(
            p169,
            case,
            candidate_indices,
            posterior,
            top_equal_weight=False,
        )
        variance = (variance - float(np.min(variance))) / (float(np.ptp(variance)) + 1e-12)
        score = variance + 0.35 * _hot_gradient_proxy(
            p169,
            case,
            candidate_indices,
            posterior["mean_params"],
        )
    else:
        raise KeyError(policy_id)
    return np.sort(candidate_indices[np.argsort(score)[-budget:]])


def _normalized_parameter_error(case: Any, mean_params: np.ndarray) -> float:
    truth = np.asarray([case.center_shift, case.source_width, case.closure_coeff], dtype=float)
    scales = np.asarray([0.097, 0.060, 0.220], dtype=float)
    return float(np.linalg.norm((mean_params - truth) / scales))


def _case_metric(
    *,
    p169: Any,
    case: Any,
    policy_id: str,
    family: str,
    seed: int,
    initial_indices: np.ndarray,
    center_grid: np.ndarray,
    width_grid: np.ndarray,
    budget: int,
) -> dict[str, Any]:
    all_indices = np.arange(len(case.y), dtype=int)
    initial_set = set(int(index) for index in initial_indices)
    candidate_indices = np.asarray(
        [index for index in all_indices if int(index) not in initial_set],
        dtype=int,
    )
    before = _posterior_particles(
        p169,
        _case_view(p169, case, initial_indices),
        center_grid=center_grid,
        width_grid=width_grid,
    )
    selected = _select_indices(
        p169=p169,
        case=case,
        candidate_indices=candidate_indices,
        posterior=before,
        policy_id=policy_id,
        seed=seed,
        budget=budget,
    )
    after_indices = np.concatenate([initial_indices, selected])
    after = _posterior_particles(
        p169,
        _case_view(p169, case, after_indices),
        center_grid=center_grid,
        width_grid=width_grid,
    )
    parameter_error_before = _normalized_parameter_error(case, before["mean_params"])
    parameter_error_after = _normalized_parameter_error(case, after["mean_params"])
    closure_before = abs(float(before["mean_params"][2]) - float(case.closure_coeff))
    closure_after = abs(float(after["mean_params"][2]) - float(case.closure_coeff))
    duplicate_fraction = 0.0
    if len(selected) > 0:
        duplicate_fraction = 1.0 - len(set(int(index) for index in selected)) / len(selected)
    boundary_fraction = (
        float(np.mean((case.x[selected] <= 0.08) | (case.x[selected] >= 0.92)))
        if len(selected) > 0
        else 0.0
    )
    trace_contraction = float(before["trace"] - after["trace"])
    parameter_gain = float(parameter_error_before - parameter_error_after)
    closure_gain = float(closure_before - closure_after)
    utility_score = parameter_gain + closure_gain / 0.220 + 0.10 * trace_contraction
    return {
        "seed": seed,
        "case_id": case.case_id,
        "split": case.split,
        "policy_id": policy_id,
        "family": family,
        "selected_count": len(selected),
        "posterior_trace_before": before["trace"],
        "posterior_trace_after": after["trace"],
        "posterior_trace_contraction": trace_contraction,
        "parameter_error_before": parameter_error_before,
        "parameter_error_after": parameter_error_after,
        "parameter_error_gain": parameter_gain,
        "closure_abs_error_before": closure_before,
        "closure_abs_error_after": closure_after,
        "closure_abs_error_gain": closure_gain,
        "duplicate_fraction": duplicate_fraction,
        "boundary_fraction": boundary_fraction,
        "utility_score": utility_score,
    }


def _summary_rows(case_rows: list[dict[str, Any]], policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    family = {row["policy_id"]: row["family"] for row in policy_rows}
    rows: list[dict[str, Any]] = []
    for policy_id in sorted({row["policy_id"] for row in case_rows}):
        for split in ("val", "test"):
            subset = [
                row for row in case_rows if row["policy_id"] == policy_id and row["split"] == split
            ]
            if not subset:
                continue
            scores = [float(row["utility_score"]) for row in subset]
            rows.append(
                {
                    "policy_id": policy_id,
                    "family": family.get(policy_id, "unknown"),
                    "split": split,
                    "seed_count": len({int(row["seed"]) for row in subset}),
                    "case_count": len(subset),
                    "posterior_trace_contraction_mean": mean(
                        float(row["posterior_trace_contraction"]) for row in subset
                    ),
                    "parameter_error_gain_mean": mean(
                        float(row["parameter_error_gain"]) for row in subset
                    ),
                    "closure_abs_error_gain_mean": mean(
                        float(row["closure_abs_error_gain"]) for row in subset
                    ),
                    "parameter_error_after_mean": mean(
                        float(row["parameter_error_after"]) for row in subset
                    ),
                    "closure_abs_error_after_mean": mean(
                        float(row["closure_abs_error_after"]) for row in subset
                    ),
                    "duplicate_fraction_mean": mean(
                        float(row["duplicate_fraction"]) for row in subset
                    ),
                    "boundary_fraction_mean": mean(
                        float(row["boundary_fraction"]) for row in subset
                    ),
                    "utility_score_mean": mean(scores),
                    "utility_score_std": pstdev(scores),
                }
            )
    return rows


def _seed_summary_rows(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in sorted({int(row["seed"]) for row in case_rows}):
        for policy_id in sorted({row["policy_id"] for row in case_rows}):
            for split in ("val", "test"):
                subset = [
                    row
                    for row in case_rows
                    if int(row["seed"]) == seed
                    and row["policy_id"] == policy_id
                    and row["split"] == split
                ]
                if not subset:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "policy_id": policy_id,
                        "split": split,
                        "case_count": len(subset),
                        "posterior_trace_contraction_mean": mean(
                            float(row["posterior_trace_contraction"]) for row in subset
                        ),
                        "parameter_error_gain_mean": mean(
                            float(row["parameter_error_gain"]) for row in subset
                        ),
                        "closure_abs_error_gain_mean": mean(
                            float(row["closure_abs_error_gain"]) for row in subset
                        ),
                        "utility_score_mean": mean(float(row["utility_score"]) for row in subset),
                    }
                )
    return rows


def run_smoke(
    *,
    seeds: tuple[int, ...] = (178, 179, 180),
    noise_std: float = 0.025,
    budget: int = 12,
    center_grid_size: int = 18,
    width_grid_size: int = 15,
) -> dict[str, list[dict[str, Any]]]:
    p169 = _load_phase169_module()
    policy_rows = build_policy_rows()
    initial_indices = _initial_indices()
    center_grid = np.linspace(-0.060, 0.065, center_grid_size)
    width_grid = np.linspace(0.030, 0.105, width_grid_size)
    case_metric_rows: list[dict[str, Any]] = []
    for seed in seeds:
        cases = [
            case
            for case in p169.generate_cases(seed=seed, noise_std=noise_std)
            if case.split in {"val", "test"}
        ]
        for case in cases:
            for policy in policy_rows:
                case_metric_rows.append(
                    _case_metric(
                        p169=p169,
                        case=case,
                        policy_id=policy["policy_id"],
                        family=policy["family"],
                        seed=seed,
                        initial_indices=initial_indices,
                        center_grid=center_grid,
                        width_grid=width_grid,
                        budget=budget,
                    )
                )
    summary_rows = _summary_rows(case_metric_rows, policy_rows)
    seed_summary_rows = _seed_summary_rows(case_metric_rows)
    return {
        "policy_rows": policy_rows,
        "case_metric_rows": case_metric_rows,
        "summary_rows": summary_rows,
        "seed_summary_rows": seed_summary_rows,
    }


def _summary_lookup(rows: list[dict[str, Any]], policy_id: str, split: str) -> dict[str, Any]:
    for row in rows:
        if row["policy_id"] == policy_id and row["split"] == split:
            return row
    raise KeyError((policy_id, split))


def _seed_lookup(rows: list[dict[str, Any]], seed: int, policy_id: str, split: str) -> dict[str, Any]:
    for row in rows:
        if int(row["seed"]) == seed and row["policy_id"] == policy_id and row["split"] == split:
            return row
    raise KeyError((seed, policy_id, split))


def build_gate(
    *,
    phase177_gate: dict[str, Any],
    phase177_acquisition_rows: list[dict[str, str]],
    phase177_control_rows: list[dict[str, str]],
    policy_rows: list[dict[str, Any]],
    case_metric_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    seed_summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase177_ready = (
        phase177_gate.get("status")
        == "phase177_uncertainty_guided_latent_acquisition_design_ready_phase178_no_training_smoke"
        and _is_true(phase177_gate.get("phase178_no_training_acquisition_smoke_allowed"))
        and _is_false(phase177_gate.get("phase177_model_training_allowed"))
        and _is_false(phase177_gate.get("phase178_training_allowed_now"))
        and _is_false(phase177_gate.get("a100_80gb_request_now"))
    )
    acquisition_contract_ready = len(phase177_acquisition_rows) >= 6
    control_contract_ready = len(phase177_control_rows) >= 10
    val_rows = [row for row in summary_rows if row["split"] == "val"]
    selected = max(val_rows, key=lambda row: float(row["utility_score_mean"]))
    candidate_rows = [row for row in val_rows if row["family"] == "acquisition_candidate"]
    control_rows = [row for row in val_rows if row["family"] == "control"]
    best_candidate = max(candidate_rows, key=lambda row: float(row["utility_score_mean"]))
    best_control = max(control_rows, key=lambda row: float(row["utility_score_mean"]))
    candidate_test = _summary_lookup(summary_rows, best_candidate["policy_id"], "test")
    control_test = _summary_lookup(summary_rows, best_control["policy_id"], "test")
    validation_gain = float(best_candidate["utility_score_mean"]) - float(
        best_control["utility_score_mean"]
    )
    test_reversal = float(best_candidate["utility_score_mean"]) - float(
        control_test["utility_score_mean"]
    )
    closure_gain_vs_control = float(best_candidate["closure_abs_error_gain_mean"]) - float(
        best_control["closure_abs_error_gain_mean"]
    )
    seeds = sorted({int(row["seed"]) for row in seed_summary_rows})
    stable_seed_count = 0
    for seed in seeds:
        candidate_seed = _seed_lookup(
            seed_summary_rows,
            seed,
            best_candidate["policy_id"],
            "val",
        )
        control_seed = _seed_lookup(seed_summary_rows, seed, best_control["policy_id"], "val")
        if float(candidate_seed["utility_score_mean"]) > float(
            control_seed["utility_score_mean"]
        ):
            stable_seed_count += 1
    seed_stability_pass_rate = stable_seed_count / max(1, len(seeds))
    duplicate_ok = all(float(row["duplicate_fraction"]) == 0.0 for row in case_metric_rows)
    boundary_ok = max(float(row["boundary_fraction"]) for row in case_metric_rows) <= 0.75
    selected_candidate = selected["family"] == "acquisition_candidate"
    pass_gate = (
        phase177_ready
        and acquisition_contract_ready
        and control_contract_ready
        and selected_candidate
        and validation_gain >= 0.01
        and test_reversal >= -0.005
        and closure_gain_vs_control >= 0.0
        and seed_stability_pass_rate >= 1.0
        and duplicate_ok
        and boundary_ok
    )
    blockers: list[str] = []
    if not phase177_ready:
        blockers.append("phase177_gate_not_ready")
    if not acquisition_contract_ready:
        blockers.append("phase177_acquisition_contract_missing")
    if not control_contract_ready:
        blockers.append("phase177_control_contract_missing")
    if not selected_candidate:
        blockers.append("validation_selected_control_policy")
    if validation_gain < 0.01:
        blockers.append("validation_utility_gain_vs_best_control")
    if test_reversal < -0.005:
        blockers.append("test_utility_reversal_vs_best_control")
    if closure_gain_vs_control < 0.0:
        blockers.append("closure_gain_vs_best_control")
    if seed_stability_pass_rate < 1.0:
        blockers.append("seed_stability_guard")
    if not duplicate_ok:
        blockers.append("duplicate_acquisition_guard")
    if not boundary_ok:
        blockers.append("boundary_fraction_guard")
    return {
        "status": (
            "phase178_uncertainty_guided_acquisition_smoke_ready_phase179_training_design"
            if pass_gate
            else "phase178_uncertainty_guided_acquisition_smoke_closed_no_guarded_acquisition_gain"
        ),
        "selected_policy": selected["policy_id"],
        "best_candidate_policy": best_candidate["policy_id"],
        "best_control_policy": best_control["policy_id"],
        "candidate_validation_utility_score": best_candidate["utility_score_mean"],
        "best_control_validation_utility_score": best_control["utility_score_mean"],
        "validation_utility_gain_vs_best_control": validation_gain,
        "candidate_test_utility_score": candidate_test["utility_score_mean"],
        "best_control_test_utility_score": control_test["utility_score_mean"],
        "test_utility_reversal_vs_best_control": test_reversal,
        "candidate_validation_closure_gain": best_candidate["closure_abs_error_gain_mean"],
        "best_control_validation_closure_gain": best_control["closure_abs_error_gain_mean"],
        "closure_gain_vs_best_control": closure_gain_vs_control,
        "candidate_validation_parameter_gain": best_candidate["parameter_error_gain_mean"],
        "best_control_validation_parameter_gain": best_control["parameter_error_gain_mean"],
        "candidate_validation_trace_contraction": best_candidate[
            "posterior_trace_contraction_mean"
        ],
        "best_control_validation_trace_contraction": best_control[
            "posterior_trace_contraction_mean"
        ],
        "seed_stability_pass_rate": seed_stability_pass_rate,
        "seed_count": len(seeds),
        "policy_rows": len(policy_rows),
        "case_metric_rows": len(case_metric_rows),
        "summary_rows": len(summary_rows),
        "blocking_audits": blockers,
        "phase179_training_design_allowed": bool(pass_gate),
        "phase178_model_mechanism_allowed": False,
        "phase178_model_training_allowed": False,
        "phase179_training_allowed_now": False,
        "bayesian_pinn_training_allowed_now": False,
        "adaptive_sampling_training_allowed_now": False,
        "gcn_pinn_training_allowed_now": False,
        "cnn_operator_training_allowed_now": False,
        "am_bench_training_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": (
            "enter Phase 179 training design only"
            if pass_gate
            else "close uncertainty-guided acquisition route or redesign before any training"
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
    policy_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    seed_summary_rows: list[dict[str, Any]],
) -> str:
    val_test_summary = [row for row in summary_rows if row["split"] in {"val", "test"}]
    lines = [
        "# Phase 178 Uncertainty-Guided Acquisition Utility Smoke",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Selected policy: `{gate['selected_policy']}`",
        f"- Best candidate policy: `{gate['best_candidate_policy']}`",
        f"- Best control policy: `{gate['best_control_policy']}`",
        f"- Validation utility gain vs best control: `{_csv_value(gate['validation_utility_gain_vs_best_control'])}`",
        f"- Phase 179 training design allowed: `{_csv_value(gate['phase179_training_design_allowed'])}`",
        f"- Phase 178 model training allowed: `{_csv_value(gate['phase178_model_training_allowed'])}`",
        f"- Phase 179 training allowed now: `{_csv_value(gate['phase179_training_allowed_now'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "This no-training smoke tests acquisition utility only. If a same-budget "
            "uniform/random/no-new control wins validation selection, the uncertainty "
            "acquisition route closes before any model training."
        ),
        "",
        "## Policies",
        *_markdown_table(policy_rows, POLICY_FIELDS),
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
    phase177_gate = _read_json(resolved["phase177_gate"])
    phase177_acquisition_rows = _read_csv(resolved["phase177_acquisition_table"])
    phase177_control_rows = _read_csv(resolved["phase177_control_table"])
    smoke = run_smoke()
    policy_rows = smoke["policy_rows"]
    case_metric_rows = smoke["case_metric_rows"]
    summary_rows = smoke["summary_rows"]
    seed_summary_rows = smoke["seed_summary_rows"]
    gate = build_gate(
        phase177_gate=phase177_gate,
        phase177_acquisition_rows=phase177_acquisition_rows,
        phase177_control_rows=phase177_control_rows,
        policy_rows=policy_rows,
        case_metric_rows=case_metric_rows,
        summary_rows=summary_rows,
        seed_summary_rows=seed_summary_rows,
    )

    policy_path = output_dir / "phase178_acquisition_policy_table.csv"
    case_metric_path = output_dir / "phase178_acquisition_case_metric_table.csv"
    summary_path = output_dir / "phase178_acquisition_summary_table.csv"
    seed_summary_path = output_dir / "phase178_acquisition_seed_summary_table.csv"
    gate_path = output_dir / "phase178_uncertainty_guided_acquisition_utility_gate.json"
    markdown_path = output_dir / "phase178_uncertainty_guided_acquisition_utility_smoke.md"
    manifest_path = output_dir / "phase178_uncertainty_guided_acquisition_utility_manifest.json"

    _write_csv(policy_path, policy_rows, POLICY_FIELDS)
    _write_csv(case_metric_path, case_metric_rows, CASE_METRIC_FIELDS)
    _write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    _write_csv(seed_summary_path, seed_summary_rows, SEED_SUMMARY_FIELDS)
    _write_json(gate_path, gate)
    with markdown_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            build_markdown(
                gate=gate,
                policy_rows=policy_rows,
                summary_rows=summary_rows,
                seed_summary_rows=seed_summary_rows,
            )
        )

    manifest = {
        "phase": 178,
        "description": "no-training uncertainty-guided acquisition utility smoke",
        "inputs": {name: _display_path(path, root) for name, path in sorted(resolved.items())},
        "outputs": {
            "policy_table": _display_path(policy_path, root),
            "case_metric_table": _display_path(case_metric_path, root),
            "summary_table": _display_path(summary_path, root),
            "seed_summary_table": _display_path(seed_summary_path, root),
            "gate": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "policy_rows": len(policy_rows),
            "case_metric_rows": len(case_metric_rows),
            "summary_rows": len(summary_rows),
            "seed_summary_rows": len(seed_summary_rows),
            "phase177_acquisition_rows": len(phase177_acquisition_rows),
            "phase177_control_rows": len(phase177_control_rows),
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
    manifest = build_package(
        root=args.root,
        output_dir=args.output_dir,
        phase_inputs=phase_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
