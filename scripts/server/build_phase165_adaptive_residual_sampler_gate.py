#!/usr/bin/env python3
"""Build Phase 165 adaptive residual sampler no-training gate.

This phase evaluates collocation point sets only. It uses analytic moving
heat-source residual/gradient fields and fixed point budgets to decide whether
adaptive residual sampling is worth a later low-budget PINN smoke. No neural
model is trained in this phase.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUTPUT_DIR = Path("docs/results/phase165_adaptive_residual_sampler_gate")

PHASE_INPUTS = {
    "phase164_gate": Path(
        "docs/results/phase164_synthetic_bayesian_inverse_heat_identifiability_gate/"
        "phase164_synthetic_bayesian_inverse_heat_identifiability_gate.json"
    ),
    "phase163_literature_table": Path(
        "docs/results/phase163_pinn_bayesian_hybrid_roadmap/"
        "phase163_literature_evidence_table.csv"
    ),
}

SAMPLER_FIELDS = (
    "sampler_id",
    "sampler_family",
    "description",
    "uses_residual",
    "uses_hot_gradient",
    "is_control",
)

METRIC_FIELDS = (
    "sampler_id",
    "sampler_family",
    "scenario",
    "seed_count",
    "point_budget",
    "high_residual_recall_mean",
    "hot_recall_mean",
    "gradient_recall_mean",
    "boundary_fraction_mean",
    "coverage_uniformity_mean",
    "score_mean",
    "score_std",
)

SEED_FIELDS = (
    "sampler_id",
    "scenario",
    "seed",
    "point_budget",
    "high_residual_recall",
    "hot_recall",
    "gradient_recall",
    "boundary_fraction",
    "coverage_uniformity",
    "score",
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


def _is_false(value: Any) -> bool:
    if value is False or value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "0", "false", "none", "no"}
    return False


def build_sampler_rows() -> list[dict[str, Any]]:
    return [
        {
            "sampler_id": "uniform_grid_control",
            "sampler_family": "control",
            "description": "deterministic approximately uniform grid subset",
            "uses_residual": False,
            "uses_hot_gradient": False,
            "is_control": True,
        },
        {
            "sampler_id": "jittered_stratified_control",
            "sampler_family": "control",
            "description": "stratified grid with deterministic seed jitter",
            "uses_residual": False,
            "uses_hot_gradient": False,
            "is_control": True,
        },
        {
            "sampler_id": "residual_density_adaptive",
            "sampler_family": "adaptive",
            "description": "sample proportional to analytic residual density",
            "uses_residual": True,
            "uses_hot_gradient": False,
            "is_control": False,
        },
        {
            "sampler_id": "rad_rar_d_residual_gradient",
            "sampler_family": "adaptive",
            "description": "residual-adaptive distribution using residual and gradient scores",
            "uses_residual": True,
            "uses_hot_gradient": True,
            "is_control": False,
        },
        {
            "sampler_id": "failure_informed_hot_gradient",
            "sampler_family": "adaptive",
            "description": "failure-informed mix of residual, hotspot, gradient, and boundary quotas",
            "uses_residual": True,
            "uses_hot_gradient": True,
            "is_control": False,
        },
    ]


def analytic_field(scenario: str) -> dict[str, np.ndarray]:
    x_values = np.linspace(0.0, 1.0, 80)
    t_values = np.linspace(0.0, 1.0, 56)
    t_grid, x_grid = np.meshgrid(t_values, x_values, indexing="ij")
    x = x_grid.ravel()
    t = t_grid.ravel()
    if scenario == "validation_nominal":
        center = 0.22 + 0.52 * t
        width = 0.045 + 0.025 * t
        amp = 1.0
    elif scenario == "test_shifted":
        center = 0.18 + 0.60 * t + 0.025 * np.sin(2.0 * math.pi * t)
        width = 0.038 + 0.034 * t
        amp = 0.92
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    radius = x - center
    source = amp * np.exp(-0.5 * (radius / width) ** 2)
    residual = np.abs(source * (1.0 + 4.0 * np.abs(radius) / np.maximum(width, 1e-8)))
    gradient = np.abs(source * radius / np.maximum(width**2, 1e-8))
    boundary = ((x <= 0.04) | (x >= 0.96) | (t <= 0.04) | (t >= 0.96)).astype(float)
    return {
        "x": x,
        "t": t,
        "source": source,
        "residual": residual,
        "gradient": gradient,
        "boundary": boundary,
    }


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    span = float(values.max() - values.min())
    if span <= 0.0:
        return np.zeros_like(values)
    return (values - float(values.min())) / span


def _top_mask(values: np.ndarray, quantile: float = 0.90) -> np.ndarray:
    threshold = float(np.quantile(values, quantile))
    return values >= threshold


def _weighted_choice(rng: np.random.Generator, weights: np.ndarray, budget: int) -> np.ndarray:
    safe = np.maximum(weights, 0.0).astype(float)
    safe = safe + 1e-12
    probs = safe / float(np.sum(safe))
    return np.sort(rng.choice(np.arange(len(weights)), size=budget, replace=False, p=probs))


def _uniform_grid_indices(field: dict[str, np.ndarray], budget: int) -> np.ndarray:
    x = field["x"]
    t = field["t"]
    side_t = max(2, int(round(math.sqrt(budget * 0.7))))
    side_x = max(2, int(math.ceil(budget / side_t)))
    target_t = np.linspace(0.0, 1.0, side_t)
    target_x = np.linspace(0.0, 1.0, side_x)
    chosen: list[int] = []
    used: set[int] = set()
    for tv in target_t:
        for xv in target_x:
            score = (x - xv) ** 2 + (t - tv) ** 2
            for index in np.argsort(score):
                idx = int(index)
                if idx not in used:
                    used.add(idx)
                    chosen.append(idx)
                    break
            if len(chosen) >= budget:
                return np.asarray(sorted(chosen), dtype=int)
    return np.asarray(sorted(chosen[:budget]), dtype=int)


def _jittered_stratified_indices(
    field: dict[str, np.ndarray],
    budget: int,
    rng: np.random.Generator,
) -> np.ndarray:
    x = field["x"]
    t = field["t"]
    bins_t = 14
    bins_x = 16
    chosen: list[int] = []
    used: set[int] = set()
    for ti in range(bins_t):
        for xi in range(bins_x):
            mask = (
                (t >= ti / bins_t)
                & (t < (ti + 1) / bins_t)
                & (x >= xi / bins_x)
                & (x < (xi + 1) / bins_x)
            )
            candidates = np.where(mask)[0]
            if len(candidates):
                idx = int(rng.choice(candidates))
                used.add(idx)
                chosen.append(idx)
            if len(chosen) >= budget:
                return np.asarray(sorted(chosen), dtype=int)
    remaining = np.asarray([idx for idx in range(len(x)) if idx not in used], dtype=int)
    fill = rng.choice(remaining, size=max(0, budget - len(chosen)), replace=False)
    return np.asarray(sorted([*chosen, *[int(idx) for idx in fill]]), dtype=int)


def sample_indices(
    sampler_id: str,
    field: dict[str, np.ndarray],
    *,
    budget: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    residual = _normalize(field["residual"])
    gradient = _normalize(field["gradient"])
    hot = _normalize(field["source"])
    boundary = field["boundary"]
    if sampler_id == "uniform_grid_control":
        return _uniform_grid_indices(field, budget)
    if sampler_id == "jittered_stratified_control":
        return _jittered_stratified_indices(field, budget, rng)
    if sampler_id == "residual_density_adaptive":
        return _weighted_choice(rng, residual**2 + 0.02, budget)
    if sampler_id == "rad_rar_d_residual_gradient":
        return _weighted_choice(rng, residual**1.5 + 0.7 * gradient + 0.02, budget)
    if sampler_id == "failure_informed_hot_gradient":
        selected: set[int] = set()
        quotas = [
            (_top_mask(field["residual"], 0.90), residual + 0.2 * gradient, int(0.34 * budget)),
            (_top_mask(field["source"], 0.90), hot + 0.3 * residual, int(0.22 * budget)),
            (_top_mask(field["gradient"], 0.90), gradient + 0.2 * residual, int(0.22 * budget)),
            (boundary > 0, boundary + 0.05, int(0.12 * budget)),
        ]
        for mask, weights, count in quotas:
            candidates = np.where(mask)[0]
            available = np.asarray([idx for idx in candidates if int(idx) not in selected], dtype=int)
            if len(available) == 0 or count <= 0:
                continue
            local_weights = weights[available] + 1e-12
            take = min(count, len(available))
            chosen = rng.choice(
                available,
                size=take,
                replace=False,
                p=local_weights / float(np.sum(local_weights)),
            )
            selected.update(int(idx) for idx in chosen)
        fill_count = budget - len(selected)
        if fill_count > 0:
            weights = residual + gradient + hot + 0.1 * boundary + 0.02
            available = np.asarray(
                [idx for idx in range(len(weights)) if idx not in selected],
                dtype=int,
            )
            local_weights = weights[available]
            chosen = rng.choice(
                available,
                size=fill_count,
                replace=False,
                p=local_weights / float(np.sum(local_weights)),
            )
            selected.update(int(idx) for idx in chosen)
        return np.asarray(sorted(selected), dtype=int)
    raise ValueError(f"Unknown sampler: {sampler_id}")


def evaluate_selection(
    field: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    budget: int,
) -> dict[str, float]:
    selected = np.zeros(len(field["x"]), dtype=bool)
    selected[indices] = True
    high_residual = _top_mask(field["residual"], 0.90)
    hot = _top_mask(field["source"], 0.90)
    gradient = _top_mask(field["gradient"], 0.90)
    boundary = field["boundary"] > 0
    high_residual_recall = float(np.sum(selected & high_residual) / np.sum(high_residual))
    hot_recall = float(np.sum(selected & hot) / np.sum(hot))
    gradient_recall = float(np.sum(selected & gradient) / np.sum(gradient))
    boundary_fraction = float(np.mean(boundary[indices]))
    x = field["x"][indices]
    t = field["t"][indices]
    hist, _, _ = np.histogram2d(t, x, bins=(7, 8), range=((0.0, 1.0), (0.0, 1.0)))
    occupied = float(np.mean(hist > 0))
    expected_fraction = min(1.0, budget / len(field["x"]) * 8.0)
    coverage_uniformity = min(1.0, occupied / max(expected_fraction, 1e-12))
    score = (
        0.42 * high_residual_recall
        + 0.22 * hot_recall
        + 0.22 * gradient_recall
        + 0.08 * min(boundary_fraction / 0.12, 1.0)
        + 0.06 * coverage_uniformity
    )
    return {
        "high_residual_recall": high_residual_recall,
        "hot_recall": hot_recall,
        "gradient_recall": gradient_recall,
        "boundary_fraction": boundary_fraction,
        "coverage_uniformity": coverage_uniformity,
        "score": score,
    }


def build_seed_rows(
    sampler_rows: list[dict[str, Any]],
    *,
    budget: int = 256,
    seeds: tuple[int, ...] = (165, 166, 167),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in ("validation_nominal", "test_shifted"):
        field = analytic_field(scenario)
        for sampler in sampler_rows:
            for seed in seeds:
                indices = sample_indices(sampler["sampler_id"], field, budget=budget, seed=seed)
                metrics = evaluate_selection(field, indices, budget=budget)
                rows.append(
                    {
                        "sampler_id": sampler["sampler_id"],
                        "scenario": scenario,
                        "seed": seed,
                        "point_budget": budget,
                        **metrics,
                    }
                )
    return rows


def build_metric_rows(
    sampler_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    family_by_id = {row["sampler_id"]: row["sampler_family"] for row in sampler_rows}
    output: list[dict[str, Any]] = []
    for sampler_id in sorted({row["sampler_id"] for row in seed_rows}):
        for scenario in ("validation_nominal", "test_shifted"):
            subset = [
                row
                for row in seed_rows
                if row["sampler_id"] == sampler_id and row["scenario"] == scenario
            ]
            output.append(
                {
                    "sampler_id": sampler_id,
                    "sampler_family": family_by_id[sampler_id],
                    "scenario": scenario,
                    "seed_count": len(subset),
                    "point_budget": subset[0]["point_budget"] if subset else 0,
                    "high_residual_recall_mean": float(
                        np.mean([row["high_residual_recall"] for row in subset])
                    ),
                    "hot_recall_mean": float(np.mean([row["hot_recall"] for row in subset])),
                    "gradient_recall_mean": float(
                        np.mean([row["gradient_recall"] for row in subset])
                    ),
                    "boundary_fraction_mean": float(
                        np.mean([row["boundary_fraction"] for row in subset])
                    ),
                    "coverage_uniformity_mean": float(
                        np.mean([row["coverage_uniformity"] for row in subset])
                    ),
                    "score_mean": float(np.mean([row["score"] for row in subset])),
                    "score_std": float(np.std([row["score"] for row in subset], ddof=0)),
                }
            )
    return output


def _metric_lookup(rows: list[dict[str, Any]], sampler_id: str, scenario: str) -> dict[str, Any]:
    for row in rows:
        if row["sampler_id"] == sampler_id and row["scenario"] == scenario:
            return row
    raise KeyError((sampler_id, scenario))


def build_gate(
    *,
    phase164_gate: dict[str, Any],
    literature_rows: list[dict[str, str]],
    metric_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase164_ready = (
        phase164_gate.get("status")
        == "phase164_synthetic_bayesian_inverse_heat_identifiability_ready_phase165_sampler_gate"
        and _is_true(phase164_gate.get("phase165_adaptive_sampler_gate_allowed"))
    )
    adaptive_lit_rows = [
        row
        for row in literature_rows
        if row.get("source_id") in {"P163-LIT-003", "P163-LIT-004", "P163-LIT-010"}
    ]
    val_rows = [row for row in metric_rows if row["scenario"] == "validation_nominal"]
    selected = max(val_rows, key=lambda row: float(row["score_mean"]))
    control_rows = [row for row in val_rows if row["sampler_family"] == "control"]
    best_control = max(control_rows, key=lambda row: float(row["score_mean"]))
    selected_test = _metric_lookup(metric_rows, selected["sampler_id"], "test_shifted")
    control_test = _metric_lookup(metric_rows, best_control["sampler_id"], "test_shifted")
    validation_gain = float(selected["score_mean"]) - float(best_control["score_mean"])
    test_gain = float(selected_test["score_mean"]) - float(control_test["score_mean"])
    boundary_ok = (
        float(selected["boundary_fraction_mean"]) >= 0.08
        and float(selected_test["boundary_fraction_mean"]) >= 0.08
    )
    stability_ok = (
        float(selected["score_std"]) <= 0.04 and float(selected_test["score_std"]) <= 0.04
    )
    adaptive_selected = selected["sampler_family"] == "adaptive"
    pass_gate = (
        phase164_ready
        and len(adaptive_lit_rows) >= 3
        and adaptive_selected
        and validation_gain >= 0.08
        and test_gain >= 0.05
        and boundary_ok
        and stability_ok
    )
    blockers: list[str] = []
    if not phase164_ready:
        blockers.append("phase164_gate_not_ready")
    if len(adaptive_lit_rows) < 3:
        blockers.append("adaptive_literature_evidence_missing")
    if not adaptive_selected:
        blockers.append("validation_selected_control_sampler")
    if validation_gain < 0.08:
        blockers.append("validation_gain_vs_best_control")
    if test_gain < 0.05:
        blockers.append("test_gain_vs_best_control")
    if not boundary_ok:
        blockers.append("boundary_coverage_guard")
    if not stability_ok:
        blockers.append("seed_stability_guard")
    return {
        "status": (
            "phase165_adaptive_residual_sampler_ready_low_budget_pinn_smoke_design"
            if pass_gate
            else "phase165_adaptive_residual_sampler_closed_no_stable_sampler_gain"
        ),
        "selected_sampler": selected["sampler_id"],
        "best_control_sampler": best_control["sampler_id"],
        "validation_score_gain_vs_best_control": validation_gain,
        "test_score_gain_vs_best_control": test_gain,
        "selected_validation_score": selected["score_mean"],
        "selected_test_score": selected_test["score_mean"],
        "selected_validation_boundary_fraction": selected["boundary_fraction_mean"],
        "selected_test_boundary_fraction": selected_test["boundary_fraction_mean"],
        "selected_validation_score_std": selected["score_std"],
        "selected_test_score_std": selected_test["score_std"],
        "blocking_audits": blockers,
        "phase166_low_budget_pinn_smoke_design_allowed": bool(pass_gate),
        "phase165_low_capacity_training_allowed": False,
        "phase165_model_mechanism_allowed": False,
        "phase165_model_training_allowed": False,
        "bayesian_pinn_training_allowed_now": False,
        "adaptive_sampling_training_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": (
            "enter Phase 166 as a low-budget PINN smoke design only, still gated before training"
            if pass_gate
            else "close or redesign adaptive sampler route before any model training"
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
    lines = [
        "# Phase 165 Adaptive Residual Sampler Gate",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Selected sampler: `{gate['selected_sampler']}`",
        f"- Best control sampler: `{gate['best_control_sampler']}`",
        f"- Validation score gain vs best control: `{_csv_value(gate['validation_score_gain_vs_best_control'])}`",
        f"- Test score gain vs best control: `{_csv_value(gate['test_score_gain_vs_best_control'])}`",
        f"- Phase 166 low-budget PINN smoke design allowed: `{_csv_value(gate['phase166_low_budget_pinn_smoke_design_allowed'])}`",
        f"- Phase 165 model training allowed: `{_csv_value(gate['phase165_model_training_allowed'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "This package evaluates point sets only. A positive result means the adaptive "
            "sampler covers high-residual, hot, and high-gradient regions better than "
            "uniform controls at the same point budget on analytic heat fields. It does "
            "not train a PINN."
        ),
        "",
        "## Sampler Metrics",
        *_markdown_table(metric_rows, METRIC_FIELDS),
        "",
    ]
    return "\n".join(lines)


def build_package(*, root: Path, output_dir: Path, phase_inputs: dict[str, Path]) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    resolved = {
        name: path if path.is_absolute() else root / path for name, path in phase_inputs.items()
    }
    phase164_gate = _read_json(resolved["phase164_gate"])
    literature_rows = _read_csv(resolved["phase163_literature_table"])
    sampler_rows = build_sampler_rows()
    seed_rows = build_seed_rows(sampler_rows)
    metric_rows = build_metric_rows(sampler_rows, seed_rows)
    gate = build_gate(
        phase164_gate=phase164_gate,
        literature_rows=literature_rows,
        metric_rows=metric_rows,
    )

    sampler_path = output_dir / "phase165_sampler_table.csv"
    seed_path = output_dir / "phase165_sampler_seed_metric_table.csv"
    metric_path = output_dir / "phase165_sampler_metric_table.csv"
    gate_path = output_dir / "phase165_adaptive_residual_sampler_gate.json"
    markdown_path = output_dir / "phase165_adaptive_residual_sampler_gate.md"
    manifest_path = output_dir / "phase165_adaptive_residual_sampler_manifest.json"

    _write_csv(sampler_path, sampler_rows, SAMPLER_FIELDS)
    _write_csv(seed_path, seed_rows, SEED_FIELDS)
    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_json(gate_path, gate)
    with markdown_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(build_markdown(gate=gate, metric_rows=metric_rows))

    manifest = {
        "phase": 165,
        "description": "adaptive residual sampler no-training gate",
        "inputs": {name: _display_path(path, root) for name, path in resolved.items()},
        "outputs": {
            "sampler_table": _display_path(sampler_path, root),
            "seed_metric_table": _display_path(seed_path, root),
            "metric_table": _display_path(metric_path, root),
            "gate": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "sampler_rows": len(sampler_rows),
            "seed_metric_rows": len(seed_rows),
            "metric_rows": len(metric_rows),
            "literature_rows": len(literature_rows),
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
