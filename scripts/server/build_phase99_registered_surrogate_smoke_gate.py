#!/usr/bin/env python3
"""Build the Phase 99 local registered-surrogate baseline-first smoke gate.

Phase 99 runs a stricter local smoke on the Phase 98 generated registered
surrogate. It can support local mechanism/appendix evidence or a future
registered-target search, but it never opens AM-Bench A100 training.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


METRIC_FIELDS = (
    "row_id",
    "method_id",
    "method_family",
    "train_points",
    "budget_class",
    "gate_comparator",
    "validation_rmse",
    "validation_pde_residual_rmse",
    "validation_hot_q90_rmse",
    "validation_gradient_q90_rmse",
    "test_rmse",
    "test_pde_residual_rmse",
    "test_hot_q90_rmse",
    "test_gradient_q90_rmse",
    "global_delta_vs_vanilla",
    "residual_delta_vs_vanilla",
    "hot_delta_vs_vanilla",
    "gradient_delta_vs_vanilla",
    "notes",
)

COMPARISON_FIELDS = (
    "comparison_id",
    "candidate_method",
    "baseline_method",
    "metric",
    "candidate_value",
    "baseline_value",
    "delta_baseline_minus_candidate",
    "pass",
    "scope",
)


def _phase96_module():
    script = Path(__file__).with_name("build_phase96_pfhub_local_smoke_gate.py")
    spec = importlib.util.spec_from_file_location("phase96_smoke", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Phase 96 smoke module from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        return f"{value:.6f}"
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
    phase98 = root / "docs/results/phase98_registered_target_unlock_gate"
    return {
        "phase98_gate": phase98 / "phase98_registered_target_unlock_gate.json",
        "phase98_surrogate_card": phase98 / "phase98_registered_surrogate_data_card.json",
    }


def _make_grid_from_card(module: Any, card: dict[str, Any], split: str):
    grid_spec = card["split_plan"][split]
    return module.make_grid(int(grid_spec["nx"]), int(grid_spec["nt"]))


def _fit_predict(
    module: Any,
    feature_fn: Any,
    train_grid: Any,
    train_target: np.ndarray,
    eval_grid: Any,
    selected: np.ndarray | None = None,
) -> np.ndarray:
    if selected is None:
        selected = np.arange(len(train_grid.x))
    weights = module.fit_ridge(
        feature_fn(train_grid.x[selected], train_grid.t[selected]),
        train_target[selected],
        1.0e-8,
    )
    return module.predict(feature_fn, weights, eval_grid)


def _score_row(
    module: Any,
    *,
    method_id: str,
    method_family: str,
    budget_class: str,
    gate_comparator: bool,
    feature_fn: Any,
    train_grid: Any,
    train_target: np.ndarray,
    validation_grid: Any,
    validation_target: np.ndarray,
    test_grid: Any,
    test_target: np.ndarray,
    selected: np.ndarray | None = None,
    notes: str,
) -> dict[str, Any]:
    if selected is None:
        selected = np.arange(len(train_grid.x))
    validation_prediction = _fit_predict(
        module, feature_fn, train_grid, train_target, validation_grid, selected
    )
    test_prediction = _fit_predict(module, feature_fn, train_grid, train_target, test_grid, selected)
    validation = module.metric_bundle(validation_target, validation_prediction, validation_grid)
    test = module.metric_bundle(test_target, test_prediction, test_grid)
    return {
        "method_id": method_id,
        "method_family": method_family,
        "train_points": int(len(selected)),
        "budget_class": budget_class,
        "gate_comparator": gate_comparator,
        "validation_rmse": validation["rmse"],
        "validation_pde_residual_rmse": validation["pde_residual_rmse"],
        "validation_hot_q90_rmse": validation["hot_q90_rmse"],
        "validation_gradient_q90_rmse": validation["gradient_q90_rmse"],
        "test_rmse": test["rmse"],
        "test_pde_residual_rmse": test["pde_residual_rmse"],
        "test_hot_q90_rmse": test["hot_q90_rmse"],
        "test_gradient_q90_rmse": test["gradient_q90_rmse"],
        "notes": notes,
    }


def run_registered_surrogate_smoke(card: dict[str, Any]) -> list[dict[str, Any]]:
    module = _phase96_module()
    train_grid = _make_grid_from_card(module, card, "train_grid")
    validation_grid = _make_grid_from_card(module, card, "validation_grid")
    test_grid = _make_grid_from_card(module, card, "test_grid")
    train_target = module.target_solution(train_grid.x, train_grid.t)
    validation_target = module.target_solution(validation_grid.x, validation_grid.t)
    test_target = module.target_solution(test_grid.x, test_grid.t)

    random_selected = module.random_indices(train_grid, module.ADAPTIVE_BUDGET)
    source_quota_selected = module.adaptive_indices(train_grid, module.ADAPTIVE_BUDGET)

    rows = [
        _score_row(
            module,
            method_id="low_order_interpolation_full",
            method_family="baseline",
            budget_class="full_grid",
            gate_comparator=True,
            feature_fn=module.low_order_features,
            train_grid=train_grid,
            train_target=train_target,
            validation_grid=validation_grid,
            validation_target=validation_target,
            test_grid=test_grid,
            test_target=test_target,
            notes="simple low-order registered-surrogate baseline",
        ),
        _score_row(
            module,
            method_id="vanilla_deterministic_surrogate_full",
            method_family="baseline",
            budget_class="full_grid",
            gate_comparator=True,
            feature_fn=module.vanilla_features,
            train_grid=train_grid,
            train_target=train_target,
            validation_grid=validation_grid,
            validation_target=validation_target,
            test_grid=test_grid,
            test_target=test_target,
            notes="full-grid deterministic surrogate comparator",
        ),
        _score_row(
            module,
            method_id="fixed_green_function_features_full",
            method_family="candidate",
            budget_class="full_grid",
            gate_comparator=False,
            feature_fn=module.green_features,
            train_grid=train_grid,
            train_target=train_target,
            validation_grid=validation_grid,
            validation_target=validation_target,
            test_grid=test_grid,
            test_target=test_target,
            notes="candidate fixed registered Green's-function / heat-kernel feature basis",
        ),
        _score_row(
            module,
            method_id="random_collocation_same_budget",
            method_family="baseline",
            budget_class="same_budget",
            gate_comparator=True,
            feature_fn=module.green_features,
            train_grid=train_grid,
            train_target=train_target,
            validation_grid=validation_grid,
            validation_target=validation_target,
            test_grid=test_grid,
            test_target=test_target,
            selected=random_selected,
            notes="same-budget random collocation comparator",
        ),
        _score_row(
            module,
            method_id="source_quota_green_same_budget",
            method_family="diagnostic_candidate",
            budget_class="same_budget",
            gate_comparator=False,
            feature_fn=module.green_features,
            train_grid=train_grid,
            train_target=train_target,
            validation_grid=validation_grid,
            validation_target=validation_target,
            test_grid=test_grid,
            test_target=test_target,
            selected=source_quota_selected,
            notes="registered-source quota diagnostic under the same collocation budget",
        ),
    ]
    vanilla = next(row for row in rows if row["method_id"] == "vanilla_deterministic_surrogate_full")
    for index, row in enumerate(rows, start=1):
        row["row_id"] = f"P99-MET-{index:03d}"
        row["global_delta_vs_vanilla"] = vanilla["test_rmse"] - row["test_rmse"]
        row["residual_delta_vs_vanilla"] = (
            vanilla["test_pde_residual_rmse"] - row["test_pde_residual_rmse"]
        )
        row["hot_delta_vs_vanilla"] = vanilla["test_hot_q90_rmse"] - row["test_hot_q90_rmse"]
        row["gradient_delta_vs_vanilla"] = (
            vanilla["test_gradient_q90_rmse"] - row["test_gradient_q90_rmse"]
        )
    return rows


def build_comparison_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate = next(row for row in metric_rows if row["method_id"] == "fixed_green_function_features_full")
    baseline_ids = [
        "low_order_interpolation_full",
        "vanilla_deterministic_surrogate_full",
        "random_collocation_same_budget",
    ]
    metrics = [
        ("test_rmse", "global_rmse"),
        ("test_pde_residual_rmse", "pde_residual_rmse"),
        ("test_hot_q90_rmse", "hot_q90_rmse"),
        ("test_gradient_q90_rmse", "gradient_q90_rmse"),
    ]
    rows: list[dict[str, Any]] = []
    for baseline_id in baseline_ids:
        baseline = next(row for row in metric_rows if row["method_id"] == baseline_id)
        for metric_key, metric_name in metrics:
            delta = baseline[metric_key] - candidate[metric_key]
            rows.append(
                {
                    "comparison_id": f"P99-COMP-{len(rows) + 1:03d}",
                    "candidate_method": candidate["method_id"],
                    "baseline_method": baseline_id,
                    "metric": metric_name,
                    "candidate_value": candidate[metric_key],
                    "baseline_value": baseline[metric_key],
                    "delta_baseline_minus_candidate": delta,
                    "pass": delta >= 0.0,
                    "scope": "full-grid baselines" if baseline_id.endswith("_full") else "same-budget boundary audit",
                }
            )
    return rows


def build_gate(
    *,
    phase98_gate: dict[str, Any],
    card: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase98_allows = bool(phase98_gate.get("phase99_local_smoke_allowed"))
    if not phase98_allows:
        return {
            "status": "blocked_by_phase98_unlock_gate",
            "source_phase98_status": phase98_gate.get("status"),
            "source_candidate": card.get("candidate_id"),
            "phase100_local_mechanism_package_allowed": False,
            "am_bench_transfer_unlocked": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
            "submission_ready": False,
            "metric_rows": len(metric_rows),
            "comparison_rows": len(comparison_rows),
            "full_grid_baseline_pass": False,
            "same_budget_boundary_failures": 0,
            "source_quota_hot_gain_vs_random": None,
            "next_action": "repair Phase 98 registered-surrogate data card before local smoke",
            "required_before_a100_training": [
                "public registered AM-Bench/external target",
                "baseline-first local smoke on that registered target",
                "non-worse global/hot/gradient metrics",
                "server validation from a pushed commit",
            ],
        }
    full_grid_comparisons = [
        row for row in comparison_rows if row["scope"] == "full-grid baselines"
    ]
    boundary_comparisons = [
        row for row in comparison_rows if row["scope"] == "same-budget boundary audit"
    ]
    full_grid_pass = bool(full_grid_comparisons) and all(row["pass"] for row in full_grid_comparisons)
    boundary_failures = [row for row in boundary_comparisons if not row["pass"]]
    source_quota = next(row for row in metric_rows if row["method_id"] == "source_quota_green_same_budget")
    random_budget = next(row for row in metric_rows if row["method_id"] == "random_collocation_same_budget")
    source_quota_hot_gain = random_budget["test_hot_q90_rmse"] - source_quota["test_hot_q90_rmse"]

    if full_grid_pass and boundary_failures:
        status = "local_surrogate_positive_with_focused_baseline_boundary"
        next_action = (
            "package as local mechanism evidence and keep AM-Bench transfer blocked until registered data exists"
        )
        local_mechanism_package_allowed = True
    elif full_grid_pass:
        status = "local_surrogate_positive_no_a100"
        next_action = "package as local mechanism evidence; do not start A100 training"
        local_mechanism_package_allowed = True
    else:
        status = "closed_local_surrogate_negative"
        next_action = "close fixed-kernel surrogate branch as diagnostic"
        local_mechanism_package_allowed = False

    return {
        "status": status,
        "source_phase98_status": phase98_gate.get("status"),
        "source_candidate": card.get("candidate_id"),
        "phase100_local_mechanism_package_allowed": local_mechanism_package_allowed,
        "am_bench_transfer_unlocked": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "submission_ready": False,
        "metric_rows": len(metric_rows),
        "comparison_rows": len(comparison_rows),
        "full_grid_baseline_pass": full_grid_pass,
        "same_budget_boundary_failures": len(boundary_failures),
        "source_quota_hot_gain_vs_random": source_quota_hot_gain,
        "next_action": next_action,
        "required_before_a100_training": [
            "public registered AM-Bench/external target",
            "baseline-first local smoke on that registered target",
            "non-worse global/hot/gradient metrics",
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
    comparison_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Phase 99 Local Registered-Surrogate Baseline-First Smoke Gate",
            "",
            "## Gate Decision",
            "",
            f"Status: `{gate['status']}`.",
            f"Local mechanism package allowed: `{str(gate['phase100_local_mechanism_package_allowed']).lower()}`.",
            f"AM-Bench transfer unlocked: `{str(gate['am_bench_transfer_unlocked']).lower()}`.",
            f"A100 training allowed now: `{str(gate['a100_training_allowed_now']).lower()}`.",
            f"A100-SXM4-80GB request now: `{str(gate['a100_80gb_request_now']).lower()}`.",
            "",
            "Phase 99 is a local registered-surrogate smoke gate. It does not provide AM-Bench transfer evidence.",
            "",
            "## Metrics",
            "",
            _markdown_table(
                metric_rows,
                [
                    ("method_id", "Method"),
                    ("budget_class", "Budget"),
                    ("test_rmse", "Test RMSE"),
                    ("test_pde_residual_rmse", "Residual"),
                    ("test_hot_q90_rmse", "Hot q90"),
                    ("test_gradient_q90_rmse", "Gradient q90"),
                ],
            ),
            "",
            "## Comparison Audit",
            "",
            _markdown_table(
                comparison_rows,
                [
                    ("baseline_method", "Baseline"),
                    ("metric", "Metric"),
                    ("delta_baseline_minus_candidate", "Delta"),
                    ("pass", "Pass"),
                    ("scope", "Scope"),
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

    phase98_gate = _read_json(resolved["phase98_gate"])
    card = _read_json(resolved["phase98_surrogate_card"])
    if phase98_gate.get("phase99_local_smoke_allowed"):
        metric_rows = run_registered_surrogate_smoke(card)
        comparison_rows = build_comparison_rows(metric_rows)
    else:
        metric_rows = []
        comparison_rows = []
    gate = build_gate(
        phase98_gate=phase98_gate,
        card=card,
        metric_rows=metric_rows,
        comparison_rows=comparison_rows,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_path = output_dir / "phase99_registered_surrogate_metric_table.csv"
    comparison_path = output_dir / "phase99_registered_surrogate_comparison_table.csv"
    gate_path = output_dir / "phase99_registered_surrogate_smoke_gate.json"
    markdown_path = output_dir / "phase99_registered_surrogate_smoke_gate.md"
    manifest_path = output_dir / "phase99_registered_surrogate_smoke_gate_manifest.json"

    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_csv(comparison_path, comparison_rows, COMPARISON_FIELDS)
    _write_json(gate_path, gate)
    markdown_path.write_text(build_markdown(gate, metric_rows, comparison_rows), encoding="utf-8")

    manifest = {
        "phase": 99,
        "objective": "local_registered_surrogate_baseline_first_smoke",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "metric_table": _display_path(metric_path, root),
            "comparison_table": _display_path(comparison_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "metric_rows": len(metric_rows),
            "comparison_rows": len(comparison_rows),
            "same_budget_boundary_failures": gate["same_budget_boundary_failures"],
        },
        "gate": gate,
        "phase98_gate": {
            "status": phase98_gate.get("status"),
            "phase99_local_smoke_allowed": phase98_gate.get("phase99_local_smoke_allowed"),
            "am_bench_transfer_unlocked": phase98_gate.get("am_bench_transfer_unlocked"),
        },
        "surrogate_card": {
            "candidate_id": card.get("candidate_id"),
            "source_target_id": card.get("source_target_id"),
            "dataset_family": card.get("dataset_family"),
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
        default=Path("docs/results/phase99_registered_surrogate_smoke_gate"),
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
