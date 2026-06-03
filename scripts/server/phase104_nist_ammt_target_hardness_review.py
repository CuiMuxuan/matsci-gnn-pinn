#!/usr/bin/env python3
"""Phase 104 NIST AMMT target-hardness review.

Evaluates alternative numeric target summaries from the registered NIST AMMT
table before opening Phase 105 mechanisms. This uses only already-generated
small Phase 104 CSV/JSON artifacts and does not read raw ZIP members or train
Macro PINN models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from gnnpinn.eval.field_baseline import evaluate_table


METHODS = ("mean", "knn", "extra_trees")
DEFAULT_TARGET_COLUMNS = (
    "target_intensity_mean",
    "target_intensity_std",
    "target_intensity_min",
    "target_intensity_max",
    "target_intensity_q90",
)
DEFAULT_TARGET_PRIORITY = (
    "target_intensity_std",
    "target_intensity_q90",
    "target_intensity_min",
    "target_intensity_mean",
    "target_intensity_max",
)
FEATURE_COLUMNS = (
    "x",
    "y",
    "t",
    "source_p_mean",
    "source_p_nonzero_fraction",
    "source_x_range",
    "source_y_range",
    "target_camera_code",
)
METRIC_FIELDS = (
    "target",
    "method",
    "baseline",
    "split",
    "n_points",
    "rmse",
    "mae",
    "relative_l2",
    "normalized_rmse",
    "hot_q90_rmse",
    "gradient_q90_rmse",
)
REVIEW_FIELDS = (
    "target",
    "target_min",
    "target_max",
    "target_range",
    "selected_validation_method",
    "selected_validation_rmse",
    "selected_test_rmse",
    "mean_validation_rmse",
    "mean_test_rmse",
    "validation_relative_improvement_over_mean",
    "selected_test_improves_mean",
    "zero_variance_target",
    "physical_priority",
    "status",
    "phase105_candidate",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.6f}"
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
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


def _region_rmse(split_payload: dict[str, Any], name: str) -> float | None:
    region = (split_payload.get("region_metrics") or {}).get(name) or {}
    metrics = region.get("metrics") or {}
    value = metrics.get("rmse")
    return float(value) if isinstance(value, (int, float)) else None


def _metric_rows(payloads: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target, target_payloads in payloads.items():
        for method, payload in target_payloads.items():
            result = payload["results"][0]
            for split, split_payload in result["split_metrics"].items():
                metrics = split_payload["metrics"]
                rows.append(
                    {
                        "target": target,
                        "method": method,
                        "baseline": result["baseline"],
                        "split": split,
                        "n_points": split_payload["n_points"],
                        "rmse": metrics.get("rmse"),
                        "mae": metrics.get("mae"),
                        "relative_l2": metrics.get("relative_l2"),
                        "normalized_rmse": metrics.get("normalized_rmse"),
                        "hot_q90_rmse": _region_rmse(split_payload, "hot_q90"),
                        "gradient_q90_rmse": _region_rmse(split_payload, "gradient_q90"),
                    }
                )
    return rows


def _target_ranges(field_rows: list[dict[str, str]], target_columns: tuple[str, ...]) -> dict[str, dict[str, float]]:
    ranges: dict[str, dict[str, float]] = {}
    for target in target_columns:
        values = [float(row[target]) for row in field_rows if row.get(target) not in (None, "")]
        if not values:
            raise ValueError(f"No numeric values found for target column {target}")
        ranges[target] = {
            "target_min": min(values),
            "target_max": max(values),
            "target_range": max(values) - min(values),
        }
    return ranges


def _rmse_index(metric_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], float]:
    index: dict[tuple[str, str, str], float] = {}
    for row in metric_rows:
        value = row.get("rmse")
        if isinstance(value, (int, float)):
            index[(str(row["target"]), str(row["method"]), str(row["split"]))] = float(value)
    return index


def _review_rows(
    *,
    metric_rows: list[dict[str, Any]],
    target_ranges: dict[str, dict[str, float]],
    target_columns: tuple[str, ...],
    target_priority: tuple[str, ...],
    min_validation_relative_improvement: float,
) -> list[dict[str, Any]]:
    rmse = _rmse_index(metric_rows)
    priority = {target: index for index, target in enumerate(target_priority)}
    rows: list[dict[str, Any]] = []
    for target in target_columns:
        val_candidates = [
            (rmse[(target, method, "val")], method)
            for method in METHODS
            if (target, method, "val") in rmse
        ]
        if not val_candidates:
            raise ValueError(f"No validation metrics found for {target}")
        selected_val, selected_method = min(val_candidates)
        selected_test = rmse.get((target, selected_method, "test"))
        mean_val = rmse.get((target, "mean", "val"))
        mean_test = rmse.get((target, "mean", "test"))
        if mean_val is None or mean_test is None or selected_test is None:
            raise ValueError(f"Incomplete mean/selected metrics for {target}")
        target_range = target_ranges[target]["target_range"]
        zero_variance = abs(target_range) <= 1e-12
        relative_improvement = (mean_val - selected_val) / mean_val if mean_val > 0 else 0.0
        selected_test_improves_mean = selected_test < mean_test
        if zero_variance:
            status = "blocked_zero_variance_target"
        elif selected_method == "mean":
            status = "blocked_mean_baseline_best"
        elif relative_improvement < min_validation_relative_improvement:
            status = "blocked_weak_validation_gap"
        elif not selected_test_improves_mean:
            status = "blocked_test_reversal_against_mean"
        else:
            status = "candidate_target_ready_for_phase105_design"
        rows.append(
            {
                "target": target,
                **target_ranges[target],
                "selected_validation_method": selected_method,
                "selected_validation_rmse": selected_val,
                "selected_test_rmse": selected_test,
                "mean_validation_rmse": mean_val,
                "mean_test_rmse": mean_test,
                "validation_relative_improvement_over_mean": relative_improvement,
                "selected_test_improves_mean": selected_test_improves_mean,
                "zero_variance_target": zero_variance,
                "physical_priority": priority.get(target, len(priority)),
                "status": status,
                "phase105_candidate": status == "candidate_target_ready_for_phase105_design",
            }
        )
    return rows


def _build_gate(
    *,
    baseline_gate: dict[str, Any],
    review_rows: list[dict[str, Any]],
    target_columns: tuple[str, ...],
    min_validation_relative_improvement: float,
) -> dict[str, Any]:
    candidates = [row for row in review_rows if row["phase105_candidate"]]
    candidates.sort(
        key=lambda row: (
            int(row["physical_priority"]),
            -float(row["validation_relative_improvement_over_mean"]),
        )
    )
    selected = candidates[0] if candidates else None
    baseline_ready = bool(baseline_gate.get("baseline_smoke_completed")) and bool(
        baseline_gate.get("sample_size_sufficient_for_phase105")
    )
    if not baseline_ready:
        status = "phase104_target_hardness_blocked_by_baseline_gate"
        next_action = "complete Phase 104 expanded baseline smoke before target review"
    elif selected:
        status = "phase104_target_hardness_review_ready_phase105_design"
        next_action = (
            "enter Phase 105 low-capacity mechanism design on "
            f"{selected['target']} without opening A100 training"
        )
    else:
        status = "phase104_target_hardness_review_blocked_no_candidate_target"
        next_action = "refine registered targets or add harder leakage-safe target summaries"
    return {
        "status": status,
        "baseline_gate_status": baseline_gate.get("status"),
        "baseline_smoke_completed": bool(baseline_gate.get("baseline_smoke_completed")),
        "sample_size_sufficient_for_phase105": bool(
            baseline_gate.get("sample_size_sufficient_for_phase105")
        ),
        "target_columns": list(target_columns),
        "candidate_targets": [row["target"] for row in candidates],
        "selected_target": selected["target"] if selected else None,
        "selected_validation_method": selected["selected_validation_method"] if selected else None,
        "selected_validation_rmse": selected["selected_validation_rmse"] if selected else None,
        "selected_test_rmse": selected["selected_test_rmse"] if selected else None,
        "min_validation_relative_improvement": min_validation_relative_improvement,
        "phase105_model_mechanism_allowed": bool(selected and baseline_ready),
        "phase105_low_capacity_design_only": bool(selected and baseline_ready),
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
    }


def _write_markdown(path: Path, gate: dict[str, Any], review_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 104 NIST AMMT Target-Hardness Review",
        "",
        f"- Status: `{gate['status']}`",
        f"- Baseline gate status: `{gate['baseline_gate_status']}`",
        f"- Selected target: `{gate['selected_target']}`",
        f"- Phase 105 low-capacity design allowed: `{gate['phase105_model_mechanism_allowed']}`",
        "- A100 training allowed now: `false`",
        "",
        "| Target | Status | Selected method | Val RMSE | Test RMSE | Mean val RMSE | Mean test RMSE | Val improvement vs mean |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in review_rows:
        lines.append(
            "| {target} | {status} | {selected_validation_method} | {selected_validation_rmse} | {selected_test_rmse} | {mean_validation_rmse} | {mean_test_rmse} | {validation_relative_improvement_over_mean} |".format(
                **{key: _csv_value(value) for key, value in row.items()}
            )
        )
    lines.extend(["", f"Next action: {gate['next_action']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_package(
    *,
    root: Path,
    field_table: Path,
    split_manifest: Path,
    baseline_gate_path: Path,
    output_dir: Path,
    target_columns: tuple[str, ...],
    target_priority: tuple[str, ...],
    min_validation_relative_improvement: float,
    n_neighbors: int,
    n_estimators: int,
) -> dict[str, Any]:
    baseline_gate = _read_json(baseline_gate_path)
    field_rows = _read_csv(field_table)
    target_ranges = _target_ranges(field_rows, target_columns)
    payloads: dict[str, dict[str, dict[str, Any]]] = {}
    for target in target_columns:
        payloads[target] = {}
        for method in METHODS:
            payloads[target][method] = {
                "target": target,
                "strategy": method,
                "split_manifest": str(split_manifest),
                "fit_split": "train",
                "results": [
                    evaluate_table(
                        table_path=field_table,
                        target=target,
                        strategy=method,
                        split_manifest_path=split_manifest,
                        fit_split="train",
                        feature_columns=list(FEATURE_COLUMNS) if method != "mean" else None,
                        n_neighbors=n_neighbors,
                        n_estimators=n_estimators,
                        random_state=7,
                        hot_quantiles=[0.9],
                        gradient_quantiles=[0.9],
                    )
                ],
            }
    metric_rows = _metric_rows(payloads)
    review_rows = _review_rows(
        metric_rows=metric_rows,
        target_ranges=target_ranges,
        target_columns=target_columns,
        target_priority=target_priority,
        min_validation_relative_improvement=min_validation_relative_improvement,
    )
    gate = _build_gate(
        baseline_gate=baseline_gate,
        review_rows=review_rows,
        target_columns=target_columns,
        min_validation_relative_improvement=min_validation_relative_improvement,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "phase104_nist_ammt_target_hardness_metric_table.csv"
    review_path = output_dir / "phase104_nist_ammt_target_hardness_review_table.csv"
    gate_path = output_dir / "phase104_nist_ammt_target_hardness_review_gate.json"
    markdown_path = output_dir / "phase104_nist_ammt_target_hardness_review_summary.md"
    manifest_path = output_dir / "phase104_nist_ammt_target_hardness_review_manifest.json"
    payload_path = output_dir / "phase104_nist_ammt_target_hardness_payloads.json"
    _write_csv(metrics_path, metric_rows, METRIC_FIELDS)
    _write_csv(review_path, review_rows, REVIEW_FIELDS)
    _write_json(gate_path, gate)
    _write_json(payload_path, payloads)
    _write_markdown(markdown_path, gate, review_rows)
    manifest = {
        "phase": 104,
        "objective": "nist_ammt_registered_target_hardness_review_no_training",
        "inputs": {
            "field_table": _display_path(field_table, root),
            "split_manifest": _display_path(split_manifest, root),
            "baseline_gate": _display_path(baseline_gate_path, root),
        },
        "outputs": {
            "metric_table": _display_path(metrics_path, root),
            "review_table": _display_path(review_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown_summary": _display_path(markdown_path, root),
            "payloads": _display_path(payload_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "target_columns": len(target_columns),
            "metric_rows": len(metric_rows),
            "candidate_targets": len(gate["candidate_targets"]),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _split_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--field-table",
        type=Path,
        default=Path(
            "docs/results/phase104_nist_ammt_baseline_smoke/"
            "phase104_nist_ammt_tiny_numeric_field_table.csv"
        ),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path(
            "docs/results/phase104_nist_ammt_baseline_smoke/"
            "phase104_nist_ammt_tiny_numeric_split_manifest.json"
        ),
    )
    parser.add_argument(
        "--baseline-gate",
        type=Path,
        default=Path(
            "docs/results/phase104_nist_ammt_baseline_smoke/"
            "phase104_nist_ammt_baseline_smoke_gate.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase104_nist_ammt_target_hardness_review"),
    )
    parser.add_argument("--target-columns", default=",".join(DEFAULT_TARGET_COLUMNS))
    parser.add_argument("--target-priority", default=",".join(DEFAULT_TARGET_PRIORITY))
    parser.add_argument("--min-validation-relative-improvement", type=float, default=0.05)
    parser.add_argument("--n-neighbors", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    field_table = args.field_table if args.field_table.is_absolute() else root / args.field_table
    split_manifest = (
        args.split_manifest if args.split_manifest.is_absolute() else root / args.split_manifest
    )
    baseline_gate = args.baseline_gate if args.baseline_gate.is_absolute() else root / args.baseline_gate
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(
        root=root,
        field_table=field_table,
        split_manifest=split_manifest,
        baseline_gate_path=baseline_gate,
        output_dir=output_dir,
        target_columns=_split_csv_arg(args.target_columns),
        target_priority=_split_csv_arg(args.target_priority),
        min_validation_relative_improvement=args.min_validation_relative_improvement,
        n_neighbors=args.n_neighbors,
        n_estimators=args.n_estimators,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
