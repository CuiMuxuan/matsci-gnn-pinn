#!/usr/bin/env python3
"""Phase 104 NIST AMMT tiny baseline-first smoke.

Runs dependency-light mean, kNN, and ExtraTrees baselines on the Phase 104 tiny
numeric field table. This remains no-mechanism and no-training for Macro PINN.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

from gnnpinn.eval.field_baseline import evaluate_table


METHODS = ("mean", "knn", "extra_trees")
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


def _metric_rows(payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, payload in payloads.items():
        result = payload["results"][0]
        for split, split_payload in result["split_metrics"].items():
            metrics = split_payload["metrics"]
            rows.append(
                {
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


def _build_gate(
    *,
    numeric_gate: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    min_rows_for_mechanism: int,
) -> dict[str, Any]:
    row_count = int(numeric_gate.get("row_count") or 0)
    methods = sorted({row["method"] for row in metric_rows})
    completed = set(methods) == set(METHODS)
    sample_sufficient = row_count >= min_rows_for_mechanism
    best_val = min(
        (
            float(row["rmse"])
            for row in metric_rows
            if row["split"] == "val" and isinstance(row.get("rmse"), (int, float))
        ),
        default=None,
    )
    best_test = min(
        (
            float(row["rmse"])
            for row in metric_rows
            if row["split"] == "test" and isinstance(row.get("rmse"), (int, float))
        ),
        default=None,
    )
    if not numeric_gate.get("numeric_field_table_ready"):
        status = "phase104_blocked_numeric_field_table_missing"
        next_action = "build the Phase 104 numeric field table first"
    elif not completed:
        status = "phase104_baseline_smoke_incomplete"
        next_action = "rerun mean/kNN/ExtraTrees baseline smoke"
    elif not sample_sufficient:
        status = "phase104_baseline_smoke_boundary_tiny_sample_mechanisms_locked"
        next_action = "expand the registered numeric table before Phase 105 mechanism testing"
    else:
        status = "phase104_baseline_smoke_complete_mechanisms_review_required"
        next_action = "review validation/test baseline gap before opening Phase 105"
    return {
        "status": status,
        "baseline_smoke_completed": completed,
        "methods": methods,
        "target": "target_intensity_mean",
        "row_count": row_count,
        "sample_size_sufficient_for_phase105": sample_sufficient,
        "min_rows_for_mechanism": min_rows_for_mechanism,
        "best_validation_rmse": best_val,
        "best_test_rmse": best_test,
        "phase105_model_mechanism_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
    }


def _write_markdown(path: Path, gate: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 104 NIST AMMT Baseline-First Tiny Smoke",
        "",
        f"- Status: `{gate['status']}`",
        f"- Baseline smoke completed: `{gate['baseline_smoke_completed']}`",
        f"- Row count: `{gate['row_count']}`",
        f"- Sample size sufficient for Phase 105: `{gate['sample_size_sufficient_for_phase105']}`",
        "- Phase 105 model mechanisms allowed: `false`",
        "- A100 training allowed now: `false`",
        "",
        "| Method | Split | N | RMSE | Hot q90 RMSE | Gradient q90 RMSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {split} | {n_points} | {rmse} | {hot_q90_rmse} | {gradient_q90_rmse} |".format(
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
    numeric_gate_path: Path,
    output_dir: Path,
    target: str,
    min_rows_for_mechanism: int,
    n_neighbors: int,
    n_estimators: int,
) -> dict[str, Any]:
    numeric_gate = _read_json(numeric_gate_path)
    payloads: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        payloads[method] = {
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
    gate = _build_gate(
        numeric_gate=numeric_gate,
        metric_rows=metric_rows,
        min_rows_for_mechanism=min_rows_for_mechanism,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "phase104_nist_ammt_baseline_metric_table.csv"
    gate_path = output_dir / "phase104_nist_ammt_baseline_smoke_gate.json"
    markdown_path = output_dir / "phase104_nist_ammt_baseline_smoke_summary.md"
    manifest_path = output_dir / "phase104_nist_ammt_baseline_smoke_manifest.json"
    payload_path = output_dir / "phase104_nist_ammt_baseline_payloads.json"
    _write_csv(metrics_path, metric_rows, METRIC_FIELDS)
    _write_json(gate_path, gate)
    _write_json(payload_path, payloads)
    _write_markdown(markdown_path, gate, metric_rows)
    manifest = {
        "phase": 104,
        "objective": "nist_ammt_tiny_baseline_first_smoke_no_mechanism",
        "inputs": {
            "field_table": _display_path(field_table, root),
            "split_manifest": _display_path(split_manifest, root),
            "numeric_gate": _display_path(numeric_gate_path, root),
        },
        "outputs": {
            "metric_table": _display_path(metrics_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown_summary": _display_path(markdown_path, root),
            "payloads": _display_path(payload_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {"metric_rows": len(metric_rows)},
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


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
        "--numeric-gate",
        type=Path,
        default=Path(
            "docs/results/phase104_nist_ammt_baseline_smoke/"
            "phase104_nist_ammt_tiny_numeric_field_gate.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase104_nist_ammt_baseline_smoke"),
    )
    parser.add_argument("--target", default="target_intensity_mean")
    parser.add_argument("--min-rows-for-mechanism", type=int, default=100)
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
    numeric_gate = args.numeric_gate if args.numeric_gate.is_absolute() else root / args.numeric_gate
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(
        root=root,
        field_table=field_table,
        split_manifest=split_manifest,
        numeric_gate_path=numeric_gate,
        output_dir=output_dir,
        target=args.target,
        min_rows_for_mechanism=args.min_rows_for_mechanism,
        n_neighbors=args.n_neighbors,
        n_estimators=args.n_estimators,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
