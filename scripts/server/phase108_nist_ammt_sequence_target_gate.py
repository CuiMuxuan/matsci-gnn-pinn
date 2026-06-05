#!/usr/bin/env python3
"""Phase 108 NIST AMMT sequence target representation gate.

Builds leakage-audited sequence/camera-pair target summaries from the Phase 106
spatial table, then runs strong baselines before any training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from gnnpinn.eval.field_baseline import evaluate_table


METHODS = ("mean", "knn", "extra_trees", "hist_gradient_boosting")
BASE_FEATURES = (
    "x",
    "y",
    "t",
    "source_p_mean",
    "source_p_nonzero_fraction",
    "source_x_range",
    "source_y_range",
    "target_camera_code",
)
BASE_TARGET = "target_center_periphery_contrast"
SEQUENCE_TARGET_COLUMNS = (
    "target_cp_layer_mean",
    "target_cp_camera_pair_delta",
    "target_cp_abs_camera_pair_delta",
    "target_cp_layer_camera_range",
    "target_cp_deviation_from_layer_mean",
    "target_cp_prev_same_camera_delta",
    "target_cp_prev_layer_mean_delta",
    "target_cp_prev_same_camera_abs_delta",
    "target_cp_prev_layer_mean_abs_delta",
)
TARGET_PRIORITY = (
    "target_cp_camera_pair_delta",
    "target_cp_prev_same_camera_delta",
    "target_cp_prev_layer_mean_delta",
    "target_cp_deviation_from_layer_mean",
    "target_cp_layer_camera_range",
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
    "selected_validation_normalized_rmse",
    "selected_test_rmse",
    "selected_test_normalized_rmse",
    "mean_validation_rmse",
    "mean_validation_normalized_rmse",
    "mean_test_rmse",
    "mean_test_normalized_rmse",
    "validation_relative_improvement_over_mean",
    "test_relative_improvement_over_mean",
    "baseline_visible_gap",
    "strong_baseline_solved",
    "zero_variance_target",
    "physical_priority",
    "status",
    "phase108_candidate",
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


def _layer(row: dict[str, str]) -> int:
    return int(float(row["source_layer_index"]))


def _camera(row: dict[str, str]) -> str:
    return str(row.get("target_camera_code") or "")


def _target_value(row: dict[str, str], target: str) -> float:
    return float(row[target])


def build_sequence_rows(rows: list[dict[str, str]], *, base_target: str = BASE_TARGET) -> list[dict[str, Any]]:
    by_layer: dict[int, list[dict[str, str]]] = defaultdict(list)
    by_camera: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_layer[_layer(row)].append(row)
        by_camera[_camera(row)].append(row)
    layer_stats: dict[int, dict[str, float]] = {}
    for layer, layer_rows in by_layer.items():
        values = [_target_value(row, base_target) for row in layer_rows]
        layer_stats[layer] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
        }
    prev_by_camera: dict[tuple[str, int], float] = {}
    for camera, camera_rows in by_camera.items():
        ordered = sorted(camera_rows, key=_layer)
        previous: float | None = None
        for row in ordered:
            current = _target_value(row, base_target)
            if previous is not None:
                prev_by_camera[(camera, _layer(row))] = current - previous
            previous = current
    rows_out: list[dict[str, Any]] = []
    for row in rows:
        layer = _layer(row)
        camera = _camera(row)
        value = _target_value(row, base_target)
        stats = layer_stats[layer]
        pair_delta = value - (sum(_target_value(item, base_target) for item in by_layer[layer] if _camera(item) != camera) / max(1, sum(1 for item in by_layer[layer] if _camera(item) != camera)))
        prev_same = prev_by_camera.get((camera, layer), 0.0)
        prev_layer_stats = layer_stats.get(layer - 1)
        prev_layer_delta = value - prev_layer_stats["mean"] if prev_layer_stats else 0.0
        rows_out.append(
            {
                **row,
                "target_cp_layer_mean": stats["mean"],
                "target_cp_camera_pair_delta": pair_delta,
                "target_cp_abs_camera_pair_delta": abs(pair_delta),
                "target_cp_layer_camera_range": stats["range"],
                "target_cp_deviation_from_layer_mean": value - stats["mean"],
                "target_cp_prev_same_camera_delta": prev_same,
                "target_cp_prev_layer_mean_delta": prev_layer_delta,
                "target_cp_prev_same_camera_abs_delta": abs(prev_same),
                "target_cp_prev_layer_mean_abs_delta": abs(prev_layer_delta),
            }
        )
    return rows_out


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


def _metric_index(metric_rows: list[dict[str, Any]], metric: str) -> dict[tuple[str, str, str], float]:
    index: dict[tuple[str, str, str], float] = {}
    for row in metric_rows:
        value = row.get(metric)
        if isinstance(value, (int, float)):
            index[(str(row["target"]), str(row["method"]), str(row["split"]))] = float(value)
    return index


def _target_ranges(field_rows: list[dict[str, Any]], target_columns: tuple[str, ...]) -> dict[str, dict[str, float]]:
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


def _metric_float(index: dict[tuple[str, str, str], float], target: str, method: str, split: str) -> float:
    key = (target, method, split)
    if key not in index:
        raise ValueError(f"Missing metric for {target}/{method}/{split}")
    return index[key]


def _review_rows(
    *,
    metric_rows: list[dict[str, Any]],
    target_ranges: dict[str, dict[str, float]],
    target_columns: tuple[str, ...],
    target_priority: tuple[str, ...],
    min_validation_relative_improvement: float,
    min_unsolved_validation_normalized_rmse: float,
) -> list[dict[str, Any]]:
    rmse = _metric_index(metric_rows, "rmse")
    normalized = _metric_index(metric_rows, "normalized_rmse")
    priority = {target: index for index, target in enumerate(target_priority)}
    rows: list[dict[str, Any]] = []
    for target in target_columns:
        val_candidates = [
            (_metric_float(rmse, target, method, "val"), method)
            for method in METHODS
            if (target, method, "val") in rmse
        ]
        if not val_candidates:
            raise ValueError(f"No validation metrics found for {target}")
        selected_val, selected_method = min(val_candidates)
        selected_test = _metric_float(rmse, target, selected_method, "test")
        mean_val = _metric_float(rmse, target, "mean", "val")
        mean_test = _metric_float(rmse, target, "mean", "test")
        selected_val_norm = normalized.get((target, selected_method, "val"))
        selected_test_norm = normalized.get((target, selected_method, "test"))
        mean_val_norm = normalized.get((target, "mean", "val"))
        mean_test_norm = normalized.get((target, "mean", "test"))
        target_range = target_ranges[target]["target_range"]
        zero_variance = abs(target_range) <= 1e-12
        val_relative = (mean_val - selected_val) / mean_val if mean_val > 0 else 0.0
        test_relative = (mean_test - selected_test) / mean_test if mean_test > 0 else 0.0
        visible_gap = selected_method != "mean" and val_relative >= min_validation_relative_improvement
        solved = (
            isinstance(selected_val_norm, float)
            and selected_val_norm < min_unsolved_validation_normalized_rmse
        )
        if zero_variance:
            status = "blocked_zero_variance_target"
        elif not visible_gap:
            status = "blocked_no_baseline_visible_gap"
        elif solved:
            status = "blocked_strong_baseline_solved_validation_target"
        else:
            status = "candidate_sequence_target_gap_ready_for_review"
        rows.append(
            {
                "target": target,
                **target_ranges[target],
                "selected_validation_method": selected_method,
                "selected_validation_rmse": selected_val,
                "selected_validation_normalized_rmse": selected_val_norm,
                "selected_test_rmse": selected_test,
                "selected_test_normalized_rmse": selected_test_norm,
                "mean_validation_rmse": mean_val,
                "mean_validation_normalized_rmse": mean_val_norm,
                "mean_test_rmse": mean_test,
                "mean_test_normalized_rmse": mean_test_norm,
                "validation_relative_improvement_over_mean": val_relative,
                "test_relative_improvement_over_mean": test_relative,
                "baseline_visible_gap": visible_gap,
                "strong_baseline_solved": solved,
                "zero_variance_target": zero_variance,
                "physical_priority": priority.get(target, len(priority)),
                "status": status,
                "phase108_candidate": status == "candidate_sequence_target_gap_ready_for_review",
            }
        )
    return rows


def _build_gate(
    *,
    phase106_gate: dict[str, Any],
    phase107_gate: dict[str, Any],
    review_rows: list[dict[str, Any]],
    min_validation_relative_improvement: float,
    min_unsolved_validation_normalized_rmse: float,
) -> dict[str, Any]:
    candidates = [row for row in review_rows if row["phase108_candidate"]]
    candidates.sort(
        key=lambda row: (
            int(row["physical_priority"]),
            -float(row["selected_validation_normalized_rmse"] or 0.0),
            float(row["selected_validation_rmse"]),
        )
    )
    selected = candidates[0] if candidates else None
    phase106_ready = (
        phase106_gate.get("status") == "phase106_spatial_target_gap_ready_focused_no_training_validation"
    )
    phase107_closed = (
        phase107_gate.get("status") == "phase107_source_region_feature_gate_blocked_no_phase106_gain"
    )
    if not phase106_ready:
        status = "phase108_sequence_target_gate_blocked_by_phase106"
        next_action = "complete Phase 106 spatial target gate first"
    elif not phase107_closed:
        status = "phase108_sequence_target_gate_blocked_by_phase107"
        next_action = "close Phase 107 source-region feature gate first"
    elif selected:
        status = "phase108_sequence_target_gap_ready_focused_review"
        next_action = (
            f"review {selected['target']} sequence target representation before any model training"
        )
    else:
        status = "phase108_sequence_target_gate_closed_no_baseline_gap"
        next_action = "close sequence target representation as diagnostic; do not train"
    return {
        "status": status,
        "base_target": BASE_TARGET,
        "phase106_gate_status": phase106_gate.get("status"),
        "phase107_gate_status": phase107_gate.get("status"),
        "target_columns": list(SEQUENCE_TARGET_COLUMNS),
        "candidate_targets": [row["target"] for row in candidates],
        "selected_target": selected["target"] if selected else None,
        "selected_validation_method": selected["selected_validation_method"] if selected else None,
        "selected_validation_rmse": selected["selected_validation_rmse"] if selected else None,
        "selected_validation_normalized_rmse": selected["selected_validation_normalized_rmse"] if selected else None,
        "selected_test_rmse": selected["selected_test_rmse"] if selected else None,
        "selected_test_normalized_rmse": selected["selected_test_normalized_rmse"] if selected else None,
        "min_validation_relative_improvement": min_validation_relative_improvement,
        "min_unsolved_validation_normalized_rmse": min_unsolved_validation_normalized_rmse,
        "phase108_focused_review_allowed": bool(selected and phase106_ready and phase107_closed),
        "phase108_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
    }


def _write_markdown(path: Path, gate: dict[str, Any], review_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 108 NIST AMMT Sequence Target Gate",
        "",
        f"- Status: `{gate['status']}`",
        f"- Base target: `{gate['base_target']}`",
        f"- Selected target: `{gate['selected_target']}`",
        f"- Focused review allowed: `{gate['phase108_focused_review_allowed']}`",
        "- Model training allowed: `false`",
        "- A100 training allowed now: `false`",
        "",
        "| Target | Status | Method | Val RMSE | Val NRMSE | Mean val RMSE | Val gain vs mean |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in review_rows:
        lines.append(
            "| {target} | {status} | {selected_validation_method} | {selected_validation_rmse} | {selected_validation_normalized_rmse} | {mean_validation_rmse} | {validation_relative_improvement_over_mean} |".format(
                **{key: _csv_value(value) for key, value in row.items()}
            )
        )
    lines.extend(["", f"Next action: {gate['next_action']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_package(
    *,
    root: Path,
    spatial_field_table: Path,
    split_manifest: Path,
    phase106_gate_path: Path,
    phase107_gate_path: Path,
    output_dir: Path,
    target_columns: tuple[str, ...],
    target_priority: tuple[str, ...],
    min_validation_relative_improvement: float,
    min_unsolved_validation_normalized_rmse: float,
    n_neighbors: int,
    n_estimators: int,
) -> dict[str, Any]:
    phase106_gate = _read_json(phase106_gate_path)
    phase107_gate = _read_json(phase107_gate_path)
    rows = build_sequence_rows(_read_csv(spatial_field_table), base_target=BASE_TARGET)
    target_ranges = _target_ranges(rows, target_columns)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "phase108_nist_ammt_sequence_target_field_table.csv"
    _write_csv(table_path, rows, tuple(rows[0].keys()))
    payloads: dict[str, dict[str, dict[str, Any]]] = {}
    for target in target_columns:
        payloads[target] = {}
        for method in METHODS:
            payloads[target][method] = {
                "target": target,
                "strategy": method,
                "split_manifest": str(split_manifest),
                "fit_split": "train",
                "feature_columns": list(BASE_FEATURES) if method != "mean" else None,
                "results": [
                    evaluate_table(
                        table_path=table_path,
                        target=target,
                        strategy=method,
                        split_manifest_path=split_manifest,
                        fit_split="train",
                        feature_columns=list(BASE_FEATURES) if method != "mean" else None,
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
        min_unsolved_validation_normalized_rmse=min_unsolved_validation_normalized_rmse,
    )
    gate = _build_gate(
        phase106_gate=phase106_gate,
        phase107_gate=phase107_gate,
        review_rows=review_rows,
        min_validation_relative_improvement=min_validation_relative_improvement,
        min_unsolved_validation_normalized_rmse=min_unsolved_validation_normalized_rmse,
    )

    metrics_path = output_dir / "phase108_nist_ammt_sequence_target_metric_table.csv"
    review_path = output_dir / "phase108_nist_ammt_sequence_target_review_table.csv"
    gate_path = output_dir / "phase108_nist_ammt_sequence_target_gate.json"
    markdown_path = output_dir / "phase108_nist_ammt_sequence_target_summary.md"
    manifest_path = output_dir / "phase108_nist_ammt_sequence_target_manifest.json"
    _write_csv(metrics_path, metric_rows, METRIC_FIELDS)
    _write_csv(review_path, review_rows, REVIEW_FIELDS)
    _write_json(gate_path, gate)
    _write_markdown(markdown_path, gate, review_rows)
    manifest = {
        "phase": 108,
        "objective": "nist_ammt_sequence_target_representation_gate_no_training",
        "inputs": {
            "spatial_field_table": _display_path(spatial_field_table, root),
            "split_manifest": _display_path(split_manifest, root),
            "phase106_gate": _display_path(phase106_gate_path, root),
            "phase107_gate": _display_path(phase107_gate_path, root),
        },
        "outputs": {
            "sequence_field_table": _display_path(table_path, root),
            "metric_table": _display_path(metrics_path, root),
            "review_table": _display_path(review_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown_summary": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "rows": len(rows),
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
        "--spatial-field-table",
        type=Path,
        default=Path(
            "docs/results/phase106_nist_ammt_spatial_target_representation_gate/"
            "phase106_nist_ammt_spatial_target_field_table.csv"
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
        "--phase106-gate",
        type=Path,
        default=Path(
            "docs/results/phase106_nist_ammt_spatial_target_representation_gate/"
            "phase106_nist_ammt_spatial_target_gate.json"
        ),
    )
    parser.add_argument(
        "--phase107-gate",
        type=Path,
        default=Path(
            "docs/results/phase107_nist_ammt_source_region_feature_gate/"
            "phase107_nist_ammt_source_region_feature_gate.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase108_nist_ammt_sequence_target_gate"),
    )
    parser.add_argument("--target-columns", default=",".join(SEQUENCE_TARGET_COLUMNS))
    parser.add_argument("--target-priority", default=",".join(TARGET_PRIORITY))
    parser.add_argument("--min-validation-relative-improvement", type=float, default=0.05)
    parser.add_argument("--min-unsolved-validation-normalized-rmse", type=float, default=0.2)
    parser.add_argument("--n-neighbors", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    spatial_field_table = (
        args.spatial_field_table if args.spatial_field_table.is_absolute() else root / args.spatial_field_table
    )
    split_manifest = (
        args.split_manifest if args.split_manifest.is_absolute() else root / args.split_manifest
    )
    phase106_gate = args.phase106_gate if args.phase106_gate.is_absolute() else root / args.phase106_gate
    phase107_gate = args.phase107_gate if args.phase107_gate.is_absolute() else root / args.phase107_gate
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(
        root=root,
        spatial_field_table=spatial_field_table,
        split_manifest=split_manifest,
        phase106_gate_path=phase106_gate,
        phase107_gate_path=phase107_gate,
        output_dir=output_dir,
        target_columns=_split_csv_arg(args.target_columns),
        target_priority=_split_csv_arg(args.target_priority),
        min_validation_relative_improvement=args.min_validation_relative_improvement,
        min_unsolved_validation_normalized_rmse=args.min_unsolved_validation_normalized_rmse,
        n_neighbors=args.n_neighbors,
        n_estimators=args.n_estimators,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
