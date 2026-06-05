#!/usr/bin/env python3
"""Phase 107 NIST AMMT source-region feature gate.

Adds deterministic source-path spatial region descriptors for the Phase 106
selected spatial target, then replays strong tabular baselines before any
neural model training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path
from typing import Any, BinaryIO

from gnnpinn.eval.field_baseline import evaluate_table


METHODS = ("knn", "extra_trees", "hist_gradient_boosting")
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
SOURCE_REGION_FEATURES = (
    "source_center_power_fraction",
    "source_periphery_power_fraction",
    "source_center_periphery_power_balance",
    "source_left_power_fraction",
    "source_right_power_fraction",
    "source_top_power_fraction",
    "source_bottom_power_fraction",
    "source_horizontal_power_balance",
    "source_vertical_power_balance",
    "source_quadrant_power_fraction_range",
    "source_grid_power_fraction_range",
    "source_power_centroid_x_norm",
    "source_power_centroid_y_norm",
    "source_power_spread_x_norm",
    "source_power_spread_y_norm",
    "source_power_radius_norm",
    "source_temporal_x_drift_norm",
    "source_temporal_y_drift_norm",
    "source_active_point_fraction_sampled",
)
FEATURE_PROFILES: dict[str, tuple[str, ...]] = {
    "phase106_guard_replay": BASE_FEATURES,
    "source_center_periphery": BASE_FEATURES
    + (
        "source_center_power_fraction",
        "source_periphery_power_fraction",
        "source_center_periphery_power_balance",
        "source_horizontal_power_balance",
        "source_vertical_power_balance",
    ),
    "source_grid_region": BASE_FEATURES
    + (
        "source_quadrant_power_fraction_range",
        "source_grid_power_fraction_range",
        "source_power_radius_norm",
    ),
    "source_moment_drift": BASE_FEATURES
    + (
        "source_power_centroid_x_norm",
        "source_power_centroid_y_norm",
        "source_power_spread_x_norm",
        "source_power_spread_y_norm",
        "source_temporal_x_drift_norm",
        "source_temporal_y_drift_norm",
    ),
    "source_region_all": BASE_FEATURES + SOURCE_REGION_FEATURES,
    "source_region_only": SOURCE_REGION_FEATURES + ("target_camera_code", "t"),
}
METRIC_FIELDS = (
    "feature_profile",
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
    "feature_profile",
    "selected_validation_method",
    "selected_validation_rmse",
    "selected_test_rmse",
    "selected_validation_normalized_rmse",
    "selected_test_normalized_rmse",
    "guard_validation_rmse",
    "guard_test_rmse",
    "validation_improvement_over_guard",
    "test_improvement_over_guard",
    "validation_improves_guard",
    "test_improves_guard",
    "status",
    "phase107_candidate",
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


def _zip_path(data_root: Path, file_name: str) -> Path:
    return data_root / file_name


def _parse_source_rows(
    handle: BinaryIO, max_source_rows: int | None
) -> tuple[list[tuple[float, float, float, float]], bool]:
    rows: list[tuple[float, float, float, float]] = []
    truncated = False
    for raw_line in handle:
        stripped = raw_line.decode("utf-8", errors="replace").strip()
        if not stripped:
            continue
        parts = [item.strip() for item in stripped.split(",")]
        if len(parts) < 4:
            continue
        try:
            x, y, power, time_value = (float(parts[index]) for index in range(4))
        except ValueError:
            continue
        rows.append((x, y, power, time_value))
        if max_source_rows is not None and len(rows) >= max_source_rows:
            truncated = True
            break
    if not rows:
        raise ValueError("No numeric XYPT rows were read")
    return rows, truncated


def _safe_range(values: list[float]) -> float:
    span = max(values) - min(values)
    return span if span > 1e-12 else 1.0


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0.0:
        return sum(values) / len(values)
    return sum(value * weight for value, weight in zip(values, weights)) / total


def _weighted_spread(values: list[float], weights: list[float], mean: float) -> float:
    total = sum(weights)
    if total <= 0.0:
        return 0.0
    return math.sqrt(sum(((value - mean) ** 2) * weight for value, weight in zip(values, weights)) / total)


def _source_region_stats(
    handle: BinaryIO, *, max_source_rows: int | None, grid_size: int
) -> dict[str, Any]:
    source_rows, truncated = _parse_source_rows(handle, max_source_rows)
    xs = [row[0] for row in source_rows]
    ys = [row[1] for row in source_rows]
    powers = [max(row[2], 0.0) for row in source_rows]
    total_power = sum(powers)
    if total_power <= 0.0:
        powers = [1.0 for _ in source_rows]
        total_power = float(len(source_rows))
    x_min = min(xs)
    y_min = min(ys)
    x_range = _safe_range(xs)
    y_range = _safe_range(ys)
    x_norm = [(value - x_min) / x_range for value in xs]
    y_norm = [(value - y_min) / y_range for value in ys]

    def fraction(mask: list[bool]) -> float:
        return sum(weight for weight, keep in zip(powers, mask) if keep) / total_power

    center_mask = [0.25 <= x <= 0.75 and 0.25 <= y <= 0.75 for x, y in zip(x_norm, y_norm)]
    left = fraction([x < 0.5 for x in x_norm])
    right = fraction([x >= 0.5 for x in x_norm])
    bottom = fraction([y < 0.5 for y in y_norm])
    top = fraction([y >= 0.5 for y in y_norm])
    quadrants = [
        fraction([x < 0.5 and y < 0.5 for x, y in zip(x_norm, y_norm)]),
        fraction([x >= 0.5 and y < 0.5 for x, y in zip(x_norm, y_norm)]),
        fraction([x < 0.5 and y >= 0.5 for x, y in zip(x_norm, y_norm)]),
        fraction([x >= 0.5 and y >= 0.5 for x, y in zip(x_norm, y_norm)]),
    ]
    grid = max(1, grid_size)
    grid_weights = [0.0 for _ in range(grid * grid)]
    for x, y, weight in zip(x_norm, y_norm, powers):
        col = min(grid - 1, max(0, int(x * grid)))
        row = min(grid - 1, max(0, int(y * grid)))
        grid_weights[row * grid + col] += weight
    grid_fractions = [weight / total_power for weight in grid_weights]
    cx = _weighted_mean(x_norm, powers)
    cy = _weighted_mean(y_norm, powers)
    sx = _weighted_spread(x_norm, powers, cx)
    sy = _weighted_spread(y_norm, powers, cy)
    split = max(1, len(source_rows) // 5)
    early_weights = powers[:split]
    late_weights = powers[-split:]
    early_x = _weighted_mean(x_norm[:split], early_weights)
    late_x = _weighted_mean(x_norm[-split:], late_weights)
    early_y = _weighted_mean(y_norm[:split], early_weights)
    late_y = _weighted_mean(y_norm[-split:], late_weights)
    center_fraction = fraction(center_mask)
    periphery_fraction = 1.0 - center_fraction
    return {
        "source_center_power_fraction": center_fraction,
        "source_periphery_power_fraction": periphery_fraction,
        "source_center_periphery_power_balance": center_fraction - periphery_fraction,
        "source_left_power_fraction": left,
        "source_right_power_fraction": right,
        "source_top_power_fraction": top,
        "source_bottom_power_fraction": bottom,
        "source_horizontal_power_balance": left - right,
        "source_vertical_power_balance": top - bottom,
        "source_quadrant_power_fraction_range": max(quadrants) - min(quadrants),
        "source_grid_power_fraction_range": max(grid_fractions) - min(grid_fractions),
        "source_power_centroid_x_norm": cx,
        "source_power_centroid_y_norm": cy,
        "source_power_spread_x_norm": sx,
        "source_power_spread_y_norm": sy,
        "source_power_radius_norm": math.hypot(sx, sy),
        "source_temporal_x_drift_norm": late_x - early_x,
        "source_temporal_y_drift_norm": late_y - early_y,
        "source_active_point_fraction_sampled": sum(1 for power in powers if power > 0.0) / len(powers),
        "source_region_rows_read": len(source_rows),
        "source_region_rows_truncated": truncated,
        "source_region_grid_size": grid,
    }


def build_source_region_rows(
    *,
    spatial_rows: list[dict[str, str]],
    data_root: Path,
    max_source_rows: int | None,
    grid_size: int,
) -> list[dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    archives: dict[str, zipfile.ZipFile] = {}
    rows: list[dict[str, Any]] = []
    try:
        for row in spatial_rows:
            source_file = row.get("source_file_name") or "Build Command Data.zip"
            source_member = row["source_member_name"]
            if source_member not in cache:
                if source_file not in archives:
                    archives[source_file] = zipfile.ZipFile(_zip_path(data_root, source_file))
                with archives[source_file].open(source_member) as handle:
                    cache[source_member] = _source_region_stats(
                        handle,
                        max_source_rows=max_source_rows,
                        grid_size=grid_size,
                    )
            rows.append({**row, **cache[source_member]})
    finally:
        for archive in archives.values():
            archive.close()
    return rows


def _region_rmse(split_payload: dict[str, Any], name: str) -> float | None:
    region = (split_payload.get("region_metrics") or {}).get(name) or {}
    metrics = region.get("metrics") or {}
    value = metrics.get("rmse")
    return float(value) if isinstance(value, (int, float)) else None


def _metric_rows(payloads: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile, profile_payloads in payloads.items():
        for method, payload in profile_payloads.items():
            result = payload["results"][0]
            for split, split_payload in result["split_metrics"].items():
                metrics = split_payload["metrics"]
                rows.append(
                    {
                        "feature_profile": profile,
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
            index[(str(row["feature_profile"]), str(row["method"]), str(row["split"]))] = float(value)
    return index


def _review_rows(
    *,
    metric_rows: list[dict[str, Any]],
    guard_validation_rmse: float,
    guard_test_rmse: float,
    min_validation_improvement: float,
) -> list[dict[str, Any]]:
    rmse = _metric_index(metric_rows, "rmse")
    normalized = _metric_index(metric_rows, "normalized_rmse")
    rows: list[dict[str, Any]] = []
    for profile in FEATURE_PROFILES:
        candidates = [
            (rmse[(profile, method, "val")], method)
            for method in METHODS
            if (profile, method, "val") in rmse
        ]
        if not candidates:
            raise ValueError(f"No validation metrics found for feature profile {profile}")
        selected_val, selected_method = min(candidates)
        selected_test = rmse[(profile, selected_method, "test")]
        validation_delta = guard_validation_rmse - selected_val
        test_delta = guard_test_rmse - selected_test
        validation_improves = validation_delta >= min_validation_improvement
        test_improves = test_delta >= 0.0
        if profile == "phase106_guard_replay":
            status = "phase106_guard_replay_reference"
            candidate = False
        elif validation_improves and test_improves:
            status = "candidate_source_region_profile_ready_for_focused_review"
            candidate = True
        elif not validation_improves:
            status = "blocked_no_validation_gain_over_phase106_guard"
            candidate = False
        else:
            status = "blocked_test_reversal_against_phase106_guard"
            candidate = False
        rows.append(
            {
                "feature_profile": profile,
                "selected_validation_method": selected_method,
                "selected_validation_rmse": selected_val,
                "selected_test_rmse": selected_test,
                "selected_validation_normalized_rmse": normalized.get((profile, selected_method, "val")),
                "selected_test_normalized_rmse": normalized.get((profile, selected_method, "test")),
                "guard_validation_rmse": guard_validation_rmse,
                "guard_test_rmse": guard_test_rmse,
                "validation_improvement_over_guard": validation_delta,
                "test_improvement_over_guard": test_delta,
                "validation_improves_guard": validation_improves,
                "test_improves_guard": test_improves,
                "status": status,
                "phase107_candidate": candidate,
            }
        )
    return rows


def _build_gate(
    *,
    phase106_gate: dict[str, Any],
    review_rows: list[dict[str, Any]],
    min_validation_improvement: float,
    max_source_rows: int | None,
    grid_size: int,
) -> dict[str, Any]:
    candidates = [row for row in review_rows if row["phase107_candidate"]]
    candidates.sort(key=lambda row: float(row["selected_validation_rmse"]))
    selected = candidates[0] if candidates else None
    phase106_ready = (
        phase106_gate.get("status") == "phase106_spatial_target_gap_ready_focused_no_training_validation"
        and bool(phase106_gate.get("phase106_seed7_focused_validation_allowed"))
    )
    target = phase106_gate.get("selected_target") or "target_center_periphery_contrast"
    if not phase106_ready:
        status = "phase107_source_region_gate_blocked_by_phase106"
        next_action = "complete Phase 106 spatial target representation gate first"
    elif selected:
        status = "phase107_source_region_feature_gate_ready_focused_review"
        next_action = (
            f"review {selected['feature_profile']} source-region features on {target} "
            "before any model training"
        )
    else:
        status = "phase107_source_region_feature_gate_blocked_no_phase106_gain"
        next_action = "close sampled source-region path features as diagnostic; do not train"
    return {
        "status": status,
        "target": target,
        "phase106_gate_status": phase106_gate.get("status"),
        "phase106_selected_validation_method": phase106_gate.get("selected_validation_method"),
        "phase106_selected_validation_rmse": phase106_gate.get("selected_validation_rmse"),
        "phase106_selected_test_rmse": phase106_gate.get("selected_test_rmse"),
        "source_region_feature_profiles": list(FEATURE_PROFILES),
        "candidate_feature_profiles": [row["feature_profile"] for row in candidates],
        "selected_feature_profile": selected["feature_profile"] if selected else None,
        "selected_validation_method": selected["selected_validation_method"] if selected else None,
        "selected_validation_rmse": selected["selected_validation_rmse"] if selected else None,
        "selected_test_rmse": selected["selected_test_rmse"] if selected else None,
        "min_validation_improvement": min_validation_improvement,
        "max_source_rows_per_member": max_source_rows,
        "source_region_grid_size": grid_size,
        "phase107_focused_review_allowed": bool(selected and phase106_ready),
        "phase107_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
    }


def _write_markdown(path: Path, gate: dict[str, Any], review_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 107 NIST AMMT Source-Region Feature Gate",
        "",
        f"- Status: `{gate['status']}`",
        f"- Target: `{gate['target']}`",
        f"- Phase 106 guard: `{gate['phase106_selected_validation_method']}` val RMSE `{gate['phase106_selected_validation_rmse']}`",
        f"- Selected feature profile: `{gate['selected_feature_profile']}`",
        f"- Focused review allowed: `{gate['phase107_focused_review_allowed']}`",
        "- Model training allowed: `false`",
        "- A100 training allowed now: `false`",
        "",
        "| Feature profile | Status | Method | Val RMSE | Test RMSE | Val gain vs guard | Test gain vs guard |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in review_rows:
        lines.append(
            "| {feature_profile} | {status} | {selected_validation_method} | {selected_validation_rmse} | {selected_test_rmse} | {validation_improvement_over_guard} | {test_improvement_over_guard} |".format(
                **{key: _csv_value(value) for key, value in row.items()}
            )
        )
    lines.extend(["", f"Next action: {gate['next_action']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_package(
    *,
    root: Path,
    data_root: Path,
    spatial_field_table: Path,
    split_manifest: Path,
    phase106_gate_path: Path,
    output_dir: Path,
    min_validation_improvement: float,
    max_source_rows: int | None,
    grid_size: int,
    n_neighbors: int,
    n_estimators: int,
) -> dict[str, Any]:
    phase106_gate = _read_json(phase106_gate_path)
    target = str(phase106_gate.get("selected_target") or "target_center_periphery_contrast")
    rows = build_source_region_rows(
        spatial_rows=_read_csv(spatial_field_table),
        data_root=data_root,
        max_source_rows=max_source_rows,
        grid_size=grid_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    augmented_path = output_dir / "phase107_nist_ammt_source_region_augmented_field_table.csv"
    _write_csv(augmented_path, rows, tuple(rows[0].keys()))
    payloads: dict[str, dict[str, dict[str, Any]]] = {}
    for profile, features in FEATURE_PROFILES.items():
        payloads[profile] = {}
        for method in METHODS:
            payloads[profile][method] = {
                "target": target,
                "strategy": method,
                "feature_profile": profile,
                "feature_columns": list(features),
                "split_manifest": str(split_manifest),
                "fit_split": "train",
                "results": [
                    evaluate_table(
                        table_path=augmented_path,
                        target=target,
                        strategy=method,
                        split_manifest_path=split_manifest,
                        fit_split="train",
                        feature_columns=list(features),
                        n_neighbors=n_neighbors,
                        n_estimators=n_estimators,
                        random_state=7,
                        hot_quantiles=[0.9],
                        gradient_quantiles=[0.9],
                    )
                ],
            }
    metric_rows = _metric_rows(payloads)
    guard_validation = float(phase106_gate["selected_validation_rmse"])
    guard_test = float(phase106_gate["selected_test_rmse"])
    review_rows = _review_rows(
        metric_rows=metric_rows,
        guard_validation_rmse=guard_validation,
        guard_test_rmse=guard_test,
        min_validation_improvement=min_validation_improvement,
    )
    gate = _build_gate(
        phase106_gate=phase106_gate,
        review_rows=review_rows,
        min_validation_improvement=min_validation_improvement,
        max_source_rows=max_source_rows,
        grid_size=grid_size,
    )

    metrics_path = output_dir / "phase107_nist_ammt_source_region_metric_table.csv"
    review_path = output_dir / "phase107_nist_ammt_source_region_review_table.csv"
    gate_path = output_dir / "phase107_nist_ammt_source_region_feature_gate.json"
    markdown_path = output_dir / "phase107_nist_ammt_source_region_feature_summary.md"
    manifest_path = output_dir / "phase107_nist_ammt_source_region_feature_manifest.json"
    _write_csv(metrics_path, metric_rows, METRIC_FIELDS)
    _write_csv(review_path, review_rows, REVIEW_FIELDS)
    _write_json(gate_path, gate)
    _write_markdown(markdown_path, gate, review_rows)
    manifest = {
        "phase": 107,
        "objective": "nist_ammt_source_region_feature_gate_no_training",
        "inputs": {
            "data_root": _display_path(data_root, root),
            "spatial_field_table": _display_path(spatial_field_table, root),
            "split_manifest": _display_path(split_manifest, root),
            "phase106_gate": _display_path(phase106_gate_path, root),
        },
        "outputs": {
            "augmented_field_table": _display_path(augmented_path, root),
            "metric_table": _display_path(metrics_path, root),
            "review_table": _display_path(review_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown_summary": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "limits": {
            "max_source_rows_per_member": max_source_rows,
            "source_region_grid_size": grid_size,
            "min_validation_improvement": min_validation_improvement,
        },
        "counts": {
            "rows": len(rows),
            "feature_profiles": len(FEATURE_PROFILES),
            "metric_rows": len(metric_rows),
            "candidate_feature_profiles": len(gate["candidate_feature_profiles"]),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/nist_ammt/mds2_2044"))
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
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase107_nist_ammt_source_region_feature_gate"),
    )
    parser.add_argument("--min-validation-improvement", type=float, default=0.005)
    parser.add_argument("--max-source-rows", type=int, default=200000)
    parser.add_argument("--source-region-grid-size", type=int, default=4)
    parser.add_argument("--n-neighbors", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    data_root = args.data_root if args.data_root.is_absolute() else root / args.data_root
    spatial_field_table = (
        args.spatial_field_table if args.spatial_field_table.is_absolute() else root / args.spatial_field_table
    )
    split_manifest = (
        args.split_manifest if args.split_manifest.is_absolute() else root / args.split_manifest
    )
    phase106_gate = args.phase106_gate if args.phase106_gate.is_absolute() else root / args.phase106_gate
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    max_source_rows = args.max_source_rows if args.max_source_rows > 0 else None
    manifest = build_package(
        root=root,
        data_root=data_root,
        spatial_field_table=spatial_field_table,
        split_manifest=split_manifest,
        phase106_gate_path=phase106_gate,
        output_dir=output_dir,
        min_validation_improvement=args.min_validation_improvement,
        max_source_rows=max_source_rows,
        grid_size=args.source_region_grid_size,
        n_neighbors=args.n_neighbors,
        n_estimators=args.n_estimators,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
