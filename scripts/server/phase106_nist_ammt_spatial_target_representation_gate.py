#!/usr/bin/env python3
"""Phase 106 NIST AMMT spatial target-representation gate.

Builds richer no-training target summaries from registered Layer Camera BMP
members, then asks strong tabular baselines whether any target is both
validation-visible and not already solved. Raw ZIP members stay server-local;
the script writes only small CSV/JSON/Markdown artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import zipfile
from pathlib import Path
from typing import Any

from gnnpinn.eval.field_baseline import evaluate_table


METHODS = ("mean", "knn", "extra_trees", "hist_gradient_boosting")
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
SPATIAL_TARGET_COLUMNS = (
    "target_center_mean",
    "target_periphery_mean",
    "target_center_periphery_contrast",
    "target_hot_fraction_q90",
    "target_top_half_mean",
    "target_bottom_half_mean",
    "target_left_half_mean",
    "target_right_half_mean",
    "target_vertical_contrast",
    "target_horizontal_contrast",
    "target_quadrant_contrast",
    "target_grid_max_mean",
    "target_grid_min_mean",
    "target_grid_mean_range",
    "target_local_variance_mean",
    "target_gradient_mean",
    "target_gradient_q90",
    "target_camera_pair_mean_delta",
    "target_camera_pair_std_delta",
    "target_camera_pair_gradient_delta",
)
TARGET_PRIORITY = (
    "target_camera_pair_gradient_delta",
    "target_camera_pair_std_delta",
    "target_center_periphery_contrast",
    "target_grid_mean_range",
    "target_local_variance_mean",
    "target_gradient_q90",
    "target_hot_fraction_q90",
    "target_quadrant_contrast",
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
    "phase106_candidate",
)
BASELINE_GATE_READY_STATUSES = {
    "phase104_baseline_smoke_complete_mechanisms_review_required",
}
PHASE105_CLOSED_STATUSES = {
    "phase105_source_path_feature_gate_blocked_no_hgb_gain",
}


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


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def _zip_path(data_root: Path, file_name: str) -> Path:
    return data_root / file_name


def _bmp_pixels(payload: bytes, *, max_pixels: int) -> dict[str, Any]:
    if payload[:2] != b"BM":
        raise ValueError("Expected BMP target member")
    pixel_offset = struct.unpack_from("<I", payload, 10)[0]
    width = struct.unpack_from("<i", payload, 18)[0]
    height_raw = struct.unpack_from("<i", payload, 22)[0]
    bits_per_pixel = struct.unpack_from("<H", payload, 28)[0]
    if bits_per_pixel != 8:
        raise ValueError(f"Only 8-bit BMP targets are supported, got {bits_per_pixel}")
    height = abs(height_raw)
    row_stride = ((width * bits_per_pixel + 31) // 32) * 4
    stride = 1
    if max_pixels > 0 and width * height > max_pixels:
        stride = int(math.ceil(math.sqrt((width * height) / max_pixels)))
    pixels: list[list[int]] = []
    for row_index in range(height):
        if row_index % stride != 0:
            continue
        start = pixel_offset + row_index * row_stride
        raw_row = payload[start : start + width]
        if len(raw_row) != width:
            raise ValueError("BMP row shorter than expected")
        pixels.append(list(raw_row[::stride]))
    return {
        "width": width,
        "height": height,
        "sample_stride": stride,
        "sampled_width": len(pixels[0]) if pixels else 0,
        "sampled_height": len(pixels),
        "bits_per_pixel": bits_per_pixel,
        "pixels": pixels,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(quantile * len(ordered))) - 1))
    return float(ordered[index])


def _window_values(
    pixels: list[list[int]], row_start: int, row_stop: int, col_start: int, col_stop: int
) -> list[float]:
    return [
        float(pixels[row][col])
        for row in range(max(0, row_start), min(len(pixels), row_stop))
        for col in range(max(0, col_start), min(len(pixels[row]), col_stop))
    ]


def _grid_means(pixels: list[list[int]], grid: int) -> list[float]:
    height = len(pixels)
    width = len(pixels[0]) if pixels else 0
    means: list[float] = []
    for row_bin in range(grid):
        row_start = (row_bin * height) // grid
        row_stop = ((row_bin + 1) * height) // grid
        for col_bin in range(grid):
            col_start = (col_bin * width) // grid
            col_stop = ((col_bin + 1) * width) // grid
            means.append(_mean(_window_values(pixels, row_start, row_stop, col_start, col_stop)))
    return means


def _gradient_values(pixels: list[list[int]]) -> list[float]:
    height = len(pixels)
    width = len(pixels[0]) if pixels else 0
    values: list[float] = []
    for row in range(height):
        for col in range(width):
            gx = 0.0
            gy = 0.0
            if col + 1 < width:
                gx = float(pixels[row][col + 1]) - float(pixels[row][col])
            if row + 1 < height:
                gy = float(pixels[row + 1][col]) - float(pixels[row][col])
            if gx or gy:
                values.append(math.hypot(gx, gy))
    return values


def _local_variances(
    pixels: list[list[int]], window_radius: int, *, max_centers: int = 4096
) -> list[float]:
    height = len(pixels)
    width = len(pixels[0]) if pixels else 0
    center_stride = 1
    if max_centers > 0 and width * height > max_centers:
        center_stride = int(math.ceil(math.sqrt((width * height) / max_centers)))
    values: list[float] = []
    for row in range(0, height, center_stride):
        for col in range(0, width, center_stride):
            local = _window_values(
                pixels,
                row - window_radius,
                row + window_radius + 1,
                col - window_radius,
                col + window_radius + 1,
            )
            values.append(_variance(local))
    return values


def spatial_stats_from_bmp(
    payload: bytes, *, grid_size: int = 4, max_pixels: int = 32768
) -> dict[str, Any]:
    parsed = _bmp_pixels(payload, max_pixels=max_pixels)
    pixels: list[list[int]] = parsed["pixels"]
    original_width = int(parsed["width"])
    original_height = int(parsed["height"])
    width = int(parsed["sampled_width"])
    height = int(parsed["sampled_height"])
    all_values = [float(value) for row in pixels for value in row]
    if not all_values:
        raise ValueError("BMP target has no pixels")
    row_mid = max(1, height // 2)
    col_mid = max(1, width // 2)
    row_margin = height // 4
    col_margin = width // 4
    center = _window_values(pixels, row_margin, height - row_margin, col_margin, width - col_margin)
    if not center:
        center = all_values
    center_bounds = (
        row_margin,
        max(row_margin, height - row_margin),
        col_margin,
        max(col_margin, width - col_margin),
    )
    periphery = [
        float(pixels[row][col])
        for row in range(height)
        for col in range(width)
        if not (
            center_bounds[0] <= row < center_bounds[1]
            and center_bounds[2] <= col < center_bounds[3]
        )
    ]
    if not periphery:
        periphery = all_values
    top = _window_values(pixels, 0, row_mid, 0, width)
    bottom = _window_values(pixels, row_mid, height, 0, width)
    left = _window_values(pixels, 0, height, 0, col_mid)
    right = _window_values(pixels, 0, height, col_mid, width)
    quadrants = [
        _mean(_window_values(pixels, 0, row_mid, 0, col_mid)),
        _mean(_window_values(pixels, 0, row_mid, col_mid, width)),
        _mean(_window_values(pixels, row_mid, height, 0, col_mid)),
        _mean(_window_values(pixels, row_mid, height, col_mid, width)),
    ]
    grid_means = _grid_means(pixels, max(1, grid_size))
    gradients = _gradient_values(pixels)
    local_variances = _local_variances(pixels, window_radius=1)
    q90 = _quantile(all_values, 0.9)
    center_mean = _mean(center)
    periphery_mean = _mean(periphery)
    top_mean = _mean(top)
    bottom_mean = _mean(bottom)
    left_mean = _mean(left)
    right_mean = _mean(right)
    return {
        "target_center_mean": center_mean,
        "target_periphery_mean": periphery_mean,
        "target_center_periphery_contrast": center_mean - periphery_mean,
        "target_hot_fraction_q90": sum(1 for value in all_values if value >= q90) / len(all_values),
        "target_top_half_mean": top_mean,
        "target_bottom_half_mean": bottom_mean,
        "target_left_half_mean": left_mean,
        "target_right_half_mean": right_mean,
        "target_vertical_contrast": top_mean - bottom_mean,
        "target_horizontal_contrast": left_mean - right_mean,
        "target_quadrant_contrast": max(quadrants) - min(quadrants),
        "target_grid_max_mean": max(grid_means),
        "target_grid_min_mean": min(grid_means),
        "target_grid_mean_range": max(grid_means) - min(grid_means),
        "target_local_variance_mean": _mean(local_variances),
        "target_gradient_mean": _mean(gradients),
        "target_gradient_q90": _quantile(gradients, 0.9),
        "target_width": original_width,
        "target_height": original_height,
        "target_sample_stride": int(parsed["sample_stride"]),
        "target_sampled_width": width,
        "target_sampled_height": height,
        "target_sampled_pixel_count": len(all_values),
        "target_intensity_q90_recomputed": q90,
    }


def _camera_key(member_name: str) -> str | None:
    name = Path(member_name).name
    if name.startswith("A"):
        return "A"
    if name.startswith("B"):
        return "B"
    return None


def _camera_pair_deltas(rows: list[dict[str, Any]]) -> None:
    by_layer: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        camera = _camera_key(str(row.get("target_member_name") or ""))
        if camera not in {"A", "B"}:
            continue
        by_layer.setdefault(int(row["source_layer_index"]), {})[camera] = row
    for row in rows:
        pair = by_layer.get(int(row["source_layer_index"]), {})
        other = pair.get("B" if _camera_key(str(row.get("target_member_name") or "")) == "A" else "A")
        if other:
            row["target_camera_pair_mean_delta"] = _float_value(row, "target_center_mean") - _float_value(
                other, "target_center_mean"
            )
            row["target_camera_pair_std_delta"] = _float_value(row, "target_intensity_std") - _float_value(
                other, "target_intensity_std"
            )
            row["target_camera_pair_gradient_delta"] = _float_value(row, "target_gradient_mean") - _float_value(
                other, "target_gradient_mean"
            )
        else:
            row["target_camera_pair_mean_delta"] = 0.0
            row["target_camera_pair_std_delta"] = 0.0
            row["target_camera_pair_gradient_delta"] = 0.0


def _float_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float, str)) and str(value) != "" else 0.0


def build_spatial_rows(
    *,
    numeric_rows: list[dict[str, str]],
    data_root: Path,
    grid_size: int,
    max_pixels_per_target: int,
) -> list[dict[str, Any]]:
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    archives: dict[str, zipfile.ZipFile] = {}
    rows: list[dict[str, Any]] = []
    try:
        for row in numeric_rows:
            target_file = row.get("target_file_name") or "In-situ Meas Data.zip"
            target_member = row["target_member_name"]
            target_key = (target_file, target_member)
            if target_key not in cache:
                if target_file not in archives:
                    archives[target_file] = zipfile.ZipFile(_zip_path(data_root, target_file))
                archive = archives[target_file]
                cache[target_key] = spatial_stats_from_bmp(
                    archive.read(target_member),
                    grid_size=grid_size,
                    max_pixels=max_pixels_per_target,
                )
            rows.append(
                {
                    **row,
                    **cache[target_key],
                }
            )
    finally:
        for archive in archives.values():
            archive.close()
    _camera_pair_deltas(rows)
    return rows


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
            status = "candidate_spatial_target_gap_ready_for_focused_validation"
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
                "phase106_candidate": status
                == "candidate_spatial_target_gap_ready_for_focused_validation",
            }
        )
    return rows


def _build_gate(
    *,
    baseline_gate: dict[str, Any],
    phase105_gate: dict[str, Any],
    review_rows: list[dict[str, Any]],
    min_validation_relative_improvement: float,
    min_unsolved_validation_normalized_rmse: float,
) -> dict[str, Any]:
    candidates = [row for row in review_rows if row["phase106_candidate"]]
    candidates.sort(
        key=lambda row: (
            int(row["physical_priority"]),
            -float(row["selected_validation_normalized_rmse"] or 0.0),
            float(row["selected_validation_rmse"]),
        )
    )
    selected = candidates[0] if candidates else None
    baseline_ready = (
        baseline_gate.get("status") in BASELINE_GATE_READY_STATUSES
        and bool(baseline_gate.get("baseline_smoke_completed"))
        and bool(baseline_gate.get("sample_size_sufficient_for_phase105"))
    )
    phase105_closed = phase105_gate.get("status") in PHASE105_CLOSED_STATUSES
    if not baseline_ready:
        status = "phase106_spatial_target_gate_blocked_by_phase104_baseline"
        next_action = "complete Phase 104 expanded baseline smoke first"
    elif not phase105_closed:
        status = "phase106_spatial_target_gate_blocked_by_phase105_boundary"
        next_action = "close Phase 105 scalar source/path proxy gate before refining targets"
    elif selected:
        status = "phase106_spatial_target_gap_ready_focused_no_training_validation"
        next_action = (
            "enter seed-7 focused validation or low-capacity mechanism design only after "
            f"reviewing {selected['target']} against route guards"
        )
    else:
        status = "phase106_spatial_target_gate_closed_no_baseline_gap"
        next_action = "close as target-representation diagnostic; do not train models on these targets"
    return {
        "status": status,
        "baseline_gate_status": baseline_gate.get("status"),
        "phase105_gate_status": phase105_gate.get("status"),
        "target_representation": "registered_layer_camera_spatial_statistics_v1",
        "target_columns": list(SPATIAL_TARGET_COLUMNS),
        "candidate_targets": [row["target"] for row in candidates],
        "selected_target": selected["target"] if selected else None,
        "selected_validation_method": selected["selected_validation_method"] if selected else None,
        "selected_validation_rmse": selected["selected_validation_rmse"] if selected else None,
        "selected_validation_normalized_rmse": selected["selected_validation_normalized_rmse"] if selected else None,
        "selected_test_rmse": selected["selected_test_rmse"] if selected else None,
        "selected_test_normalized_rmse": selected["selected_test_normalized_rmse"] if selected else None,
        "min_validation_relative_improvement": min_validation_relative_improvement,
        "min_unsolved_validation_normalized_rmse": min_unsolved_validation_normalized_rmse,
        "phase106_seed7_focused_validation_allowed": bool(selected and baseline_ready and phase105_closed),
        "phase106_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
    }


def _write_markdown(path: Path, gate: dict[str, Any], review_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 106 NIST AMMT Spatial Target Representation Gate",
        "",
        f"- Status: `{gate['status']}`",
        f"- Representation: `{gate['target_representation']}`",
        f"- Selected target: `{gate['selected_target']}`",
        f"- Focused validation allowed: `{gate['phase106_seed7_focused_validation_allowed']}`",
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
    data_root: Path,
    numeric_field_table: Path,
    split_manifest: Path,
    baseline_gate_path: Path,
    phase105_gate_path: Path,
    output_dir: Path,
    grid_size: int,
    max_pixels_per_target: int,
    target_columns: tuple[str, ...],
    target_priority: tuple[str, ...],
    min_validation_relative_improvement: float,
    min_unsolved_validation_normalized_rmse: float,
    n_neighbors: int,
    n_estimators: int,
) -> dict[str, Any]:
    baseline_gate = _read_json(baseline_gate_path)
    phase105_gate = _read_json(phase105_gate_path)
    spatial_rows = build_spatial_rows(
        numeric_rows=_read_csv(numeric_field_table),
        data_root=data_root,
        grid_size=grid_size,
        max_pixels_per_target=max_pixels_per_target,
    )
    target_ranges = _target_ranges(spatial_rows, target_columns)
    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_table_path = output_dir / "phase106_nist_ammt_spatial_target_field_table.csv"
    spatial_fields = tuple(spatial_rows[0].keys())
    _write_csv(spatial_table_path, spatial_rows, spatial_fields)

    payloads: dict[str, dict[str, dict[str, Any]]] = {}
    for target in target_columns:
        payloads[target] = {}
        for method in METHODS:
            payloads[target][method] = {
                "target": target,
                "strategy": method,
                "split_manifest": str(split_manifest),
                "fit_split": "train",
                "feature_columns": list(FEATURE_COLUMNS) if method != "mean" else None,
                "results": [
                    evaluate_table(
                        table_path=spatial_table_path,
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
        min_unsolved_validation_normalized_rmse=min_unsolved_validation_normalized_rmse,
    )
    gate = _build_gate(
        baseline_gate=baseline_gate,
        phase105_gate=phase105_gate,
        review_rows=review_rows,
        min_validation_relative_improvement=min_validation_relative_improvement,
        min_unsolved_validation_normalized_rmse=min_unsolved_validation_normalized_rmse,
    )

    metrics_path = output_dir / "phase106_nist_ammt_spatial_target_metric_table.csv"
    review_path = output_dir / "phase106_nist_ammt_spatial_target_review_table.csv"
    gate_path = output_dir / "phase106_nist_ammt_spatial_target_gate.json"
    markdown_path = output_dir / "phase106_nist_ammt_spatial_target_summary.md"
    manifest_path = output_dir / "phase106_nist_ammt_spatial_target_manifest.json"
    payload_path = output_dir / "phase106_nist_ammt_spatial_target_payloads.json"
    _write_csv(metrics_path, metric_rows, METRIC_FIELDS)
    _write_csv(review_path, review_rows, REVIEW_FIELDS)
    _write_json(gate_path, gate)
    _write_json(payload_path, payloads)
    _write_markdown(markdown_path, gate, review_rows)
    manifest = {
        "phase": 106,
        "objective": "nist_ammt_spatial_sequence_target_representation_gate_no_training",
        "inputs": {
            "data_root": _display_path(data_root, root),
            "numeric_field_table": _display_path(numeric_field_table, root),
            "split_manifest": _display_path(split_manifest, root),
            "baseline_gate": _display_path(baseline_gate_path, root),
            "phase105_gate": _display_path(phase105_gate_path, root),
        },
        "outputs": {
            "spatial_field_table": _display_path(spatial_table_path, root),
            "metric_table": _display_path(metrics_path, root),
            "review_table": _display_path(review_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown_summary": _display_path(markdown_path, root),
            "payloads": _display_path(payload_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "limits": {
            "grid_size": grid_size,
            "max_pixels_per_target": max_pixels_per_target,
            "min_validation_relative_improvement": min_validation_relative_improvement,
            "min_unsolved_validation_normalized_rmse": min_unsolved_validation_normalized_rmse,
        },
        "counts": {
            "rows": len(spatial_rows),
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
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/nist_ammt/mds2_2044"))
    parser.add_argument(
        "--numeric-field-table",
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
        "--phase105-gate",
        type=Path,
        default=Path(
            "docs/results/phase105_nist_ammt_source_path_feature_gate/"
            "phase105_nist_ammt_source_path_feature_gate.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase106_nist_ammt_spatial_target_representation_gate"),
    )
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--max-pixels-per-target", type=int, default=65536)
    parser.add_argument("--target-columns", default=",".join(SPATIAL_TARGET_COLUMNS))
    parser.add_argument("--target-priority", default=",".join(TARGET_PRIORITY))
    parser.add_argument("--min-validation-relative-improvement", type=float, default=0.05)
    parser.add_argument("--min-unsolved-validation-normalized-rmse", type=float, default=0.2)
    parser.add_argument("--n-neighbors", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    data_root = args.data_root if args.data_root.is_absolute() else root / args.data_root
    numeric_field_table = (
        args.numeric_field_table
        if args.numeric_field_table.is_absolute()
        else root / args.numeric_field_table
    )
    split_manifest = (
        args.split_manifest if args.split_manifest.is_absolute() else root / args.split_manifest
    )
    baseline_gate = args.baseline_gate if args.baseline_gate.is_absolute() else root / args.baseline_gate
    phase105_gate = args.phase105_gate if args.phase105_gate.is_absolute() else root / args.phase105_gate
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(
        root=root,
        data_root=data_root,
        numeric_field_table=numeric_field_table,
        split_manifest=split_manifest,
        baseline_gate_path=baseline_gate,
        phase105_gate_path=phase105_gate,
        output_dir=output_dir,
        grid_size=args.grid_size,
        max_pixels_per_target=args.max_pixels_per_target,
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
