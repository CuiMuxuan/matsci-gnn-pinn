#!/usr/bin/env python3
"""Phase 112 NIST AMMT melt-pool target representation gate.

This no-training gate probes whether registered Melt Pool Camera layer
directories provide a target that is not immediately solved by strong tabular
baselines or by a trivial layer/time shortcut. Raw ZIP members stay server-local;
the script writes only small CSV/JSON/Markdown artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
import zipfile
from pathlib import Path
from typing import Any

from gnnpinn.eval.field_baseline import evaluate_table


METHODS = ("mean", "knn", "extra_trees", "hist_gradient_boosting")
TARGET_COLUMNS = (
    "target_mp_mean_mean",
    "target_mp_mean_std",
    "target_mp_q90_mean",
    "target_mp_max_mean",
    "target_mp_max_range",
    "target_mp_temporal_mean_range",
    "target_mp_early_late_mean_delta",
    "target_mp_peak_frame_position",
)
TARGET_PRIORITY = (
    "target_mp_temporal_mean_range",
    "target_mp_early_late_mean_delta",
    "target_mp_mean_std",
    "target_mp_q90_mean",
    "target_mp_max_range",
)
FEATURE_PROFILES: dict[str, tuple[str, ...]] = {
    "layer_time_only": ("t", "source_layer_index", "target_layer_index"),
    "source_geometry": ("x", "y", "source_x_range", "source_y_range"),
    "source_power_path": (
        "source_p_mean",
        "source_p_nonzero_fraction",
        "source_p_range",
        "source_t_range",
        "source_rows_read",
    ),
    "source_all_no_layer": (
        "x",
        "y",
        "source_x_range",
        "source_y_range",
        "source_p_mean",
        "source_p_nonzero_fraction",
        "source_p_range",
        "source_t_range",
        "source_rows_read",
    ),
}
PRIMARY_PROFILES = ("source_power_path", "source_geometry", "source_all_no_layer")
SHORTCUT_PROFILES = ("layer_time_only",)
METRIC_FIELDS = (
    "target",
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
    "target",
    "target_min",
    "target_max",
    "target_range",
    "selected_feature_profile",
    "selected_validation_method",
    "selected_validation_rmse",
    "selected_validation_normalized_rmse",
    "selected_test_rmse",
    "selected_test_normalized_rmse",
    "mean_validation_rmse",
    "mean_validation_normalized_rmse",
    "mean_test_rmse",
    "mean_test_normalized_rmse",
    "layer_time_validation_rmse",
    "layer_time_validation_normalized_rmse",
    "layer_time_shortcut_detected",
    "validation_relative_improvement_over_mean",
    "test_relative_improvement_over_mean",
    "baseline_visible_gap",
    "strong_baseline_solved",
    "zero_variance_target",
    "physical_priority",
    "status",
    "phase112_candidate",
)
FIELD_COLUMNS = (
    "x",
    "y",
    "t",
    "source_layer_index",
    "target_layer_index",
    "melt_pool_frame_count",
    "sampled_frame_count",
    "first_sampled_frame_index",
    "last_sampled_frame_index",
    "source_p_mean",
    "source_p_nonzero_fraction",
    "source_x_range",
    "source_y_range",
    "source_p_range",
    "source_t_range",
    "source_rows_read",
    *TARGET_COLUMNS,
    "split_name",
    "row_id",
    "target_group_key",
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


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value in (None, ""):
        return default
    return int(float(value))


def _source_rows_by_layer(numeric_rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    by_layer: dict[int, dict[str, str]] = {}
    for row in numeric_rows:
        layer = _int(row, "source_layer_index")
        current = by_layer.get(layer)
        if current is None or _int(row, "target_camera_code", 99) < _int(
            current, "target_camera_code", 99
        ):
            by_layer[layer] = row
    return by_layer


def _melt_pool_join(join_rows: list[dict[str, str]]) -> dict[str, Any] | None:
    for row in join_rows:
        if (
            row.get("target_type") == "melt_pool_camera_layer_directory"
            and row.get("join_evidence_status") == "source_target_layer_join_ready"
        ):
            return {
                "target_group_key": row["target_group_key"],
                "source_minus_target_offset": _int(row, "best_source_minus_target_offset"),
                "target_first_index": _int(row, "target_first_index"),
                "target_last_index": _int(row, "target_last_index"),
                "target_count": _int(row, "target_count"),
            }
    return None


def _groups_from_sequence_csv(sequence_rows: list[dict[str, str]]) -> dict[int, dict[str, Any]]:
    groups: dict[int, dict[str, Any]] = {}
    for row in sequence_rows:
        key = row.get("group_key", "")
        match = re.search(r"Melt Pool Camera/MIA_L(\d+)/frame", key)
        if not match:
            continue
        layer = int(match.group(1))
        groups[layer] = {
            "target_layer_index": layer,
            "frame_count": _int(row, "count"),
            "first_frame_index": _int(row, "first_index"),
            "last_frame_index": _int(row, "last_index"),
            "zero_padded_width": _int(row, "zero_padded_width", 5),
            "group_key": key,
        }
    return groups


def _groups_from_zip(data_root: Path, file_name: str = "In-situ Meas Data.zip") -> dict[int, dict[str, Any]]:
    pattern = re.compile(r"^In-situ Meas Data/Melt Pool Camera/MIA_L(\d+)/frame(\d+)\.bmp$")
    grouped: dict[int, list[int]] = {}
    widths: dict[int, int] = {}
    with zipfile.ZipFile(data_root / file_name) as archive:
        for info in archive.infolist():
            match = pattern.match(info.filename)
            if not match:
                continue
            layer = int(match.group(1))
            frame = int(match.group(2))
            grouped.setdefault(layer, []).append(frame)
            widths[layer] = len(match.group(2))
    output: dict[int, dict[str, Any]] = {}
    for layer, frames in grouped.items():
        output[layer] = {
            "target_layer_index": layer,
            "frame_count": len(frames),
            "first_frame_index": min(frames),
            "last_frame_index": max(frames),
            "zero_padded_width": widths.get(layer, 5),
            "group_key": f"In-situ Meas Data/Melt Pool Camera/MIA_L{layer:04d}/frame{{index}}.bmp",
        }
    return output


def _sample_frame_indices(first: int, last: int, count: int, max_frames: int) -> list[int]:
    if count <= 0:
        return []
    if max_frames <= 0 or count <= max_frames:
        return list(range(first, last + 1))
    if max_frames == 1:
        return [first]
    values = []
    for index in range(max_frames):
        position = first + round(index * (last - first) / (max_frames - 1))
        values.append(int(position))
    return sorted(set(values))


def _bmp_frame_stats(payload: bytes) -> dict[str, float]:
    if payload[:2] != b"BM":
        raise ValueError("Expected BMP melt-pool frame")
    pixel_offset = struct.unpack_from("<I", payload, 10)[0]
    width = struct.unpack_from("<i", payload, 18)[0]
    height_raw = struct.unpack_from("<i", payload, 22)[0]
    bits_per_pixel = struct.unpack_from("<H", payload, 28)[0]
    if bits_per_pixel != 8:
        raise ValueError(f"Only 8-bit BMP frames are supported, got {bits_per_pixel}")
    height = abs(height_raw)
    row_stride = ((width * bits_per_pixel + 31) // 32) * 4
    hist = [0 for _ in range(256)]
    for row_index in range(height):
        start = pixel_offset + row_index * row_stride
        row = payload[start : start + width]
        for value in row:
            hist[value] += 1
    pixel_count = sum(hist)
    if pixel_count != width * height:
        raise ValueError("BMP frame pixel count mismatch")
    mean = sum(index * count for index, count in enumerate(hist)) / pixel_count
    variance = sum(((index - mean) ** 2) * count for index, count in enumerate(hist)) / pixel_count
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "q90": _hist_quantile(hist, 0.9),
        "max": float(next(index for index in range(255, -1, -1) if hist[index])),
    }


def _hist_quantile(hist: list[int], quantile: float) -> float:
    threshold = quantile * (sum(hist) - 1)
    cumulative = 0
    for index, count in enumerate(hist):
        cumulative += count
        if cumulative - 1 >= threshold:
            return float(index)
    return 255.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _melt_pool_sequence_stats(
    *,
    archive: zipfile.ZipFile,
    group: dict[str, Any],
    max_frames_per_layer: int,
) -> dict[str, Any]:
    layer = int(group["target_layer_index"])
    width = int(group.get("zero_padded_width") or 5)
    frame_indices = _sample_frame_indices(
        int(group["first_frame_index"]),
        int(group["last_frame_index"]),
        int(group["frame_count"]),
        max_frames_per_layer,
    )
    frame_stats: list[dict[str, float]] = []
    for frame_index in frame_indices:
        member_name = (
            f"In-situ Meas Data/Melt Pool Camera/MIA_L{layer:04d}/"
            f"frame{frame_index:0{width}d}.bmp"
        )
        frame_stats.append(_bmp_frame_stats(archive.read(member_name)))
    means = [row["mean"] for row in frame_stats]
    q90s = [row["q90"] for row in frame_stats]
    maxs = [row["max"] for row in frame_stats]
    peak_position = 0.0
    if means and len(means) > 1:
        peak_index = max(range(len(means)), key=lambda index: means[index])
        peak_position = peak_index / (len(means) - 1)
    return {
        "melt_pool_frame_count": int(group["frame_count"]),
        "sampled_frame_count": len(frame_indices),
        "first_sampled_frame_index": frame_indices[0] if frame_indices else 0,
        "last_sampled_frame_index": frame_indices[-1] if frame_indices else 0,
        "target_mp_mean_mean": _mean(means),
        "target_mp_mean_std": _std(means),
        "target_mp_q90_mean": _mean(q90s),
        "target_mp_max_mean": _mean(maxs),
        "target_mp_max_range": max(maxs) - min(maxs) if maxs else 0.0,
        "target_mp_temporal_mean_range": max(means) - min(means) if means else 0.0,
        "target_mp_early_late_mean_delta": (means[-1] - means[0]) if len(means) >= 2 else 0.0,
        "target_mp_peak_frame_position": peak_position,
    }


def build_melt_pool_rows(
    *,
    numeric_rows: list[dict[str, str]],
    join_rows: list[dict[str, str]],
    sequence_rows: list[dict[str, str]],
    data_root: Path,
    discover_from_zip: bool,
    max_frames_per_layer: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    join = _melt_pool_join(join_rows)
    if join is None:
        return [], {"status": "missing_melt_pool_join"}
    groups = _groups_from_zip(data_root) if discover_from_zip else _groups_from_sequence_csv(sequence_rows)
    by_source_layer = _source_rows_by_layer(numeric_rows)
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(data_root / "In-situ Meas Data.zip") as archive:
        for target_layer in sorted(groups):
            source_layer = target_layer + int(join["source_minus_target_offset"])
            source = by_source_layer.get(source_layer)
            if source is None:
                continue
            stats = _melt_pool_sequence_stats(
                archive=archive,
                group=groups[target_layer],
                max_frames_per_layer=max_frames_per_layer,
            )
            rows.append(
                {
                    "x": _float(source, "x"),
                    "y": _float(source, "y"),
                    "t": float(source_layer),
                    "source_layer_index": source_layer,
                    "target_layer_index": target_layer,
                    **stats,
                    "source_p_mean": _float(source, "source_p_mean"),
                    "source_p_nonzero_fraction": _float(source, "source_p_nonzero_fraction"),
                    "source_x_range": _float(source, "source_x_range"),
                    "source_y_range": _float(source, "source_y_range"),
                    "source_p_range": _float(source, "source_p_range"),
                    "source_t_range": _float(source, "source_t_range"),
                    "source_rows_read": _float(source, "source_rows_read"),
                    "row_id": f"melt_pool_layer::{target_layer:04d}",
                    "target_group_key": str(groups[target_layer]["group_key"]),
                }
            )
    return rows, {
        "status": "melt_pool_rows_built",
        "source_minus_target_offset": int(join["source_minus_target_offset"]),
        "melt_pool_group_count": len(groups),
        "matched_row_count": len(rows),
        "discover_from_zip": discover_from_zip,
    }


def _split_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    splits = {"train": [], "val": [], "test": []}
    for index, _row in enumerate(sorted(rows, key=lambda row: int(row["source_layer_index"]))):
        bucket = index % 5
        if bucket in {0, 1, 2}:
            split = "train"
        elif bucket == 3:
            split = "val"
        else:
            split = "test"
        _row["split_name"] = split
    for index, row in enumerate(rows):
        splits[str(row["split_name"])].append(index)
    by_source: dict[int, set[str]] = {}
    for row in rows:
        by_source.setdefault(int(row["source_layer_index"]), set()).add(str(row["split_name"]))
    return {
        "splits": splits,
        "split_counts": {split: len(indices) for split, indices in splits.items()},
        "leakage_group": "source_layer_index",
        "leakage_safe": all(len(split_names) == 1 for split_names in by_source.values()),
        "row_count": len(rows),
        "group_count": len(by_source),
        "split_policy": "deterministic_source_layer_round_robin_60_20_20",
    }


def _region_rmse(split_payload: dict[str, Any], name: str) -> float | None:
    region = (split_payload.get("region_metrics") or {}).get(name) or {}
    metrics = region.get("metrics") or {}
    value = metrics.get("rmse")
    return float(value) if isinstance(value, (int, float)) else None


def _metric_rows(payloads: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target, target_payloads in payloads.items():
        for profile_method, payload in target_payloads.items():
            profile, method = profile_method.split("::", 1)
            result = payload["results"][0]
            for split, split_payload in result["split_metrics"].items():
                metrics = split_payload["metrics"]
                rows.append(
                    {
                        "target": target,
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


def _metric_index(metric_rows: list[dict[str, Any]], metric: str) -> dict[tuple[str, str, str, str], float]:
    index: dict[tuple[str, str, str, str], float] = {}
    for row in metric_rows:
        value = row.get(metric)
        if isinstance(value, (int, float)):
            index[
                (
                    str(row["target"]),
                    str(row["feature_profile"]),
                    str(row["method"]),
                    str(row["split"]),
                )
            ] = float(value)
    return index


def _best_metric(
    index: dict[tuple[str, str, str, str], float],
    *,
    target: str,
    profiles: tuple[str, ...],
    methods: tuple[str, ...],
    split: str,
) -> tuple[float, str, str]:
    candidates = [
        (value, profile, method)
        for (row_target, profile, method, row_split), value in index.items()
        if row_target == target
        and row_split == split
        and profile in profiles
        and method in methods
    ]
    if not candidates:
        raise ValueError(f"No metric rows for {target}/{profiles}/{split}")
    return min(candidates, key=lambda item: item[0])


def _target_ranges(rows: list[dict[str, Any]], target_columns: tuple[str, ...]) -> dict[str, dict[str, float]]:
    ranges: dict[str, dict[str, float]] = {}
    for target in target_columns:
        values = [float(row[target]) for row in rows if row.get(target) not in (None, "")]
        if not values:
            raise ValueError(f"No numeric values found for target column {target}")
        ranges[target] = {
            "target_min": min(values),
            "target_max": max(values),
            "target_range": max(values) - min(values),
        }
    return ranges


def _review_rows(
    *,
    field_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    target_columns: tuple[str, ...],
    target_priority: tuple[str, ...],
    min_validation_relative_improvement: float,
    min_unsolved_validation_normalized_rmse: float,
    min_shortcut_val_rmse_delta: float,
) -> list[dict[str, Any]]:
    rmse = _metric_index(metric_rows, "rmse")
    normalized = _metric_index(metric_rows, "normalized_rmse")
    ranges = _target_ranges(field_rows, target_columns)
    priority = {target: index for index, target in enumerate(target_priority)}
    rows: list[dict[str, Any]] = []
    for target in target_columns:
        selected_val, selected_profile, selected_method = _best_metric(
            rmse,
            target=target,
            profiles=PRIMARY_PROFILES,
            methods=tuple(method for method in METHODS if method != "mean"),
            split="val",
        )
        selected_test = rmse[(target, selected_profile, selected_method, "test")]
        selected_val_norm = normalized.get((target, selected_profile, selected_method, "val"))
        selected_test_norm = normalized.get((target, selected_profile, selected_method, "test"))
        mean_val = rmse[(target, "mean_guard", "mean", "val")]
        mean_test = rmse[(target, "mean_guard", "mean", "test")]
        mean_val_norm = normalized.get((target, "mean_guard", "mean", "val"))
        mean_test_norm = normalized.get((target, "mean_guard", "mean", "test"))
        shortcut_val, _shortcut_profile, _shortcut_method = _best_metric(
            rmse,
            target=target,
            profiles=SHORTCUT_PROFILES,
            methods=tuple(method for method in METHODS if method != "mean"),
            split="val",
        )
        shortcut_val_norm = normalized.get((target, _shortcut_profile, _shortcut_method, "val"))
        target_range = ranges[target]["target_range"]
        zero_variance = abs(target_range) <= 1e-12
        val_relative = (mean_val - selected_val) / mean_val if mean_val > 0 else 0.0
        test_relative = (mean_test - selected_test) / mean_test if mean_test > 0 else 0.0
        visible_gap = selected_method != "mean" and val_relative >= min_validation_relative_improvement
        solved = (
            isinstance(selected_val_norm, float)
            and selected_val_norm < min_unsolved_validation_normalized_rmse
        )
        shortcut = shortcut_val <= selected_val + min_shortcut_val_rmse_delta
        if zero_variance:
            status = "blocked_zero_variance_target"
        elif not visible_gap:
            status = "blocked_no_baseline_visible_gap"
        elif shortcut:
            status = "blocked_layer_time_shortcut"
        elif solved:
            status = "blocked_strong_baseline_solved_validation_target"
        else:
            status = "candidate_melt_pool_target_gap_ready_for_focused_review"
        rows.append(
            {
                "target": target,
                **ranges[target],
                "selected_feature_profile": selected_profile,
                "selected_validation_method": selected_method,
                "selected_validation_rmse": selected_val,
                "selected_validation_normalized_rmse": selected_val_norm,
                "selected_test_rmse": selected_test,
                "selected_test_normalized_rmse": selected_test_norm,
                "mean_validation_rmse": mean_val,
                "mean_validation_normalized_rmse": mean_val_norm,
                "mean_test_rmse": mean_test,
                "mean_test_normalized_rmse": mean_test_norm,
                "layer_time_validation_rmse": shortcut_val,
                "layer_time_validation_normalized_rmse": shortcut_val_norm,
                "layer_time_shortcut_detected": shortcut,
                "validation_relative_improvement_over_mean": val_relative,
                "test_relative_improvement_over_mean": test_relative,
                "baseline_visible_gap": visible_gap,
                "strong_baseline_solved": solved,
                "zero_variance_target": zero_variance,
                "physical_priority": priority.get(target, len(priority)),
                "status": status,
                "phase112_candidate": status
                == "candidate_melt_pool_target_gap_ready_for_focused_review",
            }
        )
    return rows


def _build_gate(
    *,
    join_gate: dict[str, Any],
    split_manifest: dict[str, Any],
    review_rows: list[dict[str, Any]],
    row_build: dict[str, Any],
    min_rows_for_review: int,
) -> dict[str, Any]:
    candidates = [row for row in review_rows if row["phase112_candidate"]]
    candidates.sort(
        key=lambda row: (
            int(row["physical_priority"]),
            -float(row["selected_validation_normalized_rmse"] or 0.0),
            float(row["selected_validation_rmse"]),
        )
    )
    selected = candidates[0] if candidates else None
    row_count = int(split_manifest.get("row_count", 0))
    shortcut_count = sum(1 for row in review_rows if row.get("layer_time_shortcut_detected"))
    empty_split = any(int(count) == 0 for count in (split_manifest.get("split_counts") or {}).values())
    if not bool(join_gate.get("melt_pool_layer_join_ready")):
        status = "phase112_melt_pool_target_gate_blocked_no_join"
        next_action = "repair Phase 103 melt-pool layer join evidence before target review"
    elif row_count < min_rows_for_review:
        status = "phase112_melt_pool_target_gate_closed_sample_size_limited"
        next_action = "close as diagnostic unless a larger registered melt-pool table is built"
    elif empty_split or not bool(split_manifest.get("leakage_safe")):
        status = "phase112_melt_pool_target_gate_closed_split_limited"
        next_action = "close as diagnostic unless a leakage-safe train/val/test melt-pool split is available"
    elif selected:
        status = "phase112_melt_pool_target_gap_ready_focused_review"
        next_action = f"review {selected['target']} and its source profile before any model training"
    elif shortcut_count:
        status = "phase112_melt_pool_target_gate_closed_layer_time_shortcut"
        next_action = "close melt-pool targets as layer/time shortcut diagnostics; do not train"
    else:
        status = "phase112_melt_pool_target_gate_closed_no_baseline_gap"
        next_action = "close as no-training target diagnostic; do not train"
    return {
        "status": status,
        "phase103_join_status": join_gate.get("status"),
        "melt_pool_layer_join_ready": bool(join_gate.get("melt_pool_layer_join_ready")),
        "melt_pool_row_build": row_build,
        "row_count": row_count,
        "split_counts": split_manifest.get("split_counts", {}),
        "leakage_safe": bool(split_manifest.get("leakage_safe")),
        "empty_split_detected": empty_split,
        "target_columns": list(TARGET_COLUMNS),
        "candidate_targets": [row["target"] for row in candidates],
        "selected_target": selected["target"] if selected else None,
        "selected_feature_profile": selected["selected_feature_profile"] if selected else None,
        "selected_validation_method": selected["selected_validation_method"] if selected else None,
        "selected_validation_rmse": selected["selected_validation_rmse"] if selected else None,
        "selected_validation_normalized_rmse": selected["selected_validation_normalized_rmse"] if selected else None,
        "selected_test_rmse": selected["selected_test_rmse"] if selected else None,
        "selected_test_normalized_rmse": selected["selected_test_normalized_rmse"] if selected else None,
        "layer_time_shortcut_target_count": shortcut_count,
        "phase112_focused_review_allowed": bool(selected),
        "phase112_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
    }


def _write_markdown(path: Path, gate: dict[str, Any], review_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 112 NIST AMMT Melt-Pool Target Gate",
        "",
        f"- Status: `{gate['status']}`",
        f"- Row count: `{gate['row_count']}`",
        f"- Selected target: `{gate['selected_target']}`",
        f"- Focused review allowed: `{gate['phase112_focused_review_allowed']}`",
        "- Model training allowed: `false`",
        "- A100 training allowed now: `false`",
        "",
        "| Target | Status | Profile | Method | Val RMSE | Val NRMSE | Layer/time val RMSE |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in review_rows:
        lines.append(
            "| {target} | {status} | {selected_feature_profile} | {selected_validation_method} | {selected_validation_rmse} | {selected_validation_normalized_rmse} | {layer_time_validation_rmse} |".format(
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
    join_candidates_csv: Path,
    join_gate_path: Path,
    sequence_groups_csv: Path,
    output_dir: Path,
    discover_from_zip: bool,
    max_frames_per_layer: int,
    target_columns: tuple[str, ...],
    target_priority: tuple[str, ...],
    min_validation_relative_improvement: float,
    min_unsolved_validation_normalized_rmse: float,
    min_shortcut_val_rmse_delta: float,
    min_rows_for_review: int,
    n_neighbors: int,
    n_estimators: int,
) -> dict[str, Any]:
    join_gate = _read_json(join_gate_path)
    field_rows, row_build = build_melt_pool_rows(
        numeric_rows=_read_csv(numeric_field_table),
        join_rows=_read_csv(join_candidates_csv),
        sequence_rows=_read_csv(sequence_groups_csv) if sequence_groups_csv.exists() else [],
        data_root=data_root,
        discover_from_zip=discover_from_zip,
        max_frames_per_layer=max_frames_per_layer,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    field_table_path = output_dir / "phase112_nist_ammt_melt_pool_target_field_table.csv"
    split_path = output_dir / "phase112_nist_ammt_melt_pool_target_split_manifest.json"
    metric_path = output_dir / "phase112_nist_ammt_melt_pool_target_metric_table.csv"
    review_path = output_dir / "phase112_nist_ammt_melt_pool_target_review_table.csv"
    gate_path = output_dir / "phase112_nist_ammt_melt_pool_target_gate.json"
    markdown_path = output_dir / "phase112_nist_ammt_melt_pool_target_summary.md"
    manifest_path = output_dir / "phase112_nist_ammt_melt_pool_target_manifest.json"

    if not field_rows:
        split_manifest = {
            "splits": {"train": [], "val": [], "test": []},
            "split_counts": {"train": 0, "val": 0, "test": 0},
            "leakage_group": "source_layer_index",
            "leakage_safe": False,
            "row_count": 0,
            "group_count": 0,
        }
        review_rows: list[dict[str, Any]] = []
        metric_rows: list[dict[str, Any]] = []
    else:
        split_manifest = _split_manifest(field_rows)
        _write_csv(field_table_path, field_rows, FIELD_COLUMNS)
        _write_json(split_path, split_manifest)
        split_counts = split_manifest.get("split_counts") or {}
        review_ready = (
            bool(join_gate.get("melt_pool_layer_join_ready"))
            and len(field_rows) >= min_rows_for_review
            and bool(split_manifest.get("leakage_safe"))
            and all(int(split_counts.get(split, 0)) > 0 for split in ("train", "val", "test"))
        )
        if review_ready:
            payloads: dict[str, dict[str, dict[str, Any]]] = {}
            for target in target_columns:
                payloads[target] = {}
                payloads[target]["mean_guard::mean"] = {
                    "target": target,
                    "strategy": "mean",
                    "feature_profile": "mean_guard",
                    "results": [
                        evaluate_table(
                            table_path=field_table_path,
                            target=target,
                            strategy="mean",
                            split_manifest_path=split_path,
                            fit_split="train",
                            hot_quantiles=[0.9],
                            gradient_quantiles=[0.9],
                        )
                    ],
                }
                for profile, features in FEATURE_PROFILES.items():
                    for method in tuple(method for method in METHODS if method != "mean"):
                        payloads[target][f"{profile}::{method}"] = {
                            "target": target,
                            "strategy": method,
                            "feature_profile": profile,
                            "feature_columns": list(features),
                            "results": [
                                evaluate_table(
                                    table_path=field_table_path,
                                    target=target,
                                    strategy=method,
                                    split_manifest_path=split_path,
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
            review_rows = _review_rows(
                field_rows=field_rows,
                metric_rows=metric_rows,
                target_columns=target_columns,
                target_priority=target_priority,
                min_validation_relative_improvement=min_validation_relative_improvement,
                min_unsolved_validation_normalized_rmse=min_unsolved_validation_normalized_rmse,
                min_shortcut_val_rmse_delta=min_shortcut_val_rmse_delta,
            )
        else:
            metric_rows = []
            review_rows = []
    gate = _build_gate(
        join_gate=join_gate,
        split_manifest=split_manifest,
        review_rows=review_rows,
        row_build=row_build,
        min_rows_for_review=min_rows_for_review,
    )

    if not field_table_path.exists():
        _write_csv(field_table_path, [], FIELD_COLUMNS)
    if not split_path.exists():
        _write_json(split_path, split_manifest)
    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_csv(review_path, review_rows, REVIEW_FIELDS)
    _write_json(gate_path, gate)
    _write_markdown(markdown_path, gate, review_rows)
    manifest = {
        "phase": 112,
        "objective": "nist_ammt_melt_pool_target_representation_gate_no_training",
        "inputs": {
            "data_root": _display_path(data_root, root),
            "numeric_field_table": _display_path(numeric_field_table, root),
            "join_candidates": _display_path(join_candidates_csv, root),
            "join_gate": _display_path(join_gate_path, root),
            "sequence_groups": _display_path(sequence_groups_csv, root),
        },
        "outputs": {
            "field_table": _display_path(field_table_path, root),
            "split_manifest": _display_path(split_path, root),
            "metric_table": _display_path(metric_path, root),
            "review_table": _display_path(review_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown_summary": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "limits": {
            "discover_from_zip": discover_from_zip,
            "max_frames_per_layer": max_frames_per_layer,
            "min_validation_relative_improvement": min_validation_relative_improvement,
            "min_unsolved_validation_normalized_rmse": min_unsolved_validation_normalized_rmse,
            "min_shortcut_val_rmse_delta": min_shortcut_val_rmse_delta,
            "min_rows_for_review": min_rows_for_review,
        },
        "counts": {
            "rows": len(field_rows),
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
        "--join-candidates",
        type=Path,
        default=Path(
            "docs/results/phase103_nist_ammt_registered_intake/"
            "phase103_nist_ammt_source_target_join_candidates.csv"
        ),
    )
    parser.add_argument(
        "--join-gate",
        type=Path,
        default=Path(
            "docs/results/phase103_nist_ammt_registered_intake/"
            "phase103_nist_ammt_join_probe_gate.json"
        ),
    )
    parser.add_argument(
        "--sequence-groups",
        type=Path,
        default=Path(
            "docs/results/phase103_nist_ammt_registered_intake/"
            "phase103_nist_ammt_deep_sequence_groups.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase112_nist_ammt_melt_pool_target_gate"),
    )
    parser.add_argument("--discover-from-zip", action="store_true")
    parser.add_argument("--max-frames-per-layer", type=int, default=9)
    parser.add_argument("--target-columns", default=",".join(TARGET_COLUMNS))
    parser.add_argument("--target-priority", default=",".join(TARGET_PRIORITY))
    parser.add_argument("--min-validation-relative-improvement", type=float, default=0.05)
    parser.add_argument("--min-unsolved-validation-normalized-rmse", type=float, default=0.2)
    parser.add_argument("--min-shortcut-val-rmse-delta", type=float, default=1e-9)
    parser.add_argument("--min-rows-for-review", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    data_root = args.data_root if args.data_root.is_absolute() else root / args.data_root
    numeric_field_table = (
        args.numeric_field_table if args.numeric_field_table.is_absolute() else root / args.numeric_field_table
    )
    join_candidates = args.join_candidates if args.join_candidates.is_absolute() else root / args.join_candidates
    join_gate = args.join_gate if args.join_gate.is_absolute() else root / args.join_gate
    sequence_groups = args.sequence_groups if args.sequence_groups.is_absolute() else root / args.sequence_groups
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(
        root=root,
        data_root=data_root,
        numeric_field_table=numeric_field_table,
        join_candidates_csv=join_candidates,
        join_gate_path=join_gate,
        sequence_groups_csv=sequence_groups,
        output_dir=output_dir,
        discover_from_zip=args.discover_from_zip,
        max_frames_per_layer=args.max_frames_per_layer,
        target_columns=_split_csv_arg(args.target_columns),
        target_priority=_split_csv_arg(args.target_priority),
        min_validation_relative_improvement=args.min_validation_relative_improvement,
        min_unsolved_validation_normalized_rmse=args.min_unsolved_validation_normalized_rmse,
        min_shortcut_val_rmse_delta=args.min_shortcut_val_rmse_delta,
        min_rows_for_review=args.min_rows_for_review,
        n_neighbors=args.n_neighbors,
        n_estimators=args.n_estimators,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
