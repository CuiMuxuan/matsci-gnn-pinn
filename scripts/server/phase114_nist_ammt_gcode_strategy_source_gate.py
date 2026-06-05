#!/usr/bin/env python3
"""Phase 114 NIST AMMT G-code strategy source gate.

This no-training gate inspects the strategy-level AM G-code bank in
``Build Command Data.zip`` and asks whether a registered, source-side strategy
representation adds baseline-visible signal for existing Layer Camera targets.
Raw ZIP members stay server-local; outputs are small CSV/JSON/Markdown files.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from gnnpinn.eval.field_baseline import evaluate_table


METHODS = ("mean", "knn", "extra_trees", "hist_gradient_boosting")
TARGET_COLUMNS = (
    "target_intensity_std",
    "target_center_periphery_contrast",
    "target_grid_mean_range",
    "target_quadrant_contrast",
)
TARGET_PRIORITY = (
    "target_center_periphery_contrast",
    "target_intensity_std",
    "target_grid_mean_range",
    "target_quadrant_contrast",
)
FEATURE_PROFILES: dict[str, tuple[str, ...]] = {
    "xypt_guard": (
        "source_p_mean",
        "source_p_nonzero_fraction",
        "source_p_range",
        "source_x_range",
        "source_y_range",
        "source_rows_read",
    ),
    "gcode_layer_path": (
        "gcode_layer_line_count",
        "gcode_layer_motion_length",
        "gcode_layer_laser_on_fraction",
        "gcode_layer_l_mean",
        "gcode_layer_l_range",
        "gcode_layer_f_mean",
        "gcode_layer_turn_score_mean",
    ),
    "gcode_strategy_params": (
        "gcode_strategy_line_count",
        "gcode_strategy_motion_length_mean",
        "gcode_strategy_laser_on_fraction_mean",
        "gcode_mode_island",
        "gcode_mode_laser_power",
        "gcode_mode_laser_path",
        "gcode_hatch_space",
        "gcode_laser_density",
    ),
    "gcode_all": (
        "gcode_strategy_id",
        "gcode_layer_line_count",
        "gcode_layer_motion_length",
        "gcode_layer_laser_on_fraction",
        "gcode_layer_l_mean",
        "gcode_layer_l_range",
        "gcode_layer_f_mean",
        "gcode_layer_turn_score_mean",
        "gcode_strategy_line_count",
        "gcode_strategy_motion_length_mean",
        "gcode_strategy_laser_on_fraction_mean",
        "gcode_mode_island",
        "gcode_mode_laser_power",
        "gcode_mode_laser_path",
        "gcode_hatch_space",
        "gcode_laser_density",
    ),
    "layer_time_strategy": ("t", "source_layer_index", "target_camera_code", "gcode_strategy_id"),
}
PRIMARY_PROFILES = ("gcode_layer_path", "gcode_strategy_params", "gcode_all")
GUARD_PROFILE = "xypt_guard"
SHORTCUT_PROFILES = ("layer_time_strategy",)
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
    "mean_test_rmse",
    "xypt_guard_validation_rmse",
    "xypt_guard_test_rmse",
    "shortcut_validation_rmse",
    "validation_gain_over_mean",
    "validation_gain_over_xypt_guard",
    "test_gain_over_mean",
    "test_gain_over_xypt_guard",
    "baseline_visible_gap",
    "beats_xypt_guard",
    "shortcut_detected",
    "zero_variance_target",
    "physical_priority",
    "status",
    "phase114_candidate",
)
STRATEGY_FIELDS = (
    "strategy_id",
    "strategy_name",
    "gcode_member",
    "generator_member",
    "interpreter_member",
    "gcode_size",
    "gcode_strategy_line_count",
    "gcode_strategy_z_layer_count",
    "gcode_strategy_motion_length_mean",
    "gcode_strategy_laser_on_fraction_mean",
    "gcode_mode_island",
    "gcode_mode_laser_power",
    "gcode_mode_laser_path",
    "gcode_hatch_space",
    "gcode_laser_density",
    "match_source_layer_coverage",
    "mean_abs_source_p_delta",
    "mean_abs_nonzero_fraction_delta",
    "strategy_match_status",
)
FIELD_COLUMNS = (
    "x",
    "y",
    "t",
    "source_layer_index",
    "target_layer_index",
    "target_camera_code",
    "source_p_mean",
    "source_p_nonzero_fraction",
    "source_p_range",
    "source_x_range",
    "source_y_range",
    "source_rows_read",
    "gcode_strategy_id",
    "gcode_layer_line_count",
    "gcode_layer_motion_length",
    "gcode_layer_laser_on_fraction",
    "gcode_layer_l_mean",
    "gcode_layer_l_range",
    "gcode_layer_f_mean",
    "gcode_layer_turn_score_mean",
    "gcode_strategy_line_count",
    "gcode_strategy_motion_length_mean",
    "gcode_strategy_laser_on_fraction_mean",
    "gcode_mode_island",
    "gcode_mode_laser_power",
    "gcode_mode_laser_path",
    "gcode_hatch_space",
    "gcode_laser_density",
    *TARGET_COLUMNS,
    "split_name",
    "row_id",
    "source_member_name",
    "target_member_name",
    "gcode_member_name",
    "gcode_strategy_name",
)
BASELINE_SPLITS = ("train", "val", "test")
COMMAND_RE = re.compile(r"(?:^|\s)([XYZFL])(-?\d+(?:\.\d+)?)")
LAYER_RE = re.compile(r"layer(\d+)\.csv$", re.IGNORECASE)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _range(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def _parse_parameter_csv(text: str) -> dict[str, float]:
    output: dict[str, float] = {}
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        variable = (row.get("Variable") or "").strip()
        value = (row.get("Value") or "").strip()
        if not variable or variable.startswith("%") or not value:
            continue
        try:
            output[variable] = float(value)
        except ValueError:
            continue
    return output


def _command_values(line: str) -> dict[str, float]:
    return {key: float(value) for key, value in COMMAND_RE.findall(line)}


def _layer_summary(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "gcode_layer_line_count": 0.0,
            "gcode_layer_motion_length": 0.0,
            "gcode_layer_laser_on_fraction": 0.0,
            "gcode_layer_l_mean": 0.0,
            "gcode_layer_l_range": 0.0,
            "gcode_layer_f_mean": 0.0,
            "gcode_layer_turn_score_mean": 0.0,
        }
    motion_length = 0.0
    turn_scores: list[float] = []
    prev_x: float | None = None
    prev_y: float | None = None
    prev_dx: float | None = None
    prev_dy: float | None = None
    l_values = [float(row["l"]) for row in rows]
    f_values = [float(row["f"]) for row in rows if "f" in row]
    for row in rows:
        x = float(row["x"])
        y = float(row["y"])
        if prev_x is not None and prev_y is not None:
            dx = x - prev_x
            dy = y - prev_y
            dist = math.hypot(dx, dy)
            motion_length += dist
            if prev_dx is not None and prev_dy is not None:
                previous_norm = math.hypot(prev_dx, prev_dy)
                current_norm = math.hypot(dx, dy)
                if previous_norm > 0.0 and current_norm > 0.0:
                    dot = (prev_dx * dx + prev_dy * dy) / (previous_norm * current_norm)
                    dot = max(-1.0, min(1.0, dot))
                    turn_scores.append(abs(math.atan2(math.sqrt(max(0.0, 1.0 - dot * dot)), dot)))
            prev_dx = dx
            prev_dy = dy
        prev_x = x
        prev_y = y
    return {
        "gcode_layer_line_count": float(len(rows)),
        "gcode_layer_motion_length": motion_length,
        "gcode_layer_laser_on_fraction": sum(1 for value in l_values if value > 0.0) / len(l_values),
        "gcode_layer_l_mean": _mean(l_values),
        "gcode_layer_l_range": _range(l_values),
        "gcode_layer_f_mean": _mean(f_values),
        "gcode_layer_turn_score_mean": _mean(turn_scores),
    }


def _parse_gcode_layers(text: str) -> dict[int, dict[str, float]]:
    x = y = z = f = l = None
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    for line in text.splitlines():
        values = _command_values(line)
        if not values:
            continue
        if "X" in values:
            x = values["X"]
        if "Y" in values:
            y = values["Y"]
        if "Z" in values:
            z = values["Z"]
        if "F" in values:
            f = values["F"]
        if "L" in values:
            l = values["L"]
        if x is None or y is None:
            continue
        layer = 0 if z is None else int(round(float(z) / 0.02))
        row = {"x": float(x), "y": float(y), "l": float(l or 0.0)}
        if f is not None:
            row["f"] = float(f)
        grouped[layer].append(row)
    return {layer: _layer_summary(rows) for layer, rows in grouped.items()}


def _strategy_id(strategy_name: str) -> int:
    match = re.match(r"^(\d+)", strategy_name)
    if match:
        return int(match.group(1))
    return 0


def _strategy_name(member: str) -> str:
    parts = member.split("/")
    try:
        index = parts.index("AM Gcode")
        return parts[index + 1]
    except (ValueError, IndexError):
        return ""


def _discover_gcode_strategies(
    *,
    archive: zipfile.ZipFile,
) -> dict[str, dict[str, Any]]:
    strategies: dict[str, dict[str, Any]] = {}
    for info in archive.infolist():
        if not info.filename.startswith("Build Command Data/AM Gcode/"):
            continue
        name = _strategy_name(info.filename)
        if not name:
            continue
        entry = strategies.setdefault(
            name,
            {
                "strategy_name": name,
                "strategy_id": _strategy_id(name),
                "gcode_member": "",
                "generator_member": "",
                "interpreter_member": "",
                "gcode_size": 0,
            },
        )
        filename = Path(info.filename).name
        if filename.endswith(".gcode"):
            entry["gcode_member"] = info.filename
            entry["gcode_size"] = info.file_size
        elif filename.endswith(".csv") and "Generator" in filename:
            entry["generator_member"] = info.filename
        elif filename.endswith(".csv") and "interpreter" in filename.lower():
            entry["interpreter_member"] = info.filename
    return strategies


def build_strategy_bank(
    *,
    data_root: Path,
    build_zip_name: str = "Build Command Data.zip",
) -> tuple[list[dict[str, Any]], dict[str, dict[int, dict[str, float]]]]:
    strategy_rows: list[dict[str, Any]] = []
    layer_features: dict[str, dict[int, dict[str, float]]] = {}
    with zipfile.ZipFile(data_root / build_zip_name) as archive:
        strategies = _discover_gcode_strategies(archive=archive)
        for name in sorted(strategies, key=lambda item: (_strategy_id(item), item)):
            entry = strategies[name]
            if not entry.get("gcode_member"):
                continue
            generator_params = (
                _parse_parameter_csv(archive.read(entry["generator_member"]).decode("utf-8", errors="replace"))
                if entry.get("generator_member")
                else {}
            )
            interpreter_params = (
                _parse_parameter_csv(
                    archive.read(entry["interpreter_member"]).decode("utf-8", errors="replace")
                )
                if entry.get("interpreter_member")
                else {}
            )
            gcode_text = archive.read(entry["gcode_member"]).decode("utf-8", errors="replace")
            layers = _parse_gcode_layers(gcode_text)
            layer_features[name] = layers
            layer_values = list(layers.values())
            row = {
                **entry,
                "gcode_strategy_line_count": sum(
                    float(layer.get("gcode_layer_line_count", 0.0)) for layer in layer_values
                ),
                "gcode_strategy_z_layer_count": len(layers),
                "gcode_strategy_motion_length_mean": _mean(
                    [float(layer.get("gcode_layer_motion_length", 0.0)) for layer in layer_values]
                ),
                "gcode_strategy_laser_on_fraction_mean": _mean(
                    [float(layer.get("gcode_layer_laser_on_fraction", 0.0)) for layer in layer_values]
                ),
                "gcode_mode_island": generator_params.get("mode_island", 0.0),
                "gcode_mode_laser_power": interpreter_params.get("mode_laser_power", 0.0),
                "gcode_mode_laser_path": interpreter_params.get("mode_laser_path", 0.0),
                "gcode_hatch_space": generator_params.get("hatch_space", 0.0),
                "gcode_laser_density": interpreter_params.get("laser_density", 0.0),
            }
            strategy_rows.append(row)
    return strategy_rows, layer_features


def _source_layer_from_member(member_name: str) -> int | None:
    match = LAYER_RE.search(member_name)
    return int(match.group(1)) if match else None


def _best_strategy_for_row(
    *,
    row: dict[str, str],
    strategy_rows: list[dict[str, Any]],
    layer_features: dict[str, dict[int, dict[str, float]]],
) -> tuple[dict[str, Any] | None, dict[str, float] | None, float]:
    source_layer = _source_layer_from_member(row.get("source_member_name", ""))
    if source_layer is None:
        source_layer = _int(row, "source_layer_index")
    source_p = _float(row, "source_p_mean")
    source_nonzero = _float(row, "source_p_nonzero_fraction")
    best: tuple[float, dict[str, Any], dict[str, float]] | None = None
    for strategy in strategy_rows:
        strategy_name = str(strategy["strategy_name"])
        layer = layer_features.get(strategy_name, {}).get(source_layer)
        if layer is None:
            continue
        score = abs(source_p - float(layer.get("gcode_layer_l_mean", 0.0))) / max(abs(source_p), 1.0)
        score += abs(source_nonzero - float(layer.get("gcode_layer_laser_on_fraction", 0.0)))
        if best is None or score < best[0]:
            best = (score, strategy, layer)
    if best is None:
        return None, None, math.inf
    return best[1], best[2], best[0]


def build_gcode_field_rows(
    *,
    registered_rows: list[dict[str, str]],
    strategy_rows: list[dict[str, Any]],
    layer_features: dict[str, dict[int, dict[str, float]]],
    target_columns: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    matched_by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    match_scores: list[float] = []
    missing = 0
    for row in registered_rows:
        strategy, layer, score = _best_strategy_for_row(
            row=row, strategy_rows=strategy_rows, layer_features=layer_features
        )
        if strategy is None or layer is None:
            missing += 1
            continue
        match_scores.append(score)
        target_values = {target: _float(row, target) for target in target_columns}
        combined = {
            "x": _float(row, "x"),
            "y": _float(row, "y"),
            "t": _float(row, "t"),
            "source_layer_index": _int(row, "source_layer_index"),
            "target_layer_index": _int(row, "target_layer_index"),
            "target_camera_code": _int(row, "target_camera_code", -1),
            "source_p_mean": _float(row, "source_p_mean"),
            "source_p_nonzero_fraction": _float(row, "source_p_nonzero_fraction"),
            "source_p_range": _float(row, "source_p_range"),
            "source_x_range": _float(row, "source_x_range"),
            "source_y_range": _float(row, "source_y_range"),
            "source_rows_read": _float(row, "source_rows_read"),
            "gcode_strategy_id": int(strategy["strategy_id"]),
            **layer,
            "gcode_strategy_line_count": float(strategy["gcode_strategy_line_count"]),
            "gcode_strategy_motion_length_mean": float(strategy["gcode_strategy_motion_length_mean"]),
            "gcode_strategy_laser_on_fraction_mean": float(
                strategy["gcode_strategy_laser_on_fraction_mean"]
            ),
            "gcode_mode_island": float(strategy["gcode_mode_island"]),
            "gcode_mode_laser_power": float(strategy["gcode_mode_laser_power"]),
            "gcode_mode_laser_path": float(strategy["gcode_mode_laser_path"]),
            "gcode_hatch_space": float(strategy["gcode_hatch_space"]),
            "gcode_laser_density": float(strategy["gcode_laser_density"]),
            **target_values,
            "split_name": row.get("split_name", ""),
            "row_id": row.get("row_id", ""),
            "source_member_name": row.get("source_member_name", ""),
            "target_member_name": row.get("target_member_name", ""),
            "gcode_member_name": strategy["gcode_member"],
            "gcode_strategy_name": strategy["strategy_name"],
        }
        output.append(combined)
        matched_by_strategy[str(strategy["strategy_name"])].append(combined)

    strategy_review_rows: list[dict[str, Any]] = []
    for strategy in strategy_rows:
        name = str(strategy["strategy_name"])
        matched = matched_by_strategy.get(name, [])
        if matched:
            p_deltas = [
                abs(float(row["source_p_mean"]) - float(row["gcode_layer_l_mean"])) for row in matched
            ]
            nonzero_deltas = [
                abs(float(row["source_p_nonzero_fraction"]) - float(row["gcode_layer_laser_on_fraction"]))
                for row in matched
            ]
            coverage = len({int(row["source_layer_index"]) for row in matched}) / max(
                1, int(strategy.get("gcode_strategy_z_layer_count") or 1)
            )
            status = "matched_registered_source_layers"
        else:
            p_deltas = []
            nonzero_deltas = []
            coverage = 0.0
            status = "not_selected_by_xypt_match"
        strategy_review_rows.append(
            {
                **strategy,
                "match_source_layer_coverage": coverage,
                "mean_abs_source_p_delta": _mean(p_deltas),
                "mean_abs_nonzero_fraction_delta": _mean(nonzero_deltas),
                "strategy_match_status": status,
            }
        )
    match_summary = {
        "registered_row_count": len(registered_rows),
        "matched_row_count": len(output),
        "missing_match_count": missing,
        "strategy_count": len(strategy_rows),
        "matched_strategy_count": len(matched_by_strategy),
        "mean_match_score": _mean(match_scores),
        "max_match_score": max(match_scores) if match_scores else None,
    }
    return output, match_summary, strategy_review_rows


def _split_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    splits = {
        split: [index for index, row in enumerate(rows) if row["split_name"] == split]
        for split in BASELINE_SPLITS
    }
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
    min_xypt_guard_relative_improvement: float,
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
        xypt_guard_val, _guard_profile, _guard_method = _best_metric(
            rmse,
            target=target,
            profiles=(GUARD_PROFILE,),
            methods=tuple(method for method in METHODS if method != "mean"),
            split="val",
        )
        xypt_guard_test = rmse[(target, _guard_profile, _guard_method, "test")]
        shortcut_val, _shortcut_profile, _shortcut_method = _best_metric(
            rmse,
            target=target,
            profiles=SHORTCUT_PROFILES,
            methods=tuple(method for method in METHODS if method != "mean"),
            split="val",
        )
        target_range = ranges[target]["target_range"]
        zero_variance = abs(target_range) <= 1e-12
        val_gain_mean = (mean_val - selected_val) / mean_val if mean_val > 0 else 0.0
        val_gain_guard = (
            (xypt_guard_val - selected_val) / xypt_guard_val if xypt_guard_val > 0 else 0.0
        )
        test_gain_mean = (mean_test - selected_test) / mean_test if mean_test > 0 else 0.0
        test_gain_guard = (
            (xypt_guard_test - selected_test) / xypt_guard_test if xypt_guard_test > 0 else 0.0
        )
        visible_gap = val_gain_mean >= min_validation_relative_improvement
        beats_guard = val_gain_guard >= min_xypt_guard_relative_improvement
        shortcut = shortcut_val <= selected_val + min_shortcut_val_rmse_delta
        if zero_variance:
            status = "blocked_zero_variance_target"
        elif not visible_gap:
            status = "blocked_no_baseline_visible_gap"
        elif not beats_guard:
            status = "blocked_no_gain_over_xypt_guard"
        elif shortcut:
            status = "blocked_layer_time_strategy_shortcut"
        elif test_gain_guard <= 0.0 or test_gain_mean <= 0.0:
            status = "blocked_validation_test_reversal"
        else:
            status = "candidate_gcode_strategy_source_gap_ready_focused_review"
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
                "mean_test_rmse": mean_test,
                "xypt_guard_validation_rmse": xypt_guard_val,
                "xypt_guard_test_rmse": xypt_guard_test,
                "shortcut_validation_rmse": shortcut_val,
                "validation_gain_over_mean": val_gain_mean,
                "validation_gain_over_xypt_guard": val_gain_guard,
                "test_gain_over_mean": test_gain_mean,
                "test_gain_over_xypt_guard": test_gain_guard,
                "baseline_visible_gap": visible_gap,
                "beats_xypt_guard": beats_guard,
                "shortcut_detected": shortcut,
                "zero_variance_target": zero_variance,
                "physical_priority": priority.get(target, len(priority)),
                "status": status,
                "phase114_candidate": status
                == "candidate_gcode_strategy_source_gap_ready_focused_review",
            }
        )
    return rows


def _build_gate(
    *,
    join_gate: dict[str, Any],
    split_manifest: dict[str, Any],
    strategy_summary: dict[str, Any],
    review_rows: list[dict[str, Any]],
    min_rows_for_review: int,
    min_strategy_count: int,
) -> dict[str, Any]:
    candidates = [row for row in review_rows if row["phase114_candidate"]]
    candidates.sort(
        key=lambda row: (
            int(row["physical_priority"]),
            -float(row["validation_gain_over_xypt_guard"] or 0.0),
            float(row["selected_validation_rmse"]),
        )
    )
    selected = candidates[0] if candidates else None
    row_count = int(split_manifest.get("row_count", 0))
    empty_split = any(int(count) == 0 for count in (split_manifest.get("split_counts") or {}).values())
    if not bool(join_gate.get("layer_camera_join_ready")):
        status = "phase114_gcode_strategy_source_gate_blocked_no_layer_join"
        next_action = "repair registered Layer Camera join evidence before G-code strategy review"
    elif int(strategy_summary.get("strategy_count", 0)) < min_strategy_count:
        status = "phase114_gcode_strategy_source_gate_closed_insufficient_strategy_bank"
        next_action = "close as diagnostic unless more G-code strategies are available"
    elif row_count < min_rows_for_review:
        status = "phase114_gcode_strategy_source_gate_closed_sample_size_limited"
        next_action = "close as diagnostic unless a larger registered Layer Camera table is built"
    elif empty_split or not bool(split_manifest.get("leakage_safe")):
        status = "phase114_gcode_strategy_source_gate_closed_split_limited"
        next_action = "close as diagnostic unless a leakage-safe split is available"
    elif selected:
        status = "phase114_gcode_strategy_source_gap_ready_focused_review"
        next_action = f"review {selected['target']} and G-code profile shortcuts before any mechanism design"
    else:
        status = "phase114_gcode_strategy_source_gate_closed_no_guarded_baseline_gap"
        next_action = "close G-code strategy source gate as diagnostic; do not train"
    return {
        "status": status,
        "phase103_join_status": join_gate.get("status"),
        "layer_camera_join_ready": bool(join_gate.get("layer_camera_join_ready")),
        "strategy_summary": strategy_summary,
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
        "selected_test_rmse": selected["selected_test_rmse"] if selected else None,
        "phase114_focused_review_allowed": bool(selected),
        "phase114_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
    }


def _write_markdown(path: Path, gate: dict[str, Any], review_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 114 NIST AMMT G-code Strategy Source Gate",
        "",
        f"- Status: `{gate['status']}`",
        f"- Row count: `{gate['row_count']}`",
        f"- Strategy count: `{gate['strategy_summary'].get('strategy_count', 0)}`",
        f"- Selected target: `{gate['selected_target']}`",
        f"- Focused review allowed: `{gate['phase114_focused_review_allowed']}`",
        "- Model training allowed: `false`",
        "- A100 training allowed now: `false`",
        "",
        "| Target | Status | Profile | Method | Val RMSE | XYPT guard val RMSE | Test gain vs guard |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in review_rows:
        lines.append(
            "| {target} | {status} | {selected_feature_profile} | {selected_validation_method} | {selected_validation_rmse} | {xypt_guard_validation_rmse} | {test_gain_over_xypt_guard} |".format(
                **{key: _csv_value(value) for key, value in row.items()}
            )
        )
    lines.extend(["", f"Next action: {gate['next_action']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_package(
    *,
    root: Path,
    data_root: Path,
    registered_field_table: Path,
    join_gate_path: Path,
    output_dir: Path,
    target_columns: tuple[str, ...],
    target_priority: tuple[str, ...],
    min_rows_for_review: int,
    min_strategy_count: int,
    min_validation_relative_improvement: float,
    min_xypt_guard_relative_improvement: float,
    min_shortcut_val_rmse_delta: float,
    n_neighbors: int,
    n_estimators: int,
) -> dict[str, Any]:
    join_gate = _read_json(join_gate_path)
    registered_rows = _read_csv(registered_field_table)
    strategy_rows, layer_features = build_strategy_bank(data_root=data_root)
    field_rows, match_summary, strategy_review_rows = build_gcode_field_rows(
        registered_rows=registered_rows,
        strategy_rows=strategy_rows,
        layer_features=layer_features,
        target_columns=target_columns,
    )
    split_manifest = _split_manifest(field_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    field_table_path = output_dir / "phase114_nist_ammt_gcode_strategy_field_table.csv"
    split_path = output_dir / "phase114_nist_ammt_gcode_strategy_split_manifest.json"
    strategy_path = output_dir / "phase114_nist_ammt_gcode_strategy_review_table.csv"
    metric_path = output_dir / "phase114_nist_ammt_gcode_strategy_metric_table.csv"
    review_path = output_dir / "phase114_nist_ammt_gcode_strategy_target_review_table.csv"
    gate_path = output_dir / "phase114_nist_ammt_gcode_strategy_source_gate.json"
    markdown_path = output_dir / "phase114_nist_ammt_gcode_strategy_source_gate.md"
    manifest_path = output_dir / "phase114_nist_ammt_gcode_strategy_source_manifest.json"

    _write_csv(field_table_path, field_rows, FIELD_COLUMNS)
    _write_json(split_path, split_manifest)
    _write_csv(strategy_path, strategy_review_rows, STRATEGY_FIELDS)

    split_counts = split_manifest.get("split_counts") or {}
    review_ready = (
        bool(join_gate.get("layer_camera_join_ready"))
        and len(strategy_rows) >= min_strategy_count
        and len(field_rows) >= min_rows_for_review
        and bool(split_manifest.get("leakage_safe"))
        and all(int(split_counts.get(split, 0)) > 0 for split in BASELINE_SPLITS)
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
            min_xypt_guard_relative_improvement=min_xypt_guard_relative_improvement,
            min_shortcut_val_rmse_delta=min_shortcut_val_rmse_delta,
        )
    else:
        metric_rows = []
        review_rows = []

    strategy_summary = {
        **match_summary,
        "strategy_count": len(strategy_rows),
        "target_columns": list(target_columns),
    }
    gate = _build_gate(
        join_gate=join_gate,
        split_manifest=split_manifest,
        strategy_summary=strategy_summary,
        review_rows=review_rows,
        min_rows_for_review=min_rows_for_review,
        min_strategy_count=min_strategy_count,
    )
    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_csv(review_path, review_rows, REVIEW_FIELDS)
    _write_json(gate_path, gate)
    _write_markdown(markdown_path, gate, review_rows)
    manifest = {
        "phase": 114,
        "objective": "nist_ammt_gcode_strategy_source_gate_no_training",
        "inputs": {
            "data_root": _display_path(data_root, root),
            "registered_field_table": _display_path(registered_field_table, root),
            "join_gate": _display_path(join_gate_path, root),
        },
        "outputs": {
            "field_table": _display_path(field_table_path, root),
            "split_manifest": _display_path(split_path, root),
            "strategy_review_table": _display_path(strategy_path, root),
            "metric_table": _display_path(metric_path, root),
            "target_review_table": _display_path(review_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown_summary": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "limits": {
            "min_rows_for_review": min_rows_for_review,
            "min_strategy_count": min_strategy_count,
            "min_validation_relative_improvement": min_validation_relative_improvement,
            "min_xypt_guard_relative_improvement": min_xypt_guard_relative_improvement,
            "min_shortcut_val_rmse_delta": min_shortcut_val_rmse_delta,
        },
        "counts": {
            "rows": len(field_rows),
            "strategies": len(strategy_rows),
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
        "--registered-field-table",
        type=Path,
        default=Path(
            "docs/results/phase106_nist_ammt_spatial_target_representation_gate/"
            "phase106_nist_ammt_spatial_target_field_table.csv"
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
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase114_nist_ammt_gcode_strategy_source_gate"),
    )
    parser.add_argument("--target-columns", default=",".join(TARGET_COLUMNS))
    parser.add_argument("--target-priority", default=",".join(TARGET_PRIORITY))
    parser.add_argument("--min-rows-for-review", type=int, default=100)
    parser.add_argument("--min-strategy-count", type=int, default=4)
    parser.add_argument("--min-validation-relative-improvement", type=float, default=0.03)
    parser.add_argument("--min-xypt-guard-relative-improvement", type=float, default=0.03)
    parser.add_argument("--min-shortcut-val-rmse-delta", type=float, default=1e-9)
    parser.add_argument("--n-neighbors", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    data_root = args.data_root if args.data_root.is_absolute() else root / args.data_root
    registered_field_table = (
        args.registered_field_table
        if args.registered_field_table.is_absolute()
        else root / args.registered_field_table
    )
    join_gate = args.join_gate if args.join_gate.is_absolute() else root / args.join_gate
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(
        root=root,
        data_root=data_root,
        registered_field_table=registered_field_table,
        join_gate_path=join_gate,
        output_dir=output_dir,
        target_columns=_split_csv_arg(args.target_columns),
        target_priority=_split_csv_arg(args.target_priority),
        min_rows_for_review=args.min_rows_for_review,
        min_strategy_count=args.min_strategy_count,
        min_validation_relative_improvement=args.min_validation_relative_improvement,
        min_xypt_guard_relative_improvement=args.min_xypt_guard_relative_improvement,
        min_shortcut_val_rmse_delta=args.min_shortcut_val_rmse_delta,
        n_neighbors=args.n_neighbors,
        n_estimators=args.n_estimators,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
