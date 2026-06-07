#!/usr/bin/env python3
"""Phase 148 NIST AMMT path-contact graph audit.

Builds CAPL-inspired ordered path/contact/reheat descriptors from registered
AMMT source members, then runs no-training tabular baselines and shortcut
controls before any graph mechanism or neural training is allowed.
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
LAYER_TIME_FEATURES = ("source_layer_index", "t")
CAMERA_LAYER_FEATURES = ("source_layer_index", "t", "target_camera_code")
SCALAR_SOURCE_FEATURES = (
    "x",
    "y",
    "t",
    "source_p_mean",
    "source_p_nonzero_fraction",
    "source_x_range",
    "source_y_range",
    "target_camera_code",
)
ORDER_INSENSITIVE_GRAPH_FEATURES = (
    "path_contact_degree_mean",
    "path_contact_degree_std",
    "path_contact_degree_max",
    "path_nonlocal_contact_fraction",
    "path_contact_component_count_norm",
    "path_contact_largest_component_fraction",
)
ORDERED_PATH_FEATURES = (
    "path_total_length_norm",
    "path_mean_step_norm",
    "path_step_std_norm",
    "path_turn_abs_mean",
    "path_reversal_fraction",
    "path_reheat_count_mean",
    "path_reheat_count_max_norm",
    "path_recent_reheat_fraction",
    "path_reheat_lag_mean_norm",
    "path_reheat_lag_min_norm",
    "path_contact_order_gap_mean_norm",
)
SHUFFLED_PATH_FEATURES = tuple(f"{name}_shuffled" for name in ORDERED_PATH_FEATURES)
PATH_DELTA_FEATURES = tuple(f"{name}_ordered_minus_shuffled" for name in ORDERED_PATH_FEATURES)

FEATURE_PROFILES: dict[str, tuple[str, ...]] = {
    "phase106_guard_replay": BASE_FEATURES,
    "scalar_source_control": SCALAR_SOURCE_FEATURES,
    "layer_time_control": LAYER_TIME_FEATURES,
    "camera_layer_time_control": CAMERA_LAYER_FEATURES,
    "path_contact_graph_shuffled": BASE_FEATURES
    + ORDER_INSENSITIVE_GRAPH_FEATURES
    + SHUFFLED_PATH_FEATURES,
    "path_contact_graph_ordered": BASE_FEATURES
    + ORDER_INSENSITIVE_GRAPH_FEATURES
    + ORDERED_PATH_FEATURES,
    "path_contact_graph_full": BASE_FEATURES
    + ORDER_INSENSITIVE_GRAPH_FEATURES
    + ORDERED_PATH_FEATURES
    + PATH_DELTA_FEATURES,
}
CONTROL_PROFILES = (
    "scalar_source_control",
    "layer_time_control",
    "camera_layer_time_control",
    "path_contact_graph_shuffled",
)
CANDIDATE_PROFILES = ("path_contact_graph_ordered", "path_contact_graph_full")

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
    "profile_role",
    "selected_validation_method",
    "selected_validation_rmse",
    "selected_test_rmse",
    "selected_validation_normalized_rmse",
    "selected_test_normalized_rmse",
    "guard_validation_rmse",
    "guard_test_rmse",
    "best_control_profile",
    "best_control_validation_rmse",
    "best_control_test_rmse",
    "validation_improvement_over_guard",
    "test_improvement_over_guard",
    "validation_improvement_over_control",
    "test_improvement_over_control",
    "validation_improves_guard",
    "test_improves_guard",
    "validation_beats_controls",
    "test_beats_controls",
    "status",
    "phase148_candidate",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
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
        return str(value).lower()
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _display_path(path: Path, root: Path | None = None) -> str:
    try:
        if root is not None:
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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def _resample_indices(n_items: int, max_items: int) -> list[int]:
    if n_items <= max_items:
        return list(range(n_items))
    if max_items <= 1:
        return [0]
    return [
        min(n_items - 1, round(index * (n_items - 1) / (max_items - 1)))
        for index in range(max_items)
    ]


def _active_points(
    source_rows: list[tuple[float, float, float, float]]
) -> list[tuple[float, float, float, float]]:
    active = [row for row in source_rows if row[2] > 0.0]
    return active or source_rows


def _normalize_points(
    rows: list[tuple[float, float, float, float]]
) -> tuple[list[tuple[float, float, float, float, int]], float]:
    xs = [row[0] for row in rows]
    ys = [row[1] for row in rows]
    ts = [row[3] for row in rows]
    x_min = min(xs)
    y_min = min(ys)
    t_min = min(ts)
    x_range = _safe_range(xs)
    y_range = _safe_range(ys)
    t_range = _safe_range(ts)
    diag = math.hypot(x_range, y_range)
    diag = diag if diag > 1e-12 else 1.0
    normalized = [
        (
            (row[0] - x_min) / x_range,
            (row[1] - y_min) / y_range,
            max(row[2], 0.0),
            (row[3] - t_min) / t_range,
            index,
        )
        for index, row in enumerate(rows)
    ]
    return normalized, diag


def _ordered_path_stats(
    points: list[tuple[float, float, float, float, int]], contact_radius: float
) -> dict[str, float]:
    if len(points) < 2:
        return {
            "path_total_length_norm": 0.0,
            "path_mean_step_norm": 0.0,
            "path_step_std_norm": 0.0,
            "path_turn_abs_mean": 0.0,
            "path_reversal_fraction": 0.0,
            "path_reheat_count_mean": 0.0,
            "path_reheat_count_max_norm": 0.0,
            "path_recent_reheat_fraction": 0.0,
            "path_reheat_lag_mean_norm": 0.0,
            "path_reheat_lag_min_norm": 0.0,
            "path_contact_order_gap_mean_norm": 0.0,
        }
    steps: list[float] = []
    vectors: list[tuple[float, float]] = []
    for left, right in zip(points[:-1], points[1:]):
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        steps.append(math.hypot(dx, dy))
        vectors.append((dx, dy))
    turns: list[float] = []
    reversals = 0
    for first, second in zip(vectors[:-1], vectors[1:]):
        first_len = math.hypot(*first)
        second_len = math.hypot(*second)
        if first_len <= 1e-12 or second_len <= 1e-12:
            continue
        dot = (first[0] * second[0] + first[1] * second[1]) / (first_len * second_len)
        dot = min(1.0, max(-1.0, dot))
        angle = math.acos(dot) / math.pi
        turns.append(abs(angle))
        if angle > 0.5:
            reversals += 1

    reheat_counts: list[int] = []
    reheat_lags: list[float] = []
    contact_order_gaps: list[float] = []
    for index, point in enumerate(points):
        count = 0
        best_lag: float | None = None
        for previous_index in range(index):
            previous = points[previous_index]
            distance = math.hypot(point[0] - previous[0], point[1] - previous[1])
            if distance <= contact_radius:
                gap = index - previous_index
                if gap > 1:
                    contact_order_gaps.append(gap / max(1, len(points) - 1))
                count += 1
                lag = abs(point[3] - previous[3])
                if best_lag is None or lag < best_lag:
                    best_lag = lag
        reheat_counts.append(count)
        if best_lag is not None:
            reheat_lags.append(best_lag)
    return {
        "path_total_length_norm": sum(steps),
        "path_mean_step_norm": _mean(steps),
        "path_step_std_norm": _std(steps),
        "path_turn_abs_mean": _mean(turns),
        "path_reversal_fraction": reversals / len(turns) if turns else 0.0,
        "path_reheat_count_mean": _mean([float(value) for value in reheat_counts]),
        "path_reheat_count_max_norm": max(reheat_counts) / max(1, len(points) - 1),
        "path_recent_reheat_fraction": sum(1 for value in reheat_counts if value > 0) / len(points),
        "path_reheat_lag_mean_norm": _mean(reheat_lags),
        "path_reheat_lag_min_norm": min(reheat_lags) if reheat_lags else 1.0,
        "path_contact_order_gap_mean_norm": _mean(contact_order_gaps),
    }


def _contact_graph_stats(
    points: list[tuple[float, float, float, float, int]], contact_radius: float
) -> dict[str, float]:
    n_points = len(points)
    if n_points <= 1:
        return {
            "path_contact_degree_mean": 0.0,
            "path_contact_degree_std": 0.0,
            "path_contact_degree_max": 0.0,
            "path_nonlocal_contact_fraction": 0.0,
            "path_contact_component_count_norm": 1.0,
            "path_contact_largest_component_fraction": 1.0,
        }
    degrees = [0 for _ in points]
    parent = list(range(n_points))
    edge_count = 0
    nonlocal_edges = 0

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i in range(n_points):
        left = points[i]
        for j in range(i + 1, n_points):
            right = points[j]
            if math.hypot(left[0] - right[0], left[1] - right[1]) <= contact_radius:
                degrees[i] += 1
                degrees[j] += 1
                edge_count += 1
                if abs(left[4] - right[4]) > 1:
                    nonlocal_edges += 1
                union(i, j)
    components: dict[int, int] = {}
    for index in range(n_points):
        root = find(index)
        components[root] = components.get(root, 0) + 1
    degree_values = [float(value) for value in degrees]
    return {
        "path_contact_degree_mean": _mean(degree_values) / max(1, n_points - 1),
        "path_contact_degree_std": _std(degree_values) / max(1, n_points - 1),
        "path_contact_degree_max": max(degrees) / max(1, n_points - 1),
        "path_nonlocal_contact_fraction": nonlocal_edges / edge_count if edge_count else 0.0,
        "path_contact_component_count_norm": len(components) / n_points,
        "path_contact_largest_component_fraction": max(components.values()) / n_points,
    }


def _shuffled_points(
    points: list[tuple[float, float, float, float, int]]
) -> list[tuple[float, float, float, float, int]]:
    return sorted(points, key=lambda item: (round(item[0], 6), round(item[1], 6), item[4] % 7))


def _suffix_keys(stats: dict[str, float], suffix: str) -> dict[str, float]:
    return {f"{key}{suffix}": value for key, value in stats.items()}


def _path_contact_stats(
    handle: BinaryIO,
    *,
    max_source_rows: int | None,
    max_graph_points: int,
    contact_radius_fraction: float,
) -> dict[str, Any]:
    source_rows, truncated = _parse_source_rows(handle, max_source_rows)
    active = _active_points(source_rows)
    indices = _resample_indices(len(active), max_graph_points)
    sampled = [active[index] for index in indices]
    points, _diag = _normalize_points(sampled)
    contact_radius = max(1e-6, contact_radius_fraction)
    graph_stats = _contact_graph_stats(points, contact_radius)
    ordered_stats = _ordered_path_stats(points, contact_radius)
    shuffled_stats_raw = _ordered_path_stats(_shuffled_points(points), contact_radius)
    shuffled_stats = _suffix_keys(shuffled_stats_raw, "_shuffled")
    deltas = {
        f"{key}_ordered_minus_shuffled": ordered_stats[key] - shuffled_stats_raw[key]
        for key in ORDERED_PATH_FEATURES
    }
    return {
        **graph_stats,
        **ordered_stats,
        **shuffled_stats,
        **deltas,
        "path_contact_rows_read": len(source_rows),
        "path_contact_active_rows": len(active),
        "path_contact_graph_points": len(points),
        "path_contact_rows_truncated": truncated,
        "path_contact_radius_fraction": contact_radius_fraction,
    }


def build_path_contact_rows(
    *,
    spatial_rows: list[dict[str, str]],
    data_root: Path,
    max_source_rows: int | None,
    max_graph_points: int,
    contact_radius_fraction: float,
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
                    cache[source_member] = _path_contact_stats(
                        handle,
                        max_source_rows=max_source_rows,
                        max_graph_points=max_graph_points,
                        contact_radius_fraction=contact_radius_fraction,
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


def _best_by_profile(
    rmse: dict[tuple[str, str, str], float],
    normalized: dict[tuple[str, str, str], float],
    profile: str,
) -> dict[str, Any]:
    val_candidates = [
        (rmse[(profile, method, "val")], method)
        for method in METHODS
        if (profile, method, "val") in rmse
    ]
    if not val_candidates:
        raise ValueError(f"No validation metrics found for {profile}")
    selected_val, selected_method = min(val_candidates)
    return {
        "selected_validation_method": selected_method,
        "selected_validation_rmse": selected_val,
        "selected_test_rmse": rmse[(profile, selected_method, "test")],
        "selected_validation_normalized_rmse": normalized.get((profile, selected_method, "val")),
        "selected_test_normalized_rmse": normalized.get((profile, selected_method, "test")),
    }


def _profile_role(profile: str) -> str:
    if profile in CANDIDATE_PROFILES:
        return "candidate"
    if profile in CONTROL_PROFILES:
        return "control"
    return "guard_replay"


def _review_rows(
    *,
    metric_rows: list[dict[str, Any]],
    guard_validation_rmse: float,
    guard_test_rmse: float,
    min_validation_improvement: float,
    min_control_margin: float,
) -> list[dict[str, Any]]:
    rmse = _metric_index(metric_rows, "rmse")
    normalized = _metric_index(metric_rows, "normalized_rmse")
    best = {profile: _best_by_profile(rmse, normalized, profile) for profile in FEATURE_PROFILES}
    control_best = min(
        CONTROL_PROFILES,
        key=lambda profile: float(best[profile]["selected_validation_rmse"]),
    )
    control_val = float(best[control_best]["selected_validation_rmse"])
    control_test = float(best[control_best]["selected_test_rmse"])
    rows: list[dict[str, Any]] = []
    for profile in FEATURE_PROFILES:
        selected = best[profile]
        selected_val = float(selected["selected_validation_rmse"])
        selected_test = float(selected["selected_test_rmse"])
        val_gain_guard = guard_validation_rmse - selected_val
        test_gain_guard = guard_test_rmse - selected_test
        val_gain_control = control_val - selected_val
        test_gain_control = control_test - selected_test
        val_improves = val_gain_guard >= min_validation_improvement
        test_improves = test_gain_guard > 0.0
        val_beats_control = val_gain_control >= min_control_margin
        test_beats_control = test_gain_control >= 0.0
        if profile not in CANDIDATE_PROFILES:
            status = "control_or_guard_profile"
            candidate = False
        elif selected["selected_validation_method"] == "mean":
            status = "blocked_mean_selected"
            candidate = False
        elif not val_improves:
            status = "blocked_no_validation_gain_over_phase106_guard"
            candidate = False
        elif not test_improves:
            status = "blocked_validation_test_reversal_against_phase106_guard"
            candidate = False
        elif not val_beats_control:
            status = "blocked_control_dominant_validation"
            candidate = False
        elif not test_beats_control:
            status = "blocked_control_dominant_test"
            candidate = False
        else:
            status = "candidate_path_contact_graph_gap_ready_focused_review"
            candidate = True
        rows.append(
            {
                "feature_profile": profile,
                "profile_role": _profile_role(profile),
                **selected,
                "guard_validation_rmse": guard_validation_rmse,
                "guard_test_rmse": guard_test_rmse,
                "best_control_profile": control_best,
                "best_control_validation_rmse": control_val,
                "best_control_test_rmse": control_test,
                "validation_improvement_over_guard": val_gain_guard,
                "test_improvement_over_guard": test_gain_guard,
                "validation_improvement_over_control": val_gain_control,
                "test_improvement_over_control": test_gain_control,
                "validation_improves_guard": val_improves,
                "test_improves_guard": test_improves,
                "validation_beats_controls": val_beats_control,
                "test_beats_controls": test_beats_control,
                "status": status,
                "phase148_candidate": candidate,
            }
        )
    return rows


def _build_gate(
    *,
    phase147_gate: dict[str, Any],
    phase106_gate: dict[str, Any],
    phase114_gate: dict[str, Any],
    review_rows: list[dict[str, Any]],
    min_validation_improvement: float,
    min_control_margin: float,
    max_source_rows: int | None,
    max_graph_points: int,
    contact_radius_fraction: float,
) -> dict[str, Any]:
    candidates = [row for row in review_rows if row["phase148_candidate"]]
    candidates.sort(key=lambda row: float(row["selected_validation_rmse"]))
    selected = candidates[0] if candidates else None
    phase147_ready = (
        phase147_gate.get("status")
        == "phase147_literature_guided_model_roadmap_ready_phase148_no_training_design"
        and bool(phase147_gate.get("phase148_no_training_design_allowed"))
    )
    phase114_closed = "closed" in str(phase114_gate.get("status", ""))
    target = phase106_gate.get("selected_target") or "target_center_periphery_contrast"
    best_control = min(
        (row for row in review_rows if row["profile_role"] == "control"),
        key=lambda row: float(row["selected_validation_rmse"]),
    )
    if not phase147_ready:
        status = "phase148_path_contact_graph_audit_blocked_by_phase147"
        next_action = "complete Phase 147 literature-guided roadmap first"
    elif not phase114_closed:
        status = "phase148_path_contact_graph_audit_blocked_by_phase114"
        next_action = "close or review Phase 114 G-code strategy source gate before path-contact audit"
    elif selected:
        status = "phase148_path_contact_graph_audit_ready_focused_review"
        next_action = (
            f"review {selected['feature_profile']} on {target} against split/shortcut controls "
            "before any physics-hardcoded graph residual mechanism"
        )
    else:
        status = "phase148_path_contact_graph_audit_closed_no_guarded_graph_gap"
        next_action = (
            "close CAPL-inspired path-contact graph descriptors as diagnostic; try the next "
            "literature route only through a fresh no-training gate"
        )
    return {
        "status": status,
        "target": target,
        "phase147_gate_status": phase147_gate.get("status"),
        "phase106_gate_status": phase106_gate.get("status"),
        "phase106_selected_validation_method": phase106_gate.get("selected_validation_method"),
        "phase106_selected_validation_rmse": phase106_gate.get("selected_validation_rmse"),
        "phase106_selected_test_rmse": phase106_gate.get("selected_test_rmse"),
        "phase114_gate_status": phase114_gate.get("status"),
        "feature_profiles": list(FEATURE_PROFILES),
        "control_profiles": list(CONTROL_PROFILES),
        "candidate_feature_profiles": [row["feature_profile"] for row in candidates],
        "selected_feature_profile": selected["feature_profile"] if selected else None,
        "selected_validation_method": selected["selected_validation_method"] if selected else None,
        "selected_validation_rmse": selected["selected_validation_rmse"] if selected else None,
        "selected_test_rmse": selected["selected_test_rmse"] if selected else None,
        "best_control_profile": best_control["feature_profile"],
        "best_control_validation_rmse": best_control["selected_validation_rmse"],
        "best_control_test_rmse": best_control["selected_test_rmse"],
        "min_validation_improvement": min_validation_improvement,
        "min_control_margin": min_control_margin,
        "max_source_rows_per_member": max_source_rows,
        "max_graph_points_per_member": max_graph_points,
        "contact_radius_fraction": contact_radius_fraction,
        "phase148_focused_review_allowed": bool(selected and phase147_ready and phase114_closed),
        "phase148_model_mechanism_allowed": False,
        "phase148_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
    }


def _write_markdown(path: Path, gate: dict[str, Any], review_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 148 NIST AMMT Path-Contact Graph Audit",
        "",
        f"- Status: `{gate['status']}`",
        f"- Target: `{gate['target']}`",
        f"- Selected feature profile: `{gate['selected_feature_profile']}`",
        f"- Best control profile: `{gate['best_control_profile']}`",
        f"- Focused review allowed: `{gate['phase148_focused_review_allowed']}`",
        "- Model mechanism allowed: `false`",
        "- Model training allowed: `false`",
        "- A100 training allowed now: `false`",
        "",
        "| Feature profile | Role | Status | Method | Val RMSE | Test RMSE | Val gain vs guard | Val gain vs control |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in review_rows:
        lines.append(
            "| {feature_profile} | {profile_role} | {status} | {selected_validation_method} | {selected_validation_rmse} | {selected_test_rmse} | {validation_improvement_over_guard} | {validation_improvement_over_control} |".format(
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
    phase114_gate_path: Path,
    phase147_gate_path: Path,
    output_dir: Path,
    min_validation_improvement: float,
    min_control_margin: float,
    max_source_rows: int | None,
    max_graph_points: int,
    contact_radius_fraction: float,
    n_neighbors: int,
    n_estimators: int,
) -> dict[str, Any]:
    phase147_gate = _read_json(phase147_gate_path)
    phase106_gate = _read_json(phase106_gate_path)
    phase114_gate = _read_json(phase114_gate_path)
    target = str(phase106_gate.get("selected_target") or "target_center_periphery_contrast")
    rows = build_path_contact_rows(
        spatial_rows=_read_csv(spatial_field_table),
        data_root=data_root,
        max_source_rows=max_source_rows,
        max_graph_points=max_graph_points,
        contact_radius_fraction=contact_radius_fraction,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    augmented_path = output_dir / "phase148_nist_ammt_path_contact_graph_augmented_field_table.csv"
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
        min_control_margin=min_control_margin,
    )
    gate = _build_gate(
        phase147_gate=phase147_gate,
        phase106_gate=phase106_gate,
        phase114_gate=phase114_gate,
        review_rows=review_rows,
        min_validation_improvement=min_validation_improvement,
        min_control_margin=min_control_margin,
        max_source_rows=max_source_rows,
        max_graph_points=max_graph_points,
        contact_radius_fraction=contact_radius_fraction,
    )

    metric_path = output_dir / "phase148_nist_ammt_path_contact_graph_metric_table.csv"
    review_path = output_dir / "phase148_nist_ammt_path_contact_graph_review_table.csv"
    gate_path = output_dir / "phase148_nist_ammt_path_contact_graph_audit_gate.json"
    markdown_path = output_dir / "phase148_nist_ammt_path_contact_graph_audit.md"
    manifest_path = output_dir / "phase148_nist_ammt_path_contact_graph_audit_manifest.json"
    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_csv(review_path, review_rows, REVIEW_FIELDS)
    _write_json(gate_path, gate)
    _write_markdown(markdown_path, gate, review_rows)
    manifest = {
        "phase": 148,
        "objective": "nist_ammt_capl_inspired_path_contact_graph_audit_no_training",
        "inputs": {
            "data_root": _display_path(data_root, root),
            "spatial_field_table": _display_path(spatial_field_table, root),
            "split_manifest": _display_path(split_manifest, root),
            "phase106_gate": _display_path(phase106_gate_path, root),
            "phase114_gate": _display_path(phase114_gate_path, root),
            "phase147_gate": _display_path(phase147_gate_path, root),
        },
        "outputs": {
            "augmented_field_table": _display_path(augmented_path, root),
            "metric_table": _display_path(metric_path, root),
            "review_table": _display_path(review_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown_summary": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "limits": {
            "min_validation_improvement": min_validation_improvement,
            "min_control_margin": min_control_margin,
            "max_source_rows_per_member": max_source_rows,
            "max_graph_points_per_member": max_graph_points,
            "contact_radius_fraction": contact_radius_fraction,
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
        "--phase114-gate",
        type=Path,
        default=Path(
            "docs/results/phase114_nist_ammt_gcode_strategy_source_gate/"
            "phase114_nist_ammt_gcode_strategy_source_gate.json"
        ),
    )
    parser.add_argument(
        "--phase147-gate",
        type=Path,
        default=Path(
            "docs/results/phase147_literature_guided_model_roadmap/"
            "phase147_literature_guided_model_roadmap_gate.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase148_nist_ammt_path_contact_graph_audit"),
    )
    parser.add_argument("--min-validation-improvement", type=float, default=0.005)
    parser.add_argument("--min-control-margin", type=float, default=0.005)
    parser.add_argument("--max-source-rows", type=int, default=200000)
    parser.add_argument("--max-graph-points", type=int, default=512)
    parser.add_argument("--contact-radius-fraction", type=float, default=0.08)
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
    split_manifest = args.split_manifest if args.split_manifest.is_absolute() else root / args.split_manifest
    phase106_gate = args.phase106_gate if args.phase106_gate.is_absolute() else root / args.phase106_gate
    phase114_gate = args.phase114_gate if args.phase114_gate.is_absolute() else root / args.phase114_gate
    phase147_gate = args.phase147_gate if args.phase147_gate.is_absolute() else root / args.phase147_gate
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(
        root=root,
        data_root=data_root,
        spatial_field_table=spatial_field_table,
        split_manifest=split_manifest,
        phase106_gate_path=phase106_gate,
        phase114_gate_path=phase114_gate,
        phase147_gate_path=phase147_gate,
        output_dir=output_dir,
        min_validation_improvement=args.min_validation_improvement,
        min_control_margin=args.min_control_margin,
        max_source_rows=args.max_source_rows,
        max_graph_points=args.max_graph_points,
        contact_radius_fraction=args.contact_radius_fraction,
        n_neighbors=args.n_neighbors,
        n_estimators=args.n_estimators,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
