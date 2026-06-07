#!/usr/bin/env python3
"""Build Phase 151 fixed-grid tensor/split manifest and dense baseline review.

This no-training phase converts Phase 150 indexed dense CSV candidates into
frame-level fixed-grid summary tables, creates leakage-aware split contracts,
and reviews non-neural baselines. It does not train neural operators, read raw
ZIP payloads, or request larger GPUs.
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


DEFAULT_OUTPUT_DIR = Path("docs/results/phase151_fixed_grid_dense_baseline_review")

PHASE_INPUTS = {
    "phase150_gate": Path(
        "docs/results/phase150_dense_tensorization_inventory_gate/"
        "phase150_dense_tensorization_inventory_gate.json"
    ),
    "phase150_inventory": Path(
        "docs/results/phase150_dense_tensorization_inventory_gate/"
        "phase150_dense_source_inventory_table.csv"
    ),
    "phase149_gate": Path(
        "docs/results/phase149_neural_operator_readiness_gate/"
        "phase149_neural_operator_readiness_gate.json"
    ),
    "phase148_gate": Path(
        "docs/results/phase148_nist_ammt_path_contact_graph_audit/"
        "phase148_nist_ammt_path_contact_graph_audit_gate.json"
    ),
}

METHODS = ("mean", "knn", "extra_trees", "hist_gradient_boosting")
TARGET_COLUMNS = (
    "target_frame_mean",
    "target_frame_q90",
    "target_frame_std",
    "target_frame_range",
)
FEATURE_PROFILES: dict[str, tuple[str, ...]] = {
    "time_only": ("x", "t", "frame_index"),
    "process_time": ("x", "t", "frame_index", "laser_power_W", "scan_speed_mm_s", "spot_size_um"),
    "line_process_time": (
        "x",
        "t",
        "frame_index",
        "line_index",
        "laser_power_W",
        "scan_speed_mm_s",
        "spot_size_um",
    ),
}
MODEL_METHODS = ("knn", "extra_trees", "hist_gradient_boosting")

FIXED_GRID_FIELDS = (
    "candidate_id",
    "source_path",
    "x",
    "y",
    "z",
    "t",
    "frame_index",
    "line_id",
    "line_index",
    "dataset_path",
    "laser_power_W",
    "scan_speed_mm_s",
    "spot_size_um",
    "grid_point_count",
    "unique_row_count",
    "unique_col_count",
    "grid_cell_count",
    "coverage_fraction",
    *TARGET_COLUMNS,
)
TENSOR_MANIFEST_FIELDS = (
    "candidate_id",
    "source_path",
    "source_rows",
    "summary_rows",
    "target_column",
    "target_columns",
    "grid_index_columns",
    "split_axis",
    "split_contract_status",
    "leakage_safe_split",
    "train_rows",
    "val_rows",
    "test_rows",
    "tensor_manifest_status",
    "blocker",
)
SPLIT_CONTRACT_FIELDS = (
    "candidate_id",
    "split_axis",
    "split_contract_status",
    "leakage_safe_split",
    "group_count",
    "train_rows",
    "val_rows",
    "test_rows",
    "train_groups",
    "val_groups",
    "test_groups",
    "rationale",
)
METRIC_FIELDS = (
    "candidate_id",
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
)
REVIEW_FIELDS = (
    "candidate_id",
    "target",
    "split_contract_status",
    "selected_feature_profile",
    "selected_method",
    "selected_validation_rmse",
    "selected_validation_normalized_rmse",
    "selected_test_rmse",
    "selected_test_normalized_rmse",
    "mean_validation_rmse",
    "mean_test_rmse",
    "validation_relative_improvement_over_mean",
    "test_relative_improvement_over_mean",
    "baseline_visible_gap",
    "strong_baseline_solved",
    "phase152_low_capacity_dense_design_candidate",
    "status",
    "blocker",
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
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.6f}"
        return ""
    if isinstance(value, (dict, list, tuple)):
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
    try:
        if root is not None:
            return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        pass
    return path.as_posix()


def _is_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value is False
    if isinstance(value, str):
        return value.strip().lower() in {"", "0", "false", "no"}
    return not bool(value)


def _to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int_text(value: Any, default: str = "0") -> str:
    if value in (None, ""):
        return default
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(quantile * len(ordered))) - 1))
    return ordered[index]


def _target_range(rows: list[dict[str, Any]], target: str) -> float:
    values = [_to_float(row.get(target)) for row in rows]
    if not values:
        return 0.0
    return max(values) - min(values)


def _resolve_path(root: Path, path_value: str | Path) -> Path:
    path = path_value if isinstance(path_value, Path) else Path(path_value)
    return path if path.is_absolute() else root / path


def dense_candidates_from_inventory(root: Path, inventory_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in inventory_rows:
        if row.get("source_kind") != "dense_csv":
            continue
        if row.get("present") != "true":
            continue
        if not row.get("tensorization_status", "").startswith("candidate_indexed_dense_csv"):
            continue
        candidates.append(
            {
                "candidate_id": row["candidate_id"],
                "source_path": _resolve_path(root, row["path"]),
                "target_column": row.get("target_column") or "temperature_C",
            }
        )
    return candidates


def _line_index(row: dict[str, str], line_id: str, line_map: dict[str, int]) -> float:
    if row.get("line_index") not in (None, ""):
        return _to_float(row.get("line_index"))
    if line_id not in line_map:
        line_map[line_id] = len(line_map)
    return float(line_map[line_id])


def build_fixed_grid_rows(
    *,
    candidate: dict[str, Any],
    root: Path,
    min_points_per_frame: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _resolve_path(root, candidate["source_path"])
    target = str(candidate.get("target_column", "temperature_C"))
    if not path.exists():
        return [], {
            "candidate_id": candidate["candidate_id"],
            "source_path": _display_path(path, root),
            "source_rows": 0,
            "blocker": "missing_dense_csv",
        }

    groups: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    source_rows = 0
    line_map: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_rows += 1
            if row.get(target) in (None, ""):
                continue
            frame = _to_int_text(row.get("frame_index", row.get("t", "0")))
            line_id = row.get("line_id") or row.get("dataset_path") or "single_line"
            dataset_path = row.get("dataset_path", "")
            laser = str(_to_float(row.get("laser_power_W")))
            speed = str(_to_float(row.get("scan_speed_mm_s")))
            spot = str(_to_float(row.get("spot_size_um")))
            key = (line_id, frame, laser, speed, spot, dataset_path)
            bucket = groups.setdefault(
                key,
                {
                    "values": [],
                    "row_indices": set(),
                    "col_indices": set(),
                    "times": [],
                    "line_index_values": [],
                },
            )
            bucket["values"].append(_to_float(row.get(target)))
            bucket["row_indices"].add(_to_int_text(row.get("row_index", row.get("y", "0"))))
            bucket["col_indices"].add(_to_int_text(row.get("col_index", row.get("x", "0"))))
            bucket["times"].append(_to_float(row.get("t")))
            bucket["line_index_values"].append(_line_index(row, line_id, line_map))

    summary_rows: list[dict[str, Any]] = []
    for key, bucket in sorted(groups.items(), key=lambda item: (item[0][0], int(float(item[0][1])))):
        line_id, frame, laser, speed, spot, dataset_path = key
        values = list(bucket["values"])
        if len(values) < min_points_per_frame:
            continue
        row_count = len(values)
        unique_row_count = len(bucket["row_indices"])
        unique_col_count = len(bucket["col_indices"])
        grid_cell_count = max(1, unique_row_count * unique_col_count)
        line_index = _mean(list(bucket["line_index_values"]))
        summary_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "source_path": _display_path(path, root),
                "x": float(frame),
                "y": line_index,
                "z": 0.0,
                "t": _mean(list(bucket["times"])),
                "frame_index": float(frame),
                "line_id": line_id,
                "line_index": line_index,
                "dataset_path": dataset_path,
                "laser_power_W": _to_float(laser),
                "scan_speed_mm_s": _to_float(speed),
                "spot_size_um": _to_float(spot),
                "grid_point_count": row_count,
                "unique_row_count": unique_row_count,
                "unique_col_count": unique_col_count,
                "grid_cell_count": grid_cell_count,
                "coverage_fraction": row_count / grid_cell_count,
                "target_frame_mean": _mean(values),
                "target_frame_q90": _quantile(values, 0.90),
                "target_frame_std": _std(values),
                "target_frame_range": max(values) - min(values),
            }
        )
    metadata = {
        "candidate_id": candidate["candidate_id"],
        "source_path": _display_path(path, root),
        "source_rows": source_rows,
        "summary_rows": len(summary_rows),
        "target_column": target,
        "blocker": "" if summary_rows else "no_frames_after_min_points_filter",
    }
    return summary_rows, metadata


def _split_groups(sorted_groups: list[str]) -> tuple[set[str], set[str], set[str]]:
    count = len(sorted_groups)
    if count < 3:
        return set(), set(), set()
    train_count = max(1, int(math.floor(count * 0.60)))
    val_count = max(1, int(math.floor(count * 0.20)))
    if train_count + val_count >= count:
        train_count = max(1, count - 2)
        val_count = 1
    train = set(sorted_groups[:train_count])
    val = set(sorted_groups[train_count : train_count + val_count])
    test = set(sorted_groups[train_count + val_count :])
    return train, val, test


def build_split_contract(rows: list[dict[str, Any]], candidate_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    line_groups = sorted({str(row.get("line_id", "")) for row in rows if row.get("line_id") not in (None, "")})
    process_groups = sorted(
        {
            "__".join(
                [
                    f"laser={_csv_value(row.get('laser_power_W'))}",
                    f"speed={_csv_value(row.get('scan_speed_mm_s'))}",
                    f"spot={_csv_value(row.get('spot_size_um'))}",
                ]
            )
            for row in rows
        }
    )
    if len(line_groups) >= 3:
        axis = "line_id"
        groups = line_groups
        train_groups, val_groups, test_groups = _split_groups(groups)
        leakage_safe = True
        status = "leakage_safe_line_group_split"
        rationale = "line_id groups are disjoint across train/val/test"
        group_for_row = lambda row: str(row.get("line_id", ""))
    elif len(process_groups) >= 3:
        axis = "process_tuple"
        groups = process_groups
        train_groups, val_groups, test_groups = _split_groups(groups)
        leakage_safe = True
        status = "leakage_safe_process_group_split"
        rationale = "process tuples are disjoint across train/val/test"

        def group_for_row(row: dict[str, Any]) -> str:
            return "__".join(
                [
                    f"laser={_csv_value(row.get('laser_power_W'))}",
                    f"speed={_csv_value(row.get('scan_speed_mm_s'))}",
                    f"spot={_csv_value(row.get('spot_size_um'))}",
                ]
            )

    elif len(rows) >= 9:
        axis = "frame_block"
        ordered_indices = sorted(range(len(rows)), key=lambda index: _to_float(rows[index].get("frame_index")))
        train_cut = max(1, int(math.floor(len(rows) * 0.60)))
        val_cut = max(train_cut + 1, int(math.floor(len(rows) * 0.80)))
        split_manifest = {
            "sample_id": f"phase151_{candidate_id}_fixed_grid_summary",
            "split_axis": axis,
            "leakage_safe_split": False,
            "split_contract_status": "diagnostic_frame_block_split_only",
            "splits": {
                "train": ordered_indices[:train_cut],
                "val": ordered_indices[train_cut:val_cut],
                "test": ordered_indices[val_cut:],
            },
        }
        contract = {
            "candidate_id": candidate_id,
            "split_axis": axis,
            "split_contract_status": "diagnostic_frame_block_split_only",
            "leakage_safe_split": False,
            "group_count": len(rows),
            "train_rows": len(split_manifest["splits"]["train"]),
            "val_rows": len(split_manifest["splits"]["val"]),
            "test_rows": len(split_manifest["splits"]["test"]),
            "train_groups": "frame_block_train",
            "val_groups": "frame_block_val",
            "test_groups": "frame_block_test",
            "rationale": "single-group source can only be split by frame block; this is diagnostic, not operator-safe",
        }
        return split_manifest, contract
    else:
        split_manifest = {
            "sample_id": f"phase151_{candidate_id}_fixed_grid_summary",
            "split_axis": "insufficient_groups",
            "leakage_safe_split": False,
            "split_contract_status": "blocked_insufficient_groups",
            "splits": {"train": [], "val": [], "test": []},
        }
        contract = {
            "candidate_id": candidate_id,
            "split_axis": "insufficient_groups",
            "split_contract_status": "blocked_insufficient_groups",
            "leakage_safe_split": False,
            "group_count": len(line_groups) or len(process_groups) or len(rows),
            "train_rows": 0,
            "val_rows": 0,
            "test_rows": 0,
            "train_groups": "",
            "val_groups": "",
            "test_groups": "",
            "rationale": "not enough groups or frames for a three-way split",
        }
        return split_manifest, contract

    split_indices = {"train": [], "val": [], "test": []}
    for index, row in enumerate(rows):
        group = group_for_row(row)
        if group in train_groups:
            split_indices["train"].append(index)
        elif group in val_groups:
            split_indices["val"].append(index)
        elif group in test_groups:
            split_indices["test"].append(index)
    split_manifest = {
        "sample_id": f"phase151_{candidate_id}_fixed_grid_summary",
        "split_axis": axis,
        "leakage_safe_split": leakage_safe,
        "split_contract_status": status,
        "splits": split_indices,
    }
    contract = {
        "candidate_id": candidate_id,
        "split_axis": axis,
        "split_contract_status": status,
        "leakage_safe_split": leakage_safe,
        "group_count": len(groups),
        "train_rows": len(split_indices["train"]),
        "val_rows": len(split_indices["val"]),
        "test_rows": len(split_indices["test"]),
        "train_groups": sorted(train_groups),
        "val_groups": sorted(val_groups),
        "test_groups": sorted(test_groups),
        "rationale": rationale,
    }
    return split_manifest, contract


def _metric_row(
    *,
    candidate_id: str,
    target: str,
    feature_profile: str,
    method: str,
    baseline: str,
    split: str,
    split_payload: dict[str, Any],
    target_range: float,
) -> dict[str, Any]:
    metrics = split_payload.get("metrics", {})
    rmse = metrics.get("rmse")
    normalized = float(rmse) / target_range if rmse is not None and target_range > 0 else None
    return {
        "candidate_id": candidate_id,
        "target": target,
        "feature_profile": feature_profile,
        "method": method,
        "baseline": baseline,
        "split": split,
        "n_points": split_payload.get("n_points", 0),
        "rmse": rmse,
        "mae": metrics.get("mae"),
        "relative_l2": metrics.get("relative_l2"),
        "normalized_rmse": normalized,
    }


def _available_feature_columns(rows: list[dict[str, Any]], requested: tuple[str, ...]) -> list[str]:
    if not rows:
        return []
    available = set(rows[0])
    return [column for column in requested if column in available]


def evaluate_candidate_baselines(
    *,
    candidate_id: str,
    rows: list[dict[str, Any]],
    table_path: Path,
    split_manifest_path: Path,
    split_contract: dict[str, Any],
    min_validation_relative_improvement: float,
    min_unsolved_normalized_rmse: float,
    n_estimators: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    leakage_safe = bool(split_contract.get("leakage_safe_split"))

    for target in TARGET_COLUMNS:
        target_range = _target_range(rows, target)
        if target_range <= 0.0:
            review_rows.append(
                {
                    "candidate_id": candidate_id,
                    "target": target,
                    "split_contract_status": split_contract["split_contract_status"],
                    "selected_feature_profile": "",
                    "selected_method": "",
                    "status": "blocked_zero_variance_target",
                    "blocker": "target_range_is_zero",
                    "phase152_low_capacity_dense_design_candidate": False,
                    "baseline_visible_gap": False,
                    "strong_baseline_solved": False,
                }
            )
            continue

        mean_result = evaluate_table(
            table_path=table_path,
            target=target,
            strategy="mean",
            split_manifest_path=split_manifest_path,
            fit_split="train",
        )
        mean_split_metrics = mean_result["split_metrics"]
        for split_name, payload in mean_split_metrics.items():
            metric_rows.append(
                _metric_row(
                    candidate_id=candidate_id,
                    target=target,
                    feature_profile="mean_guard",
                    method="mean",
                    baseline=mean_result["baseline"],
                    split=split_name,
                    split_payload=payload,
                    target_range=target_range,
                )
            )
        mean_val_rmse = mean_split_metrics["val"]["metrics"]["rmse"]
        mean_test_rmse = mean_split_metrics["test"]["metrics"]["rmse"]

        model_candidates: list[dict[str, Any]] = []
        for profile_name, profile_columns in FEATURE_PROFILES.items():
            feature_columns = _available_feature_columns(rows, profile_columns)
            if not feature_columns:
                continue
            for method in MODEL_METHODS:
                result = evaluate_table(
                    table_path=table_path,
                    target=target,
                    strategy=method,
                    split_manifest_path=split_manifest_path,
                    fit_split="train",
                    feature_columns=feature_columns,
                    n_estimators=n_estimators,
                )
                for split_name, payload in result["split_metrics"].items():
                    metric_rows.append(
                        _metric_row(
                            candidate_id=candidate_id,
                            target=target,
                            feature_profile=profile_name,
                            method=method,
                            baseline=result["baseline"],
                            split=split_name,
                            split_payload=payload,
                            target_range=target_range,
                        )
                    )
                model_candidates.append(
                    {
                        "feature_profile": profile_name,
                        "method": method,
                        "validation_rmse": result["split_metrics"]["val"]["metrics"]["rmse"],
                        "test_rmse": result["split_metrics"]["test"]["metrics"]["rmse"],
                    }
                )

        if not model_candidates:
            review_rows.append(
                {
                    "candidate_id": candidate_id,
                    "target": target,
                    "split_contract_status": split_contract["split_contract_status"],
                    "selected_feature_profile": "",
                    "selected_method": "",
                    "status": "blocked_no_model_baseline",
                    "blocker": "no_feature_profile_available",
                    "phase152_low_capacity_dense_design_candidate": False,
                    "baseline_visible_gap": False,
                    "strong_baseline_solved": False,
                    "mean_validation_rmse": mean_val_rmse,
                    "mean_test_rmse": mean_test_rmse,
                }
            )
            continue

        selected = min(model_candidates, key=lambda item: item["validation_rmse"])
        val_gain = (mean_val_rmse - selected["validation_rmse"]) / mean_val_rmse if mean_val_rmse > 0 else 0.0
        test_gain = (mean_test_rmse - selected["test_rmse"]) / mean_test_rmse if mean_test_rmse > 0 else 0.0
        selected_val_norm = selected["validation_rmse"] / target_range
        selected_test_norm = selected["test_rmse"] / target_range
        baseline_visible_gap = val_gain >= min_validation_relative_improvement
        strong_baseline_solved = (
            selected_val_norm <= min_unsolved_normalized_rmse
            and selected_test_norm <= min_unsolved_normalized_rmse
        )
        if not leakage_safe:
            status = "blocked_no_leakage_safe_split"
            blocker = "split is diagnostic only"
        elif not baseline_visible_gap:
            status = "blocked_no_validation_gain_over_mean"
            blocker = "strong baseline did not improve enough on validation"
        elif test_gain < 0.0:
            status = "blocked_validation_test_reversal"
            blocker = "validation-selected baseline worsened test RMSE"
        elif strong_baseline_solved:
            status = "blocked_strong_baseline_solved"
            blocker = "non-neural strong baseline residual is below unsolved threshold"
        else:
            status = "phase151_dense_gap_ready_low_capacity_design"
            blocker = ""
        review_rows.append(
            {
                "candidate_id": candidate_id,
                "target": target,
                "split_contract_status": split_contract["split_contract_status"],
                "selected_feature_profile": selected["feature_profile"],
                "selected_method": selected["method"],
                "selected_validation_rmse": selected["validation_rmse"],
                "selected_validation_normalized_rmse": selected_val_norm,
                "selected_test_rmse": selected["test_rmse"],
                "selected_test_normalized_rmse": selected_test_norm,
                "mean_validation_rmse": mean_val_rmse,
                "mean_test_rmse": mean_test_rmse,
                "validation_relative_improvement_over_mean": val_gain,
                "test_relative_improvement_over_mean": test_gain,
                "baseline_visible_gap": baseline_visible_gap,
                "strong_baseline_solved": strong_baseline_solved,
                "phase152_low_capacity_dense_design_candidate": status == "phase151_dense_gap_ready_low_capacity_design",
                "status": status,
                "blocker": blocker,
            }
        )
    return metric_rows, review_rows


def build_gate(
    *,
    tensor_manifest_rows: list[dict[str, Any]],
    split_contract_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    phase150_gate: dict[str, Any],
    phase149_gate: dict[str, Any],
    phase148_gate: dict[str, Any],
) -> dict[str, Any]:
    candidate_rows = [
        row for row in review_rows if row.get("phase152_low_capacity_dense_design_candidate")
    ]
    leakage_safe_sources = [
        row for row in split_contract_rows if row.get("leakage_safe_split")
    ]
    prior_locks_false = (
        _is_false(phase150_gate.get("operator_training_allowed_now"))
        and _is_false(phase150_gate.get("phase150_model_training_allowed"))
        and _is_false(phase150_gate.get("a100_training_allowed_now"))
        and _is_false(phase150_gate.get("a100_80gb_request_now"))
        and _is_false(phase149_gate.get("operator_training_allowed_now"))
        and _is_false(phase148_gate.get("a100_training_allowed_now"))
    )
    if candidate_rows:
        status = "phase151_fixed_grid_dense_baseline_ready_low_capacity_design"
        next_action = (
            "open Phase 152 as no-training or CPU/A800-small low-capacity dense mechanism design; "
            "do not train neural operators yet"
        )
    elif leakage_safe_sources:
        status = "phase151_fixed_grid_dense_baseline_closed_no_operator_gap"
        next_action = "close neural-operator branch as diagnostic unless a new dense target/split is added"
    else:
        status = "phase151_fixed_grid_dense_baseline_closed_no_leakage_safe_split"
        next_action = "acquire or define leakage-safe dense splits before revisiting operator models"
    return {
        "status": status,
        "tensor_manifest_rows": len(tensor_manifest_rows),
        "split_contract_rows": len(split_contract_rows),
        "baseline_review_rows": len(review_rows),
        "leakage_safe_source_rows": len(leakage_safe_sources),
        "phase152_low_capacity_dense_design_candidates": len(candidate_rows),
        "phase150_gate_status": phase150_gate.get("status"),
        "phase151_model_mechanism_allowed": bool(candidate_rows),
        "phase151_model_training_allowed": False,
        "operator_training_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "prior_training_locks_verified_false": prior_locks_false,
        "next_action": next_action,
    }


def _markdown_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[str]:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(_csv_value(row.get(field)) for field in fields) + " |"
        for row in rows
    ]
    return [header, sep, *body]


def _write_markdown(
    path: Path,
    *,
    gate: dict[str, Any],
    tensor_manifest_rows: list[dict[str, Any]],
    split_contract_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Phase 151 Fixed-Grid Dense Baseline Review",
        "",
        f"- Status: `{gate['status']}`",
        f"- Leakage-safe source rows: `{gate['leakage_safe_source_rows']}`",
        f"- Phase 152 low-capacity dense design candidates: `{gate['phase152_low_capacity_dense_design_candidates']}`",
        f"- Phase 151 model mechanism allowed: `{_csv_value(gate['phase151_model_mechanism_allowed'])}`",
        "- Phase 151 model training allowed: `false`",
        "- Operator training allowed now: `false`",
        "- A100 training allowed now: `false`",
        "",
        "## Tensor Manifests",
        "",
        *_markdown_table(tensor_manifest_rows, TENSOR_MANIFEST_FIELDS),
        "",
        "## Split Contracts",
        "",
        *_markdown_table(split_contract_rows, SPLIT_CONTRACT_FIELDS),
        "",
        "## Baseline Review",
        "",
        *_markdown_table(review_rows, REVIEW_FIELDS),
        "",
        f"Next action: {gate['next_action']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_package(
    *,
    root: Path,
    output_dir: Path,
    phase_inputs: dict[str, Path] | None = None,
    dense_candidates: list[dict[str, Any]] | None = None,
    min_points_per_frame: int = 10,
    min_summary_rows: int = 9,
    min_validation_relative_improvement: float = 0.05,
    min_unsolved_normalized_rmse: float = 0.10,
    n_estimators: int = 80,
) -> dict[str, Any]:
    phase_inputs = dict(phase_inputs or PHASE_INPUTS)
    resolved_inputs = {
        key: path if path.is_absolute() else root / path
        for key, path in phase_inputs.items()
    }
    phase150_gate = _read_json(resolved_inputs["phase150_gate"])
    phase149_gate = _read_json(resolved_inputs["phase149_gate"])
    phase148_gate = _read_json(resolved_inputs["phase148_gate"])
    inventory_rows = _read_csv(resolved_inputs["phase150_inventory"])
    candidates = dense_candidates or dense_candidates_from_inventory(root, inventory_rows)

    table_dir = output_dir / "fixed_grid_tables"
    split_dir = output_dir / "split_manifests"
    table_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    combined_summary_rows: list[dict[str, Any]] = []
    tensor_manifest_rows: list[dict[str, Any]] = []
    split_contract_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    candidate_outputs: dict[str, Any] = {}

    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        summary_rows, metadata = build_fixed_grid_rows(
            candidate=candidate,
            root=root,
            min_points_per_frame=min_points_per_frame,
        )
        split_manifest, split_contract = build_split_contract(summary_rows, candidate_id)
        table_path = table_dir / f"{candidate_id}_fixed_grid_summary.csv"
        split_path = split_dir / f"{candidate_id}_split_manifest.json"
        _write_csv(table_path, summary_rows, FIXED_GRID_FIELDS)
        _write_json(split_path, split_manifest)
        combined_summary_rows.extend(summary_rows)
        tensor_status = (
            "fixed_grid_summary_ready"
            if len(summary_rows) >= min_summary_rows
            else "blocked_too_few_summary_rows"
        )
        blocker = metadata.get("blocker", "")
        if tensor_status != "fixed_grid_summary_ready":
            blocker = blocker or "too_few_summary_rows"
        tensor_manifest_rows.append(
            {
                "candidate_id": candidate_id,
                "source_path": metadata["source_path"],
                "source_rows": metadata["source_rows"],
                "summary_rows": metadata["summary_rows"],
                "target_column": metadata["target_column"],
                "target_columns": list(TARGET_COLUMNS),
                "grid_index_columns": "frame_index,row_index,col_index",
                "split_axis": split_contract["split_axis"],
                "split_contract_status": split_contract["split_contract_status"],
                "leakage_safe_split": split_contract["leakage_safe_split"],
                "train_rows": split_contract["train_rows"],
                "val_rows": split_contract["val_rows"],
                "test_rows": split_contract["test_rows"],
                "tensor_manifest_status": tensor_status,
                "blocker": blocker,
            }
        )
        split_contract_rows.append(split_contract)
        candidate_outputs[candidate_id] = {
            "fixed_grid_table": _display_path(table_path, root),
            "split_manifest": _display_path(split_path, root),
        }
        if tensor_status != "fixed_grid_summary_ready" or not split_manifest["splits"]["train"]:
            for target in TARGET_COLUMNS:
                review_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "target": target,
                        "split_contract_status": split_contract["split_contract_status"],
                        "selected_feature_profile": "",
                        "selected_method": "",
                        "status": "blocked_no_fixed_grid_review",
                        "blocker": blocker or split_contract["split_contract_status"],
                        "baseline_visible_gap": False,
                        "strong_baseline_solved": False,
                        "phase152_low_capacity_dense_design_candidate": False,
                    }
                )
            continue
        candidate_metric_rows, candidate_review_rows = evaluate_candidate_baselines(
            candidate_id=candidate_id,
            rows=summary_rows,
            table_path=table_path,
            split_manifest_path=split_path,
            split_contract=split_contract,
            min_validation_relative_improvement=min_validation_relative_improvement,
            min_unsolved_normalized_rmse=min_unsolved_normalized_rmse,
            n_estimators=n_estimators,
        )
        metric_rows.extend(candidate_metric_rows)
        review_rows.extend(candidate_review_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_table_path = output_dir / "phase151_fixed_grid_summary_table.csv"
    tensor_manifest_table_path = output_dir / "phase151_tensor_manifest_table.csv"
    split_contract_table_path = output_dir / "phase151_split_contract_table.csv"
    metric_table_path = output_dir / "phase151_dense_baseline_metric_table.csv"
    review_table_path = output_dir / "phase151_dense_baseline_review_table.csv"
    gate_path = output_dir / "phase151_fixed_grid_dense_baseline_gate.json"
    markdown_path = output_dir / "phase151_fixed_grid_dense_baseline_review.md"
    manifest_path = output_dir / "phase151_fixed_grid_dense_baseline_manifest.json"

    gate = build_gate(
        tensor_manifest_rows=tensor_manifest_rows,
        split_contract_rows=split_contract_rows,
        review_rows=review_rows,
        phase150_gate=phase150_gate,
        phase149_gate=phase149_gate,
        phase148_gate=phase148_gate,
    )
    _write_csv(combined_table_path, combined_summary_rows, FIXED_GRID_FIELDS)
    _write_csv(tensor_manifest_table_path, tensor_manifest_rows, TENSOR_MANIFEST_FIELDS)
    _write_csv(split_contract_table_path, split_contract_rows, SPLIT_CONTRACT_FIELDS)
    _write_csv(metric_table_path, metric_rows, METRIC_FIELDS)
    _write_csv(review_table_path, review_rows, REVIEW_FIELDS)
    _write_json(gate_path, gate)
    _write_markdown(
        markdown_path,
        gate=gate,
        tensor_manifest_rows=tensor_manifest_rows,
        split_contract_rows=split_contract_rows,
        review_rows=review_rows,
    )
    manifest = {
        "phase": 151,
        "objective": "fixed_grid_tensor_split_manifest_dense_baseline_review_no_training",
        "inputs": {key: _display_path(path, root) for key, path in resolved_inputs.items()},
        "candidate_outputs": candidate_outputs,
        "outputs": {
            "fixed_grid_summary_table": _display_path(combined_table_path, root),
            "tensor_manifest_table": _display_path(tensor_manifest_table_path, root),
            "split_contract_table": _display_path(split_contract_table_path, root),
            "dense_baseline_metric_table": _display_path(metric_table_path, root),
            "dense_baseline_review_table": _display_path(review_table_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown_summary": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "candidate_sources": len(candidates),
            "fixed_grid_summary_rows": len(combined_summary_rows),
            "tensor_manifest_rows": len(tensor_manifest_rows),
            "split_contract_rows": len(split_contract_rows),
            "metric_rows": len(metric_rows),
            "review_rows": len(review_rows),
        },
        "limits": {
            "min_points_per_frame": min_points_per_frame,
            "min_summary_rows": min_summary_rows,
            "min_validation_relative_improvement": min_validation_relative_improvement,
            "min_unsolved_normalized_rmse": min_unsolved_normalized_rmse,
            "n_estimators": n_estimators,
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-points-per-frame", type=int, default=10)
    parser.add_argument("--min-summary-rows", type=int, default=9)
    parser.add_argument("--min-validation-relative-improvement", type=float, default=0.05)
    parser.add_argument("--min-unsolved-normalized-rmse", type=float, default=0.10)
    parser.add_argument("--n-estimators", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(
        root=root,
        output_dir=output_dir,
        min_points_per_frame=args.min_points_per_frame,
        min_summary_rows=args.min_summary_rows,
        min_validation_relative_improvement=args.min_validation_relative_improvement,
        min_unsolved_normalized_rmse=args.min_unsolved_normalized_rmse,
        n_estimators=args.n_estimators,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
