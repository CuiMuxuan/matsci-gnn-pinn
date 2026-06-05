#!/usr/bin/env python3
"""Phase 110 NIST AMMT layer-mean target review gate.

Reviews the remaining Phase 108 alternate sequence target, `target_cp_layer_mean`,
before any model mechanism. The gate checks whether the target is only a
layer/time shortcut repeated across A/B camera rows.
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


TARGET = "target_cp_layer_mean"
METHODS = ("mean", "knn", "extra_trees", "hist_gradient_boosting")
PROFILE_FEATURES: dict[str, list[str] | None] = {
    "mean_guard": None,
    "full_phase108": [
        "x",
        "y",
        "t",
        "source_p_mean",
        "source_p_nonzero_fraction",
        "source_x_range",
        "source_y_range",
        "target_camera_code",
    ],
    "no_camera": [
        "x",
        "y",
        "t",
        "source_p_mean",
        "source_p_nonzero_fraction",
        "source_x_range",
        "source_y_range",
    ],
    "camera_only": ["target_camera_code"],
    "source_only": [
        "source_p_mean",
        "source_p_nonzero_fraction",
        "source_x_range",
        "source_y_range",
    ],
    "layer_time_only": ["t"],
    "layer_time_camera": ["t", "target_camera_code"],
}
METRIC_FIELDS = (
    "profile",
    "method",
    "split",
    "n_points",
    "rmse",
    "mae",
    "relative_l2",
    "normalized_rmse",
    "hot_q90_rmse",
    "gradient_q90_rmse",
)
PROFILE_FIELDS = (
    "profile",
    "feature_columns",
    "selected_validation_method",
    "selected_validation_rmse",
    "selected_validation_normalized_rmse",
    "selected_test_rmse",
    "selected_test_normalized_rmse",
    "validation_relative_gain_over_mean",
    "test_relative_gain_over_mean",
    "validation_delta_vs_full",
    "test_delta_vs_full",
    "status",
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
        return f"{value:.6f}" if math.isfinite(value) else ""
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
    for profile, profile_payloads in payloads.items():
        for method, payload in profile_payloads.items():
            result = payload["results"][0]
            for split, split_payload in result["split_metrics"].items():
                metrics = split_payload["metrics"]
                rows.append(
                    {
                        "profile": profile,
                        "method": method,
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


def _metric_index(rows: list[dict[str, Any]], metric: str) -> dict[tuple[str, str, str], float]:
    index: dict[tuple[str, str, str], float] = {}
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)):
            index[(str(row["profile"]), str(row["method"]), str(row["split"]))] = float(value)
    return index


def _metric_float(index: dict[tuple[str, str, str], float], profile: str, method: str, split: str) -> float:
    key = (profile, method, split)
    if key not in index:
        raise ValueError(f"Missing metric for {profile}/{method}/{split}")
    return index[key]


def _layer_diagnostics(rows: list[dict[str, str]], target: str = TARGET) -> dict[str, Any]:
    by_layer: dict[int, list[dict[str, str]]] = defaultdict(list)
    by_camera: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        layer = int(float(row["source_layer_index"]))
        value = float(row[target])
        by_layer[layer].append(row)
        by_camera[str(row.get("target_camera_code", ""))].append(value)
    duplicated = 0
    two_camera_layers = [layer_rows for layer_rows in by_layer.values() if len(layer_rows) == 2]
    for layer_rows in two_camera_layers:
        values = [float(row[target]) for row in layer_rows]
        if max(values) - min(values) <= 1e-9:
            duplicated += 1
    return {
        "row_count": len(rows),
        "layer_count": len(by_layer),
        "two_camera_layer_count": len(two_camera_layers),
        "duplicated_target_two_camera_layer_count": duplicated,
        "all_two_camera_layers_duplicate_target": duplicated == len(two_camera_layers),
        "camera_target_mean": {
            camera: sum(values) / len(values) for camera, values in sorted(by_camera.items())
        },
        "camera_target_range": {
            camera: max(values) - min(values) for camera, values in sorted(by_camera.items())
        },
    }


def _profile_rows(
    metric_rows: list[dict[str, Any]],
    *,
    min_independent_validation_gain: float,
) -> list[dict[str, Any]]:
    rmse = _metric_index(metric_rows, "rmse")
    normalized = _metric_index(metric_rows, "normalized_rmse")
    mean_val = _metric_float(rmse, "mean_guard", "mean", "val")
    mean_test = _metric_float(rmse, "mean_guard", "mean", "test")
    full_candidates = [
        (_metric_float(rmse, "full_phase108", method, "val"), method)
        for method in METHODS
        if ("full_phase108", method, "val") in rmse
    ]
    full_val, full_method = min(full_candidates)
    full_test = _metric_float(rmse, "full_phase108", full_method, "test")
    rows: list[dict[str, Any]] = []
    for profile in PROFILE_FEATURES:
        candidates = [
            (_metric_float(rmse, profile, method, "val"), method)
            for method in METHODS
            if (profile, method, "val") in rmse
        ]
        selected_val, selected_method = min(candidates)
        selected_test = _metric_float(rmse, profile, selected_method, "test")
        gain = (mean_val - selected_val) / mean_val if mean_val else 0.0
        if profile == "mean_guard":
            status = "reference_mean_guard"
        elif profile == "full_phase108":
            status = "reference_full_phase108_profile"
        elif profile in {"layer_time_only", "layer_time_camera"} and selected_val <= full_val:
            status = "layer_time_shortcut_matches_or_beats_full_validation"
        elif profile == "source_only" and gain < min_independent_validation_gain:
            status = "no_independent_source_validation_gain"
        elif selected_val >= mean_val:
            status = "no_validation_gain_over_mean_guard"
        else:
            status = "profile_has_validation_signal"
        rows.append(
            {
                "profile": profile,
                "feature_columns": PROFILE_FEATURES[profile],
                "selected_validation_method": selected_method,
                "selected_validation_rmse": selected_val,
                "selected_validation_normalized_rmse": normalized.get((profile, selected_method, "val")),
                "selected_test_rmse": selected_test,
                "selected_test_normalized_rmse": normalized.get((profile, selected_method, "test")),
                "validation_relative_gain_over_mean": gain,
                "test_relative_gain_over_mean": (mean_test - selected_test) / mean_test if mean_test else 0.0,
                "validation_delta_vs_full": selected_val - full_val,
                "test_delta_vs_full": selected_test - full_test,
                "status": status,
            }
        )
    return rows


def _build_gate(
    *,
    phase108_gate: dict[str, Any],
    phase109_gate: dict[str, Any],
    profile_rows: list[dict[str, Any]],
    layer_diagnostics: dict[str, Any],
    min_independent_validation_gain: float,
) -> dict[str, Any]:
    phase108_ready = phase108_gate.get("status") == "phase108_sequence_target_gap_ready_focused_review"
    phase109_closed = (
        phase109_gate.get("status") == "phase109_sequence_target_focused_review_closed_camera_shortcut"
    )
    layer_time_profiles = [
        row["profile"]
        for row in profile_rows
        if row["status"] == "layer_time_shortcut_matches_or_beats_full_validation"
    ]
    source_only = next(row for row in profile_rows if row["profile"] == "source_only")
    layer_time_shortcut = bool(
        layer_time_profiles
        and source_only["validation_relative_gain_over_mean"] < min_independent_validation_gain
        and layer_diagnostics["all_two_camera_layers_duplicate_target"]
    )
    full = next(row for row in profile_rows if row["profile"] == "full_phase108")
    layer_time_only = next(row for row in profile_rows if row["profile"] == "layer_time_only")
    if not phase108_ready:
        status = "phase110_layer_mean_review_blocked_by_phase108"
        next_action = "complete Phase 108 sequence target gate first"
    elif not phase109_closed:
        status = "phase110_layer_mean_review_blocked_by_phase109"
        next_action = "close Phase 109 selected-target review first"
    elif layer_time_shortcut:
        status = "phase110_layer_mean_target_review_closed_layer_time_shortcut"
        next_action = "close NIST AMMT sequence target branch as diagnostic; do not train"
    else:
        status = "phase110_layer_mean_target_review_ready_no_training_mechanism_gate"
        next_action = "design a no-training mechanism gate; keep A100 training closed"
    return {
        "status": status,
        "reviewed_phase108_status": phase108_gate.get("status"),
        "reviewed_phase109_status": phase109_gate.get("status"),
        "reviewed_target": TARGET,
        "full_phase108_validation_rmse": full["selected_validation_rmse"],
        "full_phase108_test_rmse": full["selected_test_rmse"],
        "layer_time_only_validation_rmse": layer_time_only["selected_validation_rmse"],
        "layer_time_only_test_rmse": layer_time_only["selected_test_rmse"],
        "layer_time_shortcut_profiles": layer_time_profiles,
        "source_only_validation_relative_gain_over_mean": source_only[
            "validation_relative_gain_over_mean"
        ],
        "min_independent_validation_gain": min_independent_validation_gain,
        "layer_time_shortcut_detected": layer_time_shortcut,
        "layer_diagnostics": layer_diagnostics,
        "phase110_focused_review_allowed": False,
        "phase110_model_mechanism_allowed": False,
        "phase110_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
    }


def _write_markdown(path: Path, gate: dict[str, Any], profile_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 110 NIST AMMT Layer-Mean Target Review Gate",
        "",
        f"- Status: `{gate['status']}`",
        f"- Reviewed target: `{gate['reviewed_target']}`",
        f"- Layer/time shortcut detected: `{gate['layer_time_shortcut_detected']}`",
        "- Model mechanism allowed: `false`",
        "- Model training allowed: `false`",
        "- A100 training allowed now: `false`",
        "",
        "| Profile | Status | Method | Val RMSE | Test RMSE | Delta vs full val |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in profile_rows:
        lines.append(
            "| {profile} | {status} | {selected_validation_method} | {selected_validation_rmse} | {selected_test_rmse} | {validation_delta_vs_full} |".format(
                **{key: _csv_value(value) for key, value in row.items()}
            )
        )
    lines.extend(["", f"Next action: {gate['next_action']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_package(
    *,
    root: Path,
    phase108_field_table: Path,
    split_manifest: Path,
    phase108_gate_path: Path,
    phase109_gate_path: Path,
    output_dir: Path,
    n_neighbors: int,
    n_estimators: int,
    min_independent_validation_gain: float,
) -> dict[str, Any]:
    phase108_gate = _read_json(phase108_gate_path)
    phase109_gate = _read_json(phase109_gate_path)
    source_rows = _read_csv(phase108_field_table)
    layer_diagnostics = _layer_diagnostics(source_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, dict[str, dict[str, Any]]] = {}
    for profile, feature_columns in PROFILE_FEATURES.items():
        payloads[profile] = {}
        methods = ("mean",) if profile == "mean_guard" else METHODS
        for method in methods:
            payloads[profile][method] = {
                "target": TARGET,
                "strategy": method,
                "feature_columns": feature_columns if method != "mean" else None,
                "results": [
                    evaluate_table(
                        table_path=phase108_field_table,
                        target=TARGET,
                        strategy=method,
                        split_manifest_path=split_manifest,
                        fit_split="train",
                        feature_columns=feature_columns if method != "mean" else None,
                        n_neighbors=n_neighbors,
                        n_estimators=n_estimators,
                        random_state=7,
                        hot_quantiles=[0.9],
                        gradient_quantiles=[0.9],
                    )
                ],
            }
    metric_rows = _metric_rows(payloads)
    profiles = _profile_rows(
        metric_rows,
        min_independent_validation_gain=min_independent_validation_gain,
    )
    gate = _build_gate(
        phase108_gate=phase108_gate,
        phase109_gate=phase109_gate,
        profile_rows=profiles,
        layer_diagnostics=layer_diagnostics,
        min_independent_validation_gain=min_independent_validation_gain,
    )

    metrics_path = output_dir / "phase110_nist_ammt_layer_mean_target_review_metric_table.csv"
    profile_path = output_dir / "phase110_nist_ammt_layer_mean_target_review_profile_table.csv"
    layer_path = output_dir / "phase110_nist_ammt_layer_mean_target_layer_diagnostics.json"
    gate_path = output_dir / "phase110_nist_ammt_layer_mean_target_review_gate.json"
    markdown_path = output_dir / "phase110_nist_ammt_layer_mean_target_review_summary.md"
    manifest_path = output_dir / "phase110_nist_ammt_layer_mean_target_review_manifest.json"
    _write_csv(metrics_path, metric_rows, METRIC_FIELDS)
    _write_csv(profile_path, profiles, PROFILE_FIELDS)
    _write_json(layer_path, layer_diagnostics)
    _write_json(gate_path, gate)
    _write_markdown(markdown_path, gate, profiles)
    manifest = {
        "phase": 110,
        "objective": "nist_ammt_layer_mean_target_review_no_training",
        "inputs": {
            "phase108_field_table": _display_path(phase108_field_table, root),
            "split_manifest": _display_path(split_manifest, root),
            "phase108_gate": _display_path(phase108_gate_path, root),
            "phase109_gate": _display_path(phase109_gate_path, root),
        },
        "outputs": {
            "metric_table": _display_path(metrics_path, root),
            "profile_table": _display_path(profile_path, root),
            "layer_diagnostics": _display_path(layer_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown_summary": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "rows": len(source_rows),
            "profiles": len(PROFILE_FEATURES),
            "metric_rows": len(metric_rows),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--phase108-field-table",
        type=Path,
        default=Path(
            "docs/results/phase108_nist_ammt_sequence_target_gate/"
            "phase108_nist_ammt_sequence_target_field_table.csv"
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
        "--phase108-gate",
        type=Path,
        default=Path(
            "docs/results/phase108_nist_ammt_sequence_target_gate/"
            "phase108_nist_ammt_sequence_target_gate.json"
        ),
    )
    parser.add_argument(
        "--phase109-gate",
        type=Path,
        default=Path(
            "docs/results/phase109_nist_ammt_sequence_target_focused_review_gate/"
            "phase109_nist_ammt_sequence_target_focused_review_gate.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase110_nist_ammt_layer_mean_target_review_gate"),
    )
    parser.add_argument("--n-neighbors", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--min-independent-validation-gain", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    phase108_field_table = (
        args.phase108_field_table
        if args.phase108_field_table.is_absolute()
        else root / args.phase108_field_table
    )
    split_manifest = (
        args.split_manifest if args.split_manifest.is_absolute() else root / args.split_manifest
    )
    phase108_gate = args.phase108_gate if args.phase108_gate.is_absolute() else root / args.phase108_gate
    phase109_gate = args.phase109_gate if args.phase109_gate.is_absolute() else root / args.phase109_gate
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(
        root=root,
        phase108_field_table=phase108_field_table,
        split_manifest=split_manifest,
        phase108_gate_path=phase108_gate,
        phase109_gate_path=phase109_gate,
        output_dir=output_dir,
        n_neighbors=args.n_neighbors,
        n_estimators=args.n_estimators,
        min_independent_validation_gain=args.min_independent_validation_gain,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
