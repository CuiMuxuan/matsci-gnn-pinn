#!/usr/bin/env python3
"""Compare Phase 29 and Phase 30 broad process-profile smoke artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SPLITS = ("line", "laser_power", "scan_speed", "spot_size", "process")
BASELINE_TAGS = (
    ("mean", "mean_constant"),
    ("knn_coords", "knn_coords"),
    ("knn_process", "knn_process"),
    ("extra_trees_coords", "extra_trees_coords"),
    ("extra_trees_process", "extra_trees_process"),
)
PINN_SPECS = (
    (
        "no_process",
        "process_axis_profile",
        "no_process",
    ),
    (
        "process_axis_v1",
        "process_axis_profile",
        "process_axis_profile",
    ),
    (
        "broad_process_v1",
        "broad_process_profile",
        "broad_process_profile",
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _test_metrics(data: dict[str, Any]) -> dict[str, Any]:
    results = data.get("results") or []
    if results and isinstance(results[0], dict):
        data = results[0]
    split_metrics = data.get("split_metrics") or {}
    if isinstance(split_metrics, dict) and isinstance(split_metrics.get("test"), dict):
        return split_metrics["test"]
    metrics = data.get("metrics") or {}
    if isinstance(metrics, dict) and isinstance(metrics.get("test"), dict):
        return metrics["test"]
    return data


def _metric_value(metrics: dict[str, Any], name: str) -> float | None:
    primary_metrics = metrics.get("metrics") or metrics
    value = primary_metrics.get(name)
    return float(value) if value is not None else None


def _region_rmse(metrics: dict[str, Any], name: str) -> float | None:
    regions = metrics.get("region_metrics") or metrics.get("regions") or {}
    region = regions.get(name) or {}
    region_metrics = region.get("metrics") or region
    value = region_metrics.get("rmse")
    return float(value) if value is not None else None


def _metric_fields(metrics: dict[str, Any]) -> dict[str, float | None]:
    return {
        "rmse": _metric_value(metrics, "rmse"),
        "mae": _metric_value(metrics, "mae"),
        "relative_l2": _metric_value(metrics, "relative_l2"),
        "hot_q90_rmse": _region_rmse(metrics, "hot_q90"),
        "gradient_q90_rmse": _region_rmse(metrics, "gradient_q90"),
    }


def _run_id(split: str, dataset_limit: int, dataset_order: str, profile_tag: str) -> str:
    return (
        "ambench_multiline_process_temperature_"
        f"broad{dataset_limit}_{dataset_order}_{split}_{profile_tag}_smoke_a100_sxm4_40gb_v1"
    )


def _manifest_path(root: Path, run_id: str) -> Path:
    return root / "outputs" / "data_audits" / f"{run_id}_manifest.json"


def _split_path(root: Path, run_id: str) -> Path:
    return root / "outputs" / "data_splits" / f"{run_id}_split.json"


def _baseline_run_id(split: str, dataset_limit: int, dataset_order: str) -> str:
    return _run_id(
        split=split,
        dataset_limit=dataset_limit,
        dataset_order=dataset_order,
        profile_tag="process_axis_profile",
    )


def _manifest_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    metadata = manifest.get("metadata") or {}
    sampling = metadata.get("sampling") or {}
    selection = sampling.get("selection") or {}
    return {
        "target": manifest.get("target"),
        "n_rows": manifest.get("n_rows"),
        "dataset_paths": manifest.get("dataset_paths"),
        "dataset_selection": metadata.get("dataset_selection"),
        "sampling": {
            "frame_start": sampling.get("frame_start"),
            "frame_step": sampling.get("frame_step"),
            "max_frames": sampling.get("max_frames"),
            "row_start": sampling.get("row_start"),
            "row_step": sampling.get("row_step"),
            "max_rows": sampling.get("max_rows"),
            "col_start": sampling.get("col_start"),
            "col_step": sampling.get("col_step"),
            "max_cols": sampling.get("max_cols"),
            "min_signal": sampling.get("min_signal"),
            "selection": {
                "mode": selection.get("mode"),
                "active_target": selection.get("active_target"),
                "hot_quantile": selection.get("hot_quantile"),
                "gradient_quantile": selection.get("gradient_quantile"),
                "background_fraction": selection.get("background_fraction"),
                "max_points_per_frame": selection.get("max_points_per_frame"),
            },
        },
        "process_groups": metadata.get("process_groups"),
    }


def _split_signature(split: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_rows": split.get("n_rows"),
        "n_groups": split.get("n_groups"),
        "group_key": split.get("group_key"),
        "strategy": split.get("strategy"),
        "group_order": split.get("group_order"),
        "group_splits": split.get("group_splits"),
        "rows_per_group": split.get("rows_per_group"),
        "splits": split.get("splits"),
    }


def _comparison_signature(root: Path, run_id: str) -> dict[str, Any] | None:
    manifest_file = _manifest_path(root, run_id)
    split_file = _split_path(root, run_id)
    if not manifest_file.exists() or not split_file.exists():
        return None
    return {
        "manifest": _manifest_signature(_read_json(manifest_file)),
        "split": _split_signature(_read_json(split_file)),
    }


def _comparison_reason(
    reference: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> str | None:
    if reference is None:
        return "reference manifest/split is missing"
    if candidate is None:
        return "candidate manifest/split is missing"
    if reference == candidate:
        return None
    ref_manifest = reference.get("manifest") or {}
    cand_manifest = candidate.get("manifest") or {}
    for key in ("target", "n_rows", "dataset_paths", "dataset_selection", "sampling", "process_groups"):
        if ref_manifest.get(key) != cand_manifest.get(key):
            return f"manifest {key} differs from reference"
    ref_split = reference.get("split") or {}
    cand_split = candidate.get("split") or {}
    for key in ("n_rows", "n_groups", "group_key", "strategy", "group_order", "group_splits", "rows_per_group", "splits"):
        if ref_split.get(key) != cand_split.get(key):
            return f"split {key} differs from reference"
    return "manifest/split signature differs from reference"


def _collect_profile_metadata(data: dict[str, Any]) -> dict[str, Any]:
    features = data.get("input_features") or {}
    profile = features.get("conditioning_profile") or {}
    selected = profile.get("selected") or {}
    effective = profile.get("effective") or {}
    return {
        "input_features_enabled": features.get("enabled"),
        "input_feature_count": features.get("count"),
        "conditioning_mode": features.get("conditioning_mode"),
        "feature_normalization": (features.get("normalization") or {}).get("mode"),
        "conditioning_profile": profile.get("profile"),
        "conditioning_profile_group_key": profile.get("group_key"),
        "selected_conditioning_mode": selected.get("conditioning_mode"),
        "selected_feature_normalization": selected.get("feature_normalization"),
        "effective_conditioning_mode": effective.get("conditioning_mode"),
        "effective_feature_columns": effective.get("feature_columns"),
    }


def collect_rows(
    root: Path,
    splits: tuple[str, ...],
    dataset_limit: int,
    dataset_order: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset_limit": dataset_limit,
        "dataset_order": dataset_order,
        "splits": {},
    }
    for split in splits:
        rows: dict[str, Any] = {"methods": {}}
        baseline_id = _baseline_run_id(split, dataset_limit, dataset_order)
        reference_signature = _comparison_signature(root, baseline_id)
        manifest_path = _manifest_path(root, baseline_id)
        if manifest_path.exists():
            manifest = _read_json(manifest_path)
            rows["manifest"] = {
                "path": str(manifest_path),
                "n_rows": manifest.get("n_rows"),
                "dataset_paths": manifest.get("dataset_paths"),
                "dataset_selection": (manifest.get("metadata") or {}).get("dataset_selection"),
                "sampling": (manifest.get("metadata") or {}).get("sampling"),
                "process_groups": (manifest.get("metadata") or {}).get("process_groups"),
            }
        rows["reference_run_id"] = baseline_id
        rows["reference_signature_available"] = reference_signature is not None
        rows["all_methods_comparable"] = reference_signature is not None
        for method, tag in BASELINE_TAGS:
            path = root / "outputs" / "baselines" / f"{baseline_id}_{tag}_regions_q90.json"
            row = {"path": str(path), "comparison_status": "comparable"}
            if path.exists():
                row.update(_metric_fields(_test_metrics(_read_json(path))))
            else:
                row["missing"] = True
                rows["all_methods_comparable"] = False
            rows["methods"][method] = row
        for method, profile_tag, run_tag in PINN_SPECS:
            run_id = _run_id(split, dataset_limit, dataset_order, profile_tag)
            path = root / "outputs" / "runs" / f"{run_id}_macro_pinn_minmax_{run_tag}_v1" / "metrics.json"
            candidate_signature = _comparison_signature(root, run_id)
            comparison_reason = _comparison_reason(reference_signature, candidate_signature)
            comparison_status = "comparable" if comparison_reason is None else "incomparable"
            row = {
                "path": str(path),
                "run_id": run_id,
                "comparison_status": comparison_status,
            }
            if comparison_reason:
                row["comparison_reason"] = comparison_reason
                rows["all_methods_comparable"] = False
            if path.exists():
                data = _read_json(path)
                row.update(_metric_fields(_test_metrics(data)))
                row.update(_collect_profile_metadata(data))
            else:
                row["missing"] = True
                rows["all_methods_comparable"] = False
            rows["methods"][method] = row
        summary["splits"][split] = rows
    return summary


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def print_metrics_markdown(summary: dict[str, Any]) -> None:
    print("| split | method | status | test RMSE | hot q90 RMSE | gradient q90 RMSE | selected | effective features |")
    print("|---|---|---|---:|---:|---:|---|---|")
    for split, split_rows in summary["splits"].items():
        for method in (
            "mean",
            "no_process",
            "process_axis_v1",
            "broad_process_v1",
            "extra_trees_process",
            "knn_process",
            "extra_trees_coords",
            "knn_coords",
        ):
            row = split_rows["methods"].get(method, {})
            if row.get("missing"):
                print(f"| {split} | {method} | MISSING |  |  |  |  | {row.get('path', '')} |")
                continue
            status = str(row.get("comparison_status") or "comparable")
            if status != "comparable":
                reason = row.get("comparison_reason", "")
                print(f"| {split} | {method} | INCOMPARABLE: {reason} |  |  |  |  | {row.get('path', '')} |")
                continue
            selected = ""
            if row.get("selected_conditioning_mode"):
                selected = "{}/{}".format(
                    row.get("selected_conditioning_mode"),
                    row.get("selected_feature_normalization"),
                )
            print(
                (
                    "| {split} | {method} | {status} | {rmse} | {hot} | {grad} | "
                    "{selected} | {features} |"
                ).format(
                    split=split,
                    method=method,
                    status=status,
                    rmse=_fmt(row.get("rmse")),
                    hot=_fmt(row.get("hot_q90_rmse")),
                    grad=_fmt(row.get("gradient_q90_rmse")),
                    selected=selected,
                    features=_fmt(row.get("effective_feature_columns")),
                )
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Project root containing outputs/")
    parser.add_argument("--dataset-limit", type=int, default=12)
    parser.add_argument("--dataset-order", default="process_round_robin")
    parser.add_argument("--split", action="append", choices=DEFAULT_SPLITS)
    parser.add_argument("--json-output", help="Optional path to write JSON summary")
    parser.add_argument(
        "--require-comparable",
        action="store_true",
        help="Exit non-zero if any requested method has a missing or mismatched manifest/split signature.",
    )
    args = parser.parse_args()

    splits = tuple(args.split) if args.split else DEFAULT_SPLITS
    summary = collect_rows(Path(args.root), splits, args.dataset_limit, args.dataset_order)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print_metrics_markdown(summary)
    if args.require_comparable and any(
        not split_rows.get("all_methods_comparable")
        for split_rows in summary["splits"].values()
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
