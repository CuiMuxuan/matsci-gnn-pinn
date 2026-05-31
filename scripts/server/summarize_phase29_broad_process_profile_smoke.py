#!/usr/bin/env python3
"""Summarize Phase 29 broad process-profile smoke artifacts."""
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
PINN_TAGS = (
    ("no_process", "no_process"),
    ("process_axis_profile", "process_axis_profile"),
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


def _run_id(split: str, dataset_limit: int, dataset_order: str) -> str:
    return (
        "ambench_multiline_process_temperature_"
        f"broad{dataset_limit}_{dataset_order}_{split}_process_axis_profile_smoke_a100_sxm4_40gb_v1"
    )


def _collect_manifest(root: Path, run_id: str) -> dict[str, Any]:
    path = root / "outputs" / "data_audits" / f"{run_id}_manifest.json"
    manifest = _read_json(path)
    return {
        "path": str(path),
        "n_rows": manifest.get("n_rows"),
        "dataset_paths": manifest.get("dataset_paths", []),
        "dataset_selection": (manifest.get("metadata") or {}).get("dataset_selection", {}),
        "process_groups": (manifest.get("metadata") or {}).get("process_groups", {}),
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
        run_id = _run_id(split=split, dataset_limit=dataset_limit, dataset_order=dataset_order)
        split_rows: dict[str, Any] = {
            "run_id": run_id,
            "manifest": _collect_manifest(root, run_id),
            "methods": {},
        }
        for method, tag in BASELINE_TAGS:
            path = root / "outputs" / "baselines" / f"{run_id}_{tag}_regions_q90.json"
            row = {"path": str(path)}
            if path.exists():
                row.update(_metric_fields(_test_metrics(_read_json(path))))
            else:
                row["missing"] = True
            split_rows["methods"][method] = row

        for method, tag in PINN_TAGS:
            path = root / "outputs" / "runs" / f"{run_id}_macro_pinn_minmax_{tag}_v1" / "metrics.json"
            row = {"path": str(path)}
            if path.exists():
                data = _read_json(path)
                features = data.get("input_features") or {}
                profile = features.get("conditioning_profile") or {}
                selected = profile.get("selected") or {}
                row.update(_metric_fields(_test_metrics(data)))
                row.update(
                    {
                        "conditioning_mode": features.get("conditioning_mode"),
                        "feature_normalization": (features.get("normalization") or {}).get("mode"),
                        "conditioning_profile": profile.get("profile"),
                        "conditioning_profile_group_key": profile.get("group_key"),
                        "selected_conditioning_mode": selected.get("conditioning_mode"),
                        "selected_feature_normalization": selected.get("feature_normalization"),
                    }
                )
            else:
                row["missing"] = True
            split_rows["methods"][method] = row
        summary["splits"][split] = split_rows
    return summary


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def print_manifest_markdown(summary: dict[str, Any]) -> None:
    first_split = next(iter(summary["splits"].values()))
    manifest = first_split["manifest"]
    print("| field | value |")
    print("|---|---|")
    print(f"| dataset order | {summary['dataset_order']} |")
    print(f"| dataset limit | {summary['dataset_limit']} |")
    print(f"| rows | {manifest.get('n_rows')} |")
    print(f"| selected datasets | {len(manifest.get('dataset_paths') or [])} |")
    print(f"| dataset paths | `{', '.join(manifest.get('dataset_paths') or [])}` |")
    for group_name, groups in (manifest.get("process_groups") or {}).items():
        print(f"| {group_name} groups | `{json.dumps(groups, sort_keys=True)}` |")


def print_metrics_markdown(summary: dict[str, Any]) -> None:
    print("| split | method | test RMSE | hot q90 RMSE | gradient q90 RMSE | selected route |")
    print("|---|---|---:|---:|---:|---|")
    for split, split_rows in summary["splits"].items():
        for method in (
            "mean",
            "extra_trees_process",
            "no_process",
            "process_axis_profile",
            "knn_process",
            "extra_trees_coords",
            "knn_coords",
        ):
            row = split_rows["methods"].get(method, {})
            if row.get("missing"):
                print(f"| {split} | {method} | MISSING |  |  | {row.get('path', '')} |")
                continue
            selected = ""
            if row.get("selected_conditioning_mode"):
                selected = "{}/{}".format(
                    row.get("selected_conditioning_mode"),
                    row.get("selected_feature_normalization"),
                )
            print(
                "| {split} | {method} | {rmse} | {hot} | {grad} | {selected} |".format(
                    split=split,
                    method=method,
                    rmse=_fmt(row.get("rmse")),
                    hot=_fmt(row.get("hot_q90_rmse")),
                    grad=_fmt(row.get("gradient_q90_rmse")),
                    selected=selected,
                )
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Project root containing outputs/")
    parser.add_argument("--dataset-limit", type=int, default=12)
    parser.add_argument("--dataset-order", default="process_round_robin")
    parser.add_argument("--split", action="append", choices=DEFAULT_SPLITS)
    parser.add_argument("--json-output", help="Optional path to write JSON summary")
    args = parser.parse_args()

    splits = tuple(args.split) if args.split else DEFAULT_SPLITS
    summary = collect_rows(
        root=Path(args.root),
        splits=splits,
        dataset_limit=args.dataset_limit,
        dataset_order=args.dataset_order,
    )
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print_manifest_markdown(summary)
    print()
    print_metrics_markdown(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
