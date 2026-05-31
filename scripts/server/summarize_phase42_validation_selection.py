#!/usr/bin/env python3
"""Inspect validation-selected process representation diagnostics for Phase 42."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VARIANTS = (
    {
        "name": "broad12_raw",
        "dataset_limit": 12,
        "feature_set": "raw_process",
        "profile_tag": "broad_process_profile",
        "run_tag": "broad_process_profile",
    },
    {
        "name": "broad12_derived_only",
        "dataset_limit": 12,
        "feature_set": "derived_only_am_energy_v1",
        "profile_tag": "phys_only2",
        "run_tag": "phys_only2",
    },
    {
        "name": "broad21_raw",
        "dataset_limit": 21,
        "feature_set": "raw_process",
        "profile_tag": "broad_process_profile",
        "run_tag": "broad_process_profile",
    },
    {
        "name": "broad21_raw_plus_derived",
        "dataset_limit": 21,
        "feature_set": "raw_plus_am_energy_v1",
        "profile_tag": "phys_proc",
        "run_tag": "phys_proc",
    },
    {
        "name": "broad21_derived_only",
        "dataset_limit": 21,
        "feature_set": "derived_only_am_energy_v1",
        "profile_tag": "phys_only2",
        "run_tag": "phys_only2",
    },
)


def _run_id(dataset_limit: int, profile_tag: str) -> str:
    return (
        "ambench_multiline_process_temperature_"
        f"broad{dataset_limit}_process_round_robin_laser_power_{profile_tag}_smoke_a100_sxm4_40gb_v1"
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _metric_value(split_payload: dict[str, Any], name: str) -> float | None:
    metrics = split_payload.get("metrics") or {}
    value = metrics.get(name)
    return float(value) if value is not None else None


def _region_rmse(split_payload: dict[str, Any], region_name: str) -> float | None:
    regions = split_payload.get("region_metrics") or {}
    region = regions.get(region_name) or {}
    region_metrics = region.get("metrics") or region
    value = region_metrics.get("rmse")
    return float(value) if value is not None else None


def _collect_variant(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    run_id = _run_id(int(spec["dataset_limit"]), str(spec["profile_tag"]))
    path = root / "outputs" / "runs" / f"{run_id}_macro_pinn_minmax_{spec['run_tag']}_v1" / "metrics.json"
    row = {
        "name": spec["name"],
        "dataset_limit": spec["dataset_limit"],
        "feature_set": spec["feature_set"],
        "path": str(path),
        "missing": not path.exists(),
        "splits": {},
        "effective_columns": [],
    }
    if not path.exists():
        return row
    payload = _read_json(path)
    input_features = payload.get("input_features") or {}
    row["effective_columns"] = input_features.get("effective_columns") or []
    split_metrics = payload.get("split_metrics") or {}
    for split_name in ("train", "val", "test"):
        split_payload = split_metrics.get(split_name) or {}
        row["splits"][split_name] = {
            "rmse": _metric_value(split_payload, "rmse"),
            "mae": _metric_value(split_payload, "mae"),
            "relative_l2": _metric_value(split_payload, "relative_l2"),
            "hot_q90_rmse": _region_rmse(split_payload, "hot_q90"),
            "gradient_q90_rmse": _region_rmse(split_payload, "gradient_q90"),
        }
    return row


def _select_by_metric(rows: list[dict[str, Any]], split_name: str, metric_name: str) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if not row.get("missing") and row.get("splits", {}).get(split_name, {}).get(metric_name) is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row["splits"][split_name][metric_name])


def summarize(root: Path) -> dict[str, Any]:
    rows = [_collect_variant(root, spec) for spec in VARIANTS]
    selections: dict[str, Any] = {}
    for dataset_limit in (12, 21):
        group = [row for row in rows if row["dataset_limit"] == dataset_limit]
        selections[str(dataset_limit)] = {}
        for metric_name in ("rmse", "hot_q90_rmse", "gradient_q90_rmse"):
            val_choice = _select_by_metric(group, "val", metric_name)
            test_choice = _select_by_metric(group, "test", metric_name)
            selections[str(dataset_limit)][metric_name] = {
                "val_selected": val_choice["name"] if val_choice else None,
                "test_best": test_choice["name"] if test_choice else None,
                "val_matches_test_best": (
                    val_choice is not None and test_choice is not None and val_choice["name"] == test_choice["name"]
                ),
            }
    return {"variants": rows, "selections": selections}


def print_markdown(summary: dict[str, Any]) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    print("| dataset | variant | feature set | split | RMSE | hot q90 | gradient q90 | effective columns |")
    print("|---:|---|---|---|---:|---:|---:|---|")
    for row in summary["variants"]:
        if row.get("missing"):
            print(f"| {row['dataset_limit']} | {row['name']} | {row['feature_set']} | MISSING |  |  |  | {row['path']} |")
            continue
        columns = ",".join(row.get("effective_columns") or [])
        for split_name in ("train", "val", "test"):
            metrics = row["splits"][split_name]
            print(
                "| {dataset} | {name} | {feature_set} | {split} | {rmse} | {hot} | {grad} | {columns} |".format(
                    dataset=row["dataset_limit"],
                    name=row["name"],
                    feature_set=row["feature_set"],
                    split=split_name,
                    rmse=fmt(metrics["rmse"]),
                    hot=fmt(metrics["hot_q90_rmse"]),
                    grad=fmt(metrics["gradient_q90_rmse"]),
                    columns=columns,
                )
            )
    print()
    print("| dataset | metric | val-selected | test-best | match |")
    print("|---:|---|---|---|---|")
    for dataset_limit, metric_rows in summary["selections"].items():
        for metric_name, payload in metric_rows.items():
            print(
                "| {dataset} | {metric} | {val} | {test} | {match} |".format(
                    dataset=dataset_limit,
                    metric=metric_name,
                    val=payload["val_selected"],
                    test=payload["test_best"],
                    match=payload["val_matches_test_best"],
                )
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Project root containing outputs/")
    parser.add_argument("--json-output", help="Optional path to write JSON summary")
    args = parser.parse_args()

    summary = summarize(Path(args.root))
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print_markdown(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
