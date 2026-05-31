#!/usr/bin/env python3
"""Summarize Phase 35 region-weighted seed-check metrics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_BASE_RUN_ID = (
    "ambench_multiline_process_temperature_broad12_process_round_robin_"
    "spot_size_broad_process_profile_smoke_a100_sxm4_40gb_v1"
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _test_metrics(data: dict[str, Any]) -> dict[str, Any]:
    split_metrics = data.get("split_metrics") or {}
    if isinstance(split_metrics, dict) and isinstance(split_metrics.get("test"), dict):
        return split_metrics["test"]
    return data.get("metrics") or data


def _metric_value(metrics: dict[str, Any], name: str) -> float | None:
    primary = metrics.get("metrics") or metrics
    value = primary.get(name)
    return float(value) if value is not None else None


def _region_rmse(metrics: dict[str, Any], name: str) -> float | None:
    regions = metrics.get("region_metrics") or metrics.get("regions") or {}
    region = regions.get(name) or {}
    region_metrics = region.get("metrics") or region
    value = region_metrics.get("rmse")
    return float(value) if value is not None else None


def _metrics_row(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"path": str(path)}
    if not path.exists():
        row["missing"] = True
        return row
    data = _read_json(path)
    metrics = _test_metrics(data)
    data_loss = data.get("data_loss_weighting") or {}
    data_loss_enabled = bool(data_loss.get("enabled"))
    row.update(
        {
            "missing": False,
            "rmse": _metric_value(metrics, "rmse"),
            "mae": _metric_value(metrics, "mae"),
            "relative_l2": _metric_value(metrics, "relative_l2"),
            "hot_q90_rmse": _region_rmse(metrics, "hot_q90"),
            "gradient_q90_rmse": _region_rmse(metrics, "gradient_q90"),
            "data_loss_weighting_enabled": data_loss_enabled,
            "data_loss_weighting_mode": data_loss.get("mode") if data_loss_enabled else None,
            "data_loss_region_weight": data_loss.get("region_weight") if data_loss_enabled else None,
            "data_loss_weighted_points": data_loss.get("selected_points") if data_loss_enabled else None,
        }
    )
    return row


def _seed_path(root: Path, base_run_id: str, seed: int, tag: str) -> Path:
    if seed == 7 and tag == "broad_process_profile":
        return root / "outputs" / "runs" / f"{base_run_id}_macro_pinn_minmax_{tag}_v1" / "metrics.json"
    if seed == 7:
        weighted_run_id = base_run_id.replace("broad_process_profile", tag)
        return root / "outputs" / "runs" / f"{weighted_run_id}_macro_pinn_minmax_{tag}_v1" / "metrics.json"
    return root / "outputs" / "runs" / f"{base_run_id}_seed{seed}_macro_pinn_minmax_{tag}_v1" / "metrics.json"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _pstdev(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"n": sum(1 for row in rows if not row.get("missing"))}
    for metric in ("rmse", "hot_q90_rmse", "gradient_q90_rmse", "mae", "relative_l2"):
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        aggregate[f"{metric}_mean"] = _mean(values)
        aggregate[f"{metric}_pstdev"] = _pstdev(values)
    return aggregate


def collect_summary(root: Path, seeds: list[int], tags: list[str], base_run_id: str) -> dict[str, Any]:
    methods = ["broad_process_profile", *tags]
    summary: dict[str, Any] = {
        "base_run_id": base_run_id,
        "seeds": seeds,
        "methods": methods,
        "rows": [],
        "aggregate": {},
    }
    for method in methods:
        method_rows = []
        for seed in seeds:
            row = _metrics_row(_seed_path(root, base_run_id, seed, method))
            row["seed"] = seed
            row["method"] = "broad_process_v1" if method == "broad_process_profile" else method
            method_rows.append(row)
            summary["rows"].append(row)
        summary["aggregate"][row["method"]] = _aggregate(method_rows)
    return summary


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def print_markdown(summary: dict[str, Any]) -> None:
    print("| method | seed | test RMSE | hot q90 RMSE | gradient q90 RMSE | region weight | weighted points |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in summary["rows"]:
        if row.get("missing"):
            print(f"| {row['method']} | {row['seed']} | MISSING |  |  |  |  |")
            continue
        print(
            "| {method} | {seed} | {rmse} | {hot} | {grad} | {weight} | {points} |".format(
                method=row["method"],
                seed=row["seed"],
                rmse=_fmt(row.get("rmse")),
                hot=_fmt(row.get("hot_q90_rmse")),
                grad=_fmt(row.get("gradient_q90_rmse")),
                weight=_fmt(row.get("data_loss_region_weight")),
                points=_fmt(row.get("data_loss_weighted_points")),
            )
        )
    print()
    print("| method | n | test RMSE mean +/- std | hot q90 mean +/- std | gradient q90 mean +/- std |")
    print("|---|---:|---:|---:|---:|")
    for method, row in summary["aggregate"].items():
        print(
            "| {method} | {n} | {rmse} +/- {rmse_std} | {hot} +/- {hot_std} | {grad} +/- {grad_std} |".format(
                method=method,
                n=row["n"],
                rmse=_fmt(row.get("rmse_mean")),
                rmse_std=_fmt(row.get("rmse_pstdev")),
                hot=_fmt(row.get("hot_q90_rmse_mean")),
                hot_std=_fmt(row.get("hot_q90_rmse_pstdev")),
                grad=_fmt(row.get("gradient_q90_rmse_mean")),
                grad_std=_fmt(row.get("gradient_q90_rmse_pstdev")),
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Project root containing outputs/")
    parser.add_argument("--base-run-id", default=DEFAULT_BASE_RUN_ID)
    parser.add_argument("--seed", type=int, action="append", help="Seed to include. Defaults to 7, 1, 2.")
    parser.add_argument("--tag", action="append", help="Region-weighted tag to include. Defaults to rw15 and rw125.")
    parser.add_argument("--json-output", help="Optional path to write JSON summary.")
    parser.add_argument("--require-complete", action="store_true", help="Fail if any expected metrics file is missing.")
    args = parser.parse_args()

    seeds = args.seed if args.seed else [7, 1, 2]
    tags = args.tag if args.tag else ["rw15", "rw125"]
    summary = collect_summary(Path(args.root), seeds, tags, args.base_run_id)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print_markdown(summary)
    if args.require_complete and any(row.get("missing") for row in summary["rows"]):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
