#!/usr/bin/env python3
"""Summarize Phase 36 process-graph RBF paired seed-check metrics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_SPLITS = ("spot_size", "laser_power")
DEFAULT_SEEDS = (7, 1, 2)
DEFAULT_TAGS = ("pg_rbf_global",)
VALID_SPLITS = ("line", "laser_power", "scan_speed", "spot_size", "process")


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


def _base_run_id(split: str, dataset_limit: int, dataset_order: str) -> str:
    return (
        "ambench_multiline_process_temperature_"
        f"broad{dataset_limit}_{dataset_order}_{split}_"
        "broad_process_profile_smoke_a100_sxm4_40gb_v1"
    )


def _seed_path(root: Path, base_run_id: str, seed: int, tag: str) -> Path:
    runs_root = root / "outputs" / "runs"
    if tag == "broad_process_profile":
        if seed == 7:
            return runs_root / f"{base_run_id}_macro_pinn_minmax_{tag}_v1" / "metrics.json"
        return runs_root / f"{base_run_id}_seed{seed}_macro_pinn_minmax_{tag}_v1" / "metrics.json"

    if seed == 7:
        graph_run_id = base_run_id.replace("broad_process_profile", tag)
        return runs_root / f"{graph_run_id}_macro_pinn_minmax_{tag}_v1" / "metrics.json"
    return runs_root / f"{base_run_id}_seed{seed}_macro_pinn_minmax_{tag}_v1" / "metrics.json"


def _profile_metadata(data: dict[str, Any]) -> dict[str, Any]:
    features = data.get("input_features") or {}
    profile = features.get("conditioning_profile") or {}
    selected = profile.get("selected") or {}
    effective = profile.get("effective") or {}
    graph = features.get("process_graph_features") or {}
    return {
        "input_feature_count": features.get("count"),
        "input_effective_columns": features.get("effective_columns"),
        "conditioning_profile": profile.get("profile"),
        "conditioning_profile_group_key": profile.get("group_key"),
        "selected_conditioning_mode": selected.get("conditioning_mode"),
        "selected_feature_normalization": selected.get("feature_normalization"),
        "effective_conditioning_mode": effective.get("conditioning_mode"),
        "effective_feature_columns": effective.get("feature_columns"),
        "process_graph_enabled": graph.get("enabled"),
        "process_graph_mode": graph.get("mode"),
        "process_graph_columns": graph.get("columns"),
        "process_graph_fit_scope": graph.get("fit_scope"),
        "process_graph_requested_anchor_count": graph.get("requested_anchor_count"),
        "process_graph_anchor_count": graph.get("anchor_count"),
        "process_graph_source_unique_nodes": graph.get("source_unique_nodes"),
        "process_graph_length_scale": graph.get("length_scale"),
        "process_graph_feature_names": graph.get("feature_names"),
    }


def _metrics_row(root: Path, split: str, base_run_id: str, seed: int, tag: str) -> dict[str, Any]:
    method = "broad_process_v1" if tag == "broad_process_profile" else tag
    path = _seed_path(root, base_run_id, seed, tag)
    row: dict[str, Any] = {
        "split": split,
        "seed": seed,
        "method": method,
        "tag": tag,
        "path": str(path),
    }
    if not path.exists():
        row["missing"] = True
        return row
    data = _read_json(path)
    metrics = _test_metrics(data)
    row.update(
        {
            "missing": False,
            "rmse": _metric_value(metrics, "rmse"),
            "mae": _metric_value(metrics, "mae"),
            "relative_l2": _metric_value(metrics, "relative_l2"),
            "hot_q90_rmse": _region_rmse(metrics, "hot_q90"),
            "gradient_q90_rmse": _region_rmse(metrics, "gradient_q90"),
        }
    )
    row.update(_profile_metadata(data))
    return row


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


def _deltas(aggregate: dict[str, Any], baseline_method: str, candidate_methods: list[str]) -> dict[str, Any]:
    baseline = aggregate.get(baseline_method) or {}
    deltas: dict[str, Any] = {}
    for method in candidate_methods:
        candidate = aggregate.get(method) or {}
        row: dict[str, Any] = {}
        for metric in ("rmse", "hot_q90_rmse", "gradient_q90_rmse", "mae", "relative_l2"):
            baseline_mean = baseline.get(f"{metric}_mean")
            candidate_mean = candidate.get(f"{metric}_mean")
            if baseline_mean is None or candidate_mean is None:
                row[f"{metric}_mean_delta"] = None
            else:
                row[f"{metric}_mean_delta"] = float(candidate_mean) - float(baseline_mean)
        deltas[method] = row
    return deltas


def collect_summary(
    root: Path,
    splits: list[str],
    seeds: list[int],
    tags: list[str],
    dataset_limit: int,
    dataset_order: str,
) -> dict[str, Any]:
    methods = ["broad_process_v1", *tags]
    summary: dict[str, Any] = {
        "dataset_limit": dataset_limit,
        "dataset_order": dataset_order,
        "splits": splits,
        "seeds": seeds,
        "methods": methods,
        "rows": [],
        "aggregate": {},
        "deltas_vs_broad_process_v1": {},
    }
    for split in splits:
        base_run_id = _base_run_id(split, dataset_limit, dataset_order)
        split_rows = []
        for tag in ("broad_process_profile", *tags):
            method_rows = []
            for seed in seeds:
                row = _metrics_row(root, split, base_run_id, seed, tag)
                method_rows.append(row)
                split_rows.append(row)
                summary["rows"].append(row)
            method = "broad_process_v1" if tag == "broad_process_profile" else tag
            summary.setdefault("aggregate", {}).setdefault(split, {})[method] = _aggregate(method_rows)
        summary["deltas_vs_broad_process_v1"][split] = _deltas(
            summary["aggregate"][split],
            "broad_process_v1",
            tags,
        )
    return summary


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def print_markdown(summary: dict[str, Any]) -> None:
    print("| split | method | seed | test RMSE | hot q90 RMSE | gradient q90 RMSE | graph |")
    print("|---|---|---:|---:|---:|---:|---|")
    for row in summary["rows"]:
        if row.get("missing"):
            print(f"| {row['split']} | {row['method']} | {row['seed']} | MISSING |  |  | {row['path']} |")
            continue
        graph = ""
        if row.get("process_graph_enabled"):
            graph = "{}:{}@{}".format(
                row.get("process_graph_mode"),
                row.get("process_graph_anchor_count"),
                row.get("process_graph_fit_scope"),
            )
        print(
            "| {split} | {method} | {seed} | {rmse} | {hot} | {grad} | {graph} |".format(
                split=row["split"],
                method=row["method"],
                seed=row["seed"],
                rmse=_fmt(row.get("rmse")),
                hot=_fmt(row.get("hot_q90_rmse")),
                grad=_fmt(row.get("gradient_q90_rmse")),
                graph=graph,
            )
        )
    print()
    print("| split | method | n | test RMSE mean +/- std | hot q90 mean +/- std | gradient q90 mean +/- std |")
    print("|---|---|---:|---:|---:|---:|")
    for split, methods in summary["aggregate"].items():
        for method, row in methods.items():
            print(
                "| {split} | {method} | {n} | {rmse} +/- {rmse_std} | {hot} +/- {hot_std} | {grad} +/- {grad_std} |".format(
                    split=split,
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
    print()
    print("| split | method | delta test RMSE | delta hot q90 RMSE | delta gradient q90 RMSE |")
    print("|---|---|---:|---:|---:|")
    for split, methods in summary["deltas_vs_broad_process_v1"].items():
        for method, row in methods.items():
            print(
                "| {split} | {method} | {rmse} | {hot} | {grad} |".format(
                    split=split,
                    method=method,
                    rmse=_fmt(row.get("rmse_mean_delta")),
                    hot=_fmt(row.get("hot_q90_rmse_mean_delta")),
                    grad=_fmt(row.get("gradient_q90_rmse_mean_delta")),
                )
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Project root containing outputs/")
    parser.add_argument("--dataset-limit", type=int, default=12)
    parser.add_argument("--dataset-order", default="process_round_robin")
    parser.add_argument("--split", action="append", choices=VALID_SPLITS)
    parser.add_argument("--seed", type=int, action="append", help="Seed to include. Defaults to 7, 1, 2.")
    parser.add_argument("--tag", action="append", help="Process-graph run tag. Defaults to pg_rbf_global.")
    parser.add_argument("--json-output", help="Optional path to write JSON summary.")
    parser.add_argument("--require-complete", action="store_true", help="Fail if any expected metrics file is missing.")
    args = parser.parse_args()

    splits = args.split if args.split else list(DEFAULT_SPLITS)
    seeds = args.seed if args.seed else list(DEFAULT_SEEDS)
    tags = args.tag if args.tag else list(DEFAULT_TAGS)
    summary = collect_summary(
        Path(args.root),
        splits,
        seeds,
        tags,
        args.dataset_limit,
        args.dataset_order,
    )
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
