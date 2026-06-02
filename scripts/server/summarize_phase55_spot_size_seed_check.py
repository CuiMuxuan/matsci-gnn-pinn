#!/usr/bin/env python3
"""Summarize Phase 55 spot-size route-guard seed validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METRIC_KEYS = ("rmse", "hot_q90_rmse", "gradient_q90_rmse")
STRONG_BASELINES = (
    ("mean", "mean_constant"),
    ("knn_coords", "knn_coords"),
    ("knn_process", "knn_process"),
    ("extra_trees_coords", "extra_trees_coords"),
    ("extra_trees_process", "extra_trees_process"),
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _run_id(split: str, dataset_limit: int, dataset_order: str, profile_tag: str) -> str:
    return (
        "ambench_multiline_process_temperature_"
        f"broad{dataset_limit}_{dataset_order}_{split}_{profile_tag}_smoke_a100_sxm4_40gb_v1"
    )


def _baseline_path(root: Path, run_id: str, tag: str) -> Path:
    return root / "outputs" / "baselines" / f"{run_id}_{tag}_regions_q90.json"


def _seed_metrics_path(root: Path, base_run_id: str, seed: int, run_tag: str) -> Path:
    if seed == 7:
        return root / "outputs" / "runs" / f"{base_run_id}_macro_pinn_minmax_{run_tag}_v1" / "metrics.json"
    return root / "outputs" / "runs" / f"{base_run_id}_seed{seed}_macro_pinn_minmax_{run_tag}_v1" / "metrics.json"


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
    primary = metrics.get("metrics") or metrics
    value = primary.get(name)
    return float(value) if value is not None else None


def _region_rmse(metrics: dict[str, Any], name: str) -> float | None:
    regions = metrics.get("region_metrics") or metrics.get("regions") or {}
    region = regions.get(name) or {}
    region_metrics = region.get("metrics") or region
    value = region_metrics.get("rmse")
    return float(value) if value is not None else None


def _extract_metric_fields(data: dict[str, Any]) -> dict[str, float | None]:
    metrics = _test_metrics(data)
    return {
        "rmse": _metric_value(metrics, "rmse"),
        "hot_q90_rmse": _region_rmse(metrics, "hot_q90"),
        "gradient_q90_rmse": _region_rmse(metrics, "gradient_q90"),
    }


def _metrics_row(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"path": str(path)}
    if not path.exists():
        row["missing"] = True
        return row
    data = _read_json(path)
    row.update(_extract_metric_fields(data))
    row["missing"] = any(row.get(metric) is None for metric in METRIC_KEYS)
    row["metadata"] = _route_metadata(data)
    return row


def _route_metadata(data: dict[str, Any]) -> dict[str, Any]:
    features = data.get("input_features") or {}
    profile = features.get("conditioning_profile") or {}
    selected = profile.get("selected") or {}
    effective = profile.get("effective") or {}
    return {
        "conditioning_profile": profile.get("profile"),
        "selected_conditioning_mode": selected.get("conditioning_mode"),
        "selected_feature_normalization": selected.get("feature_normalization"),
        "effective_conditioning_mode": effective.get("conditioning_mode"),
        "effective_feature_columns": effective.get("feature_columns"),
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _pstdev(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _aggregate(rows: list[dict[str, Any]], expected_n: int) -> dict[str, Any]:
    present = [row for row in rows if not row.get("missing")]
    result: dict[str, Any] = {
        "n": len(present),
        "expected_n": expected_n,
        "complete": len(present) == expected_n,
    }
    for metric in METRIC_KEYS:
        values = [float(row[metric]) for row in present if row.get(metric) is not None]
        result[metric] = {
            "mean": _mean(values),
            "pstdev": _pstdev(values),
        }
    return result


def _best_strong_baselines(rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    best: dict[str, dict[str, Any] | None] = {}
    for metric in METRIC_KEYS:
        candidates = [
            {"method": method, "value": row.get(metric)}
            for method, row in rows.items()
            if not row.get("missing") and row.get(metric) is not None
        ]
        best[metric] = min(candidates, key=lambda item: float(item["value"])) if candidates else None
    return best


def _route_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        metadata = row.get("metadata") or {}
        if metadata.get("selected_conditioning_mode") or metadata.get("effective_conditioning_mode"):
            return metadata
    return {}


def _metric_gate(
    broad_value: float | None,
    strong_value: float | None,
    no_process_value: float | None,
) -> dict[str, Any]:
    beats_strong = broad_value is not None and strong_value is not None and broad_value < strong_value
    beats_no_process = (
        broad_value is not None and no_process_value is not None and broad_value < no_process_value
    )
    return {
        "candidate": broad_value,
        "best_strong_baseline": strong_value,
        "no_process": no_process_value,
        "beats_best_strong_baseline": beats_strong,
        "beats_no_process": beats_no_process,
        "delta_vs_best_strong": (broad_value - strong_value)
        if broad_value is not None and strong_value is not None
        else None,
        "delta_vs_no_process": (broad_value - no_process_value)
        if broad_value is not None and no_process_value is not None
        else None,
    }


def _paired_seed_gate(
    seeds: list[int],
    broad_rows: list[dict[str, Any]],
    no_process_rows: list[dict[str, Any]],
    best_strong: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    by_seed = {
        "broad_process_v1": {int(row["seed"]): row for row in broad_rows},
        "no_process": {int(row["seed"]): row for row in no_process_rows},
    }
    seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        broad = by_seed["broad_process_v1"].get(seed, {})
        no_process = by_seed["no_process"].get(seed, {})
        metrics: dict[str, Any] = {}
        for metric in METRIC_KEYS:
            strong = best_strong.get(metric)
            metrics[metric] = _metric_gate(
                broad.get(metric),
                strong.get("value") if strong else None,
                no_process.get(metric),
            )
        seed_rows.append(
            {
                "seed": seed,
                "complete": not broad.get("missing") and not no_process.get("missing"),
                "metrics": metrics,
                "pass": all(
                    values["beats_best_strong_baseline"] and values["beats_no_process"]
                    for values in metrics.values()
                ),
            }
        )
    return {
        "seeds": seed_rows,
        "pass": all(row["complete"] and row["pass"] for row in seed_rows),
    }


def _status(complete: bool, aggregate_pass: bool, paired_pass: bool) -> str:
    if not complete:
        return "incomplete"
    if paired_pass:
        return "seed_robust_transfer_positive"
    if aggregate_pass:
        return "aggregate_positive_seed_mixed"
    return "seed_unstable_or_negative"


def collect_dataset(
    root: Path,
    dataset_limit: int,
    dataset_order: str,
    split: str,
    seeds: list[int],
    profile_tag: str,
) -> dict[str, Any]:
    base_run_id = _run_id(split, dataset_limit, dataset_order, profile_tag)
    baselines = {
        method: _metrics_row(_baseline_path(root, base_run_id, tag))
        for method, tag in STRONG_BASELINES
    }
    best_strong = _best_strong_baselines(baselines)

    method_rows: dict[str, list[dict[str, Any]]] = {"no_process": [], "broad_process_v1": []}
    for seed in seeds:
        for method, run_tag in (("no_process", "no_process"), ("broad_process_v1", profile_tag)):
            row = _metrics_row(_seed_metrics_path(root, base_run_id, seed, run_tag))
            row["seed"] = seed
            row["method"] = method
            method_rows[method].append(row)

    aggregates = {
        method: _aggregate(rows, expected_n=len(seeds))
        for method, rows in method_rows.items()
    }
    aggregate_metrics: dict[str, Any] = {}
    for metric in METRIC_KEYS:
        strong = best_strong.get(metric)
        aggregate_metrics[metric] = _metric_gate(
            aggregates["broad_process_v1"][metric]["mean"],
            strong.get("value") if strong else None,
            aggregates["no_process"][metric]["mean"],
        )
        if strong:
            aggregate_metrics[metric]["best_strong_baseline_method"] = strong.get("method")

    complete = all(not row.get("missing") for row in baselines.values()) and all(
        aggregate["complete"] for aggregate in aggregates.values()
    )
    aggregate_gate_pass = complete and all(
        values["beats_best_strong_baseline"] and values["beats_no_process"]
        for values in aggregate_metrics.values()
    )
    paired = _paired_seed_gate(
        seeds=seeds,
        broad_rows=method_rows["broad_process_v1"],
        no_process_rows=method_rows["no_process"],
        best_strong=best_strong,
    )
    return {
        "label": f"broad{dataset_limit}",
        "dataset_limit": dataset_limit,
        "dataset_order": dataset_order,
        "split": split,
        "base_run_id": base_run_id,
        "seeds": seeds,
        "route": _route_from_rows(method_rows["broad_process_v1"]),
        "strong_baselines": baselines,
        "best_strong_baseline_by_metric": best_strong,
        "rows": {
            "no_process": method_rows["no_process"],
            "broad_process_v1": method_rows["broad_process_v1"],
        },
        "aggregates": aggregates,
        "aggregate_gate": {
            "complete": complete,
            "pass": aggregate_gate_pass,
            "metrics": aggregate_metrics,
        },
        "paired_seed_gate": paired,
        "status": _status(complete, aggregate_gate_pass, paired["pass"]),
    }


def collect_summary(
    root: Path,
    dataset_limits: list[int],
    dataset_order: str,
    split: str,
    seeds: list[int],
    profile_tag: str = "broad_process_profile",
) -> dict[str, Any]:
    datasets = [
        collect_dataset(
            root=root,
            dataset_limit=dataset_limit,
            dataset_order=dataset_order,
            split=split,
            seeds=seeds,
            profile_tag=profile_tag,
        )
        for dataset_limit in dataset_limits
    ]
    aggregate_transfer_positive = all(dataset["aggregate_gate"]["pass"] for dataset in datasets)
    paired_seed_transfer_positive = all(dataset["paired_seed_gate"]["pass"] for dataset in datasets)
    if paired_seed_transfer_positive:
        paper_claim_status = "seed_robust_transfer_positive"
    elif aggregate_transfer_positive:
        paper_claim_status = "aggregate_positive_seed_mixed"
    elif any(dataset["status"] == "incomplete" for dataset in datasets):
        paper_claim_status = "incomplete"
    else:
        paper_claim_status = "seed_unstable_or_negative"
    return {
        "phase": 55,
        "objective": (
            "Spot-size transferable process-conditioned route-guard seed validation "
            "against strong baselines and no-process Macro PINN."
        ),
        "dataset_limits": dataset_limits,
        "dataset_order": dataset_order,
        "split": split,
        "seeds": seeds,
        "target_method": "broad_process_v1",
        "strong_baseline_methods": [method for method, _ in STRONG_BASELINES],
        "required_metrics": list(METRIC_KEYS),
        "datasets": datasets,
        "transfer_gate": {
            "aggregate_transfer_positive": aggregate_transfer_positive,
            "paired_seed_transfer_positive": paired_seed_transfer_positive,
            "paper_claim_status": paper_claim_status,
        },
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _metric_mean_std(row: dict[str, Any], metric: str) -> str:
    values = row.get(metric) or {}
    return f"{_fmt(values.get('mean'))} +/- {_fmt(values.get('pstdev'))}"


def markdown_text(summary: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Phase 55 Spot-Size Route Seed Validation",
        "",
        f"Transfer gate: `{summary['transfer_gate']['paper_claim_status']}`",
        "",
        "| dataset | route | aggregate gate | paired seed gate |",
        "|---|---|---|---|",
    ]
    for dataset in summary["datasets"]:
        route = dataset.get("route") or {}
        route_text = "{}/{}".format(
            route.get("selected_conditioning_mode") or "",
            route.get("selected_feature_normalization") or "",
        ).strip("/")
        lines.append(
            "| {label} | {route} | {aggregate} | {paired} |".format(
                label=dataset["label"],
                route=route_text,
                aggregate=_fmt(dataset["aggregate_gate"]["pass"]),
                paired=_fmt(dataset["paired_seed_gate"]["pass"]),
            )
        )
    lines.extend(["", "## Aggregate Metrics", ""])
    lines.append(
        "| dataset | method | n | test RMSE mean +/- std | hot q90 mean +/- std | gradient q90 mean +/- std |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")
    for dataset in summary["datasets"]:
        for method in ("no_process", "broad_process_v1"):
            aggregate = dataset["aggregates"][method]
            lines.append(
                "| {dataset} | {method} | {n} | {rmse} | {hot} | {grad} |".format(
                    dataset=dataset["label"],
                    method=method,
                    n=aggregate["n"],
                    rmse=_metric_mean_std(aggregate, "rmse"),
                    hot=_metric_mean_std(aggregate, "hot_q90_rmse"),
                    grad=_metric_mean_std(aggregate, "gradient_q90_rmse"),
                )
            )
    lines.extend(["", "## Strong-Baseline Deltas", ""])
    lines.append(
        "| dataset | metric | broad mean | best strong baseline | no-process mean | delta vs strong | delta vs no-process |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for dataset in summary["datasets"]:
        for metric, row in dataset["aggregate_gate"]["metrics"].items():
            best_method = row.get("best_strong_baseline_method") or ""
            lines.append(
                "| {dataset} | {metric} | {candidate} | {strong} ({method}) | {noproc} | {dstrong} | {dnoproc} |".format(
                    dataset=dataset["label"],
                    metric=metric,
                    candidate=_fmt(row.get("candidate")),
                    strong=_fmt(row.get("best_strong_baseline")),
                    method=best_method,
                    noproc=_fmt(row.get("no_process")),
                    dstrong=_fmt(row.get("delta_vs_best_strong")),
                    dnoproc=_fmt(row.get("delta_vs_no_process")),
                )
            )
    lines.extend(["", "## Per-Seed Metrics", ""])
    lines.append("| dataset | method | seed | test RMSE | hot q90 RMSE | gradient q90 RMSE |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for dataset in summary["datasets"]:
        for method in ("no_process", "broad_process_v1"):
            for row in dataset["rows"][method]:
                if row.get("missing"):
                    lines.append(f"| {dataset['label']} | {method} | {row['seed']} | MISSING |  |  |")
                    continue
                lines.append(
                    "| {dataset} | {method} | {seed} | {rmse} | {hot} | {grad} |".format(
                        dataset=dataset["label"],
                        method=method,
                        seed=row["seed"],
                        rmse=_fmt(row.get("rmse")),
                        hot=_fmt(row.get("hot_q90_rmse")),
                        grad=_fmt(row.get("gradient_q90_rmse")),
                    )
                )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root containing outputs/.")
    parser.add_argument("--dataset-limit", type=int, action="append", dest="dataset_limits")
    parser.add_argument("--dataset-order", default="process_round_robin")
    parser.add_argument("--split", default="spot_size")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--profile-tag", default="broad_process_profile")
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    dataset_limits = args.dataset_limits if args.dataset_limits else [12, 21]
    seeds = args.seeds if args.seeds is not None else [7, 1, 2]
    summary = collect_summary(
        root=Path(args.root),
        dataset_limits=dataset_limits,
        dataset_order=args.dataset_order,
        split=args.split,
        seeds=seeds,
        profile_tag=args.profile_tag,
    )
    markdown = markdown_text(summary)
    print(markdown)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote: {output}")
    if args.markdown_output:
        output = Path(args.markdown_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        print(f"Wrote: {output}")
    if args.require_complete and any(dataset["status"] == "incomplete" for dataset in summary["datasets"]):
        return 3
    if args.require_pass and not summary["transfer_gate"]["paired_seed_transfer_positive"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
