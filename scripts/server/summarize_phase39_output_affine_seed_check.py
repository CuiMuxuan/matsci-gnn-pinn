"""Summarize Phase 39 output-affine paired seed checks."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRIC_KEYS = ("rmse", "hot_q90_rmse", "gradient_q90_rmse")


def _run_id(split: str, dataset_limit: int, dataset_order: str, tag: str) -> str:
    return (
        "ambench_multiline_process_temperature_"
        f"broad{dataset_limit}_{dataset_order}_{split}_{tag}_smoke_a100_sxm4_40gb_v1"
    )


def _tags_for_seed(seed: int) -> dict[str, tuple[str, str]]:
    if seed == 7:
        return {
            "broad_process_v1": ("broad_process_profile", "broad_process_profile"),
            "broad_output_affine": ("out_affine", "out_affine"),
        }
    return {
        "broad_process_v1": (f"bpv1_s{seed}", f"bpv1_s{seed}"),
        "broad_output_affine": (f"oa_s{seed}", f"oa_s{seed}"),
    }


def _metrics_path(root: Path, run_id: str, run_tag: str) -> Path:
    return root / "outputs" / "runs" / f"{run_id}_macro_pinn_minmax_{run_tag}_v1" / "metrics.json"


def _extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
    test = payload["split_metrics"]["test"]
    metrics = test["metrics"]
    regions = test["region_metrics"]
    return {
        "rmse": float(metrics["rmse"]),
        "hot_q90_rmse": float(regions["hot_q90"]["metrics"]["rmse"]),
        "gradient_q90_rmse": float(regions["gradient_q90"]["metrics"]["rmse"]),
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def collect_summary(root: Path, split: str, dataset_limit: int, dataset_order: str, seeds: list[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for seed in seeds:
        for method, (run_tag, metric_tag) in _tags_for_seed(seed).items():
            rid = _run_id(split, dataset_limit, dataset_order, run_tag)
            path = _metrics_path(root, rid, metric_tag)
            if not path.exists():
                missing.append(str(path))
                continue
            metrics = _extract_metrics(json.loads(path.read_text(encoding="utf-8")))
            rows.append({"seed": seed, "method": method, "path": str(path), **metrics})

    aggregates: dict[str, dict[str, Any]] = {}
    for method in ("broad_process_v1", "broad_output_affine"):
        method_rows = [row for row in rows if row["method"] == method]
        aggregates[method] = {
            "n": len(method_rows),
            **{key: _mean_std([float(row[key]) for row in method_rows]) for key in METRIC_KEYS},
        }

    deltas: dict[str, float | None] = {}
    base = aggregates["broad_process_v1"]
    affine = aggregates["broad_output_affine"]
    for key in METRIC_KEYS:
        if base["n"] and affine["n"]:
            deltas[key] = affine[key]["mean"] - base[key]["mean"]
        else:
            deltas[key] = None

    return {
        "split": split,
        "dataset_limit": dataset_limit,
        "dataset_order": dataset_order,
        "seeds": seeds,
        "rows": rows,
        "aggregates": aggregates,
        "delta_affine_minus_baseline": deltas,
        "missing": missing,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def print_markdown(summary: dict[str, Any]) -> None:
    print("| method | n | test RMSE mean +/- std | hot q90 mean +/- std | gradient q90 mean +/- std |")
    print("|---|---:|---:|---:|---:|")
    for method in ("broad_process_v1", "broad_output_affine"):
        row = summary["aggregates"][method]
        print(
            "| {method} | {n} | {rmse} +/- {rmse_std} | {hot} +/- {hot_std} | {grad} +/- {grad_std} |".format(
                method=method,
                n=row["n"],
                rmse=_fmt(row["rmse"]["mean"]),
                rmse_std=_fmt(row["rmse"]["std"]),
                hot=_fmt(row["hot_q90_rmse"]["mean"]),
                hot_std=_fmt(row["hot_q90_rmse"]["std"]),
                grad=_fmt(row["gradient_q90_rmse"]["mean"]),
                grad_std=_fmt(row["gradient_q90_rmse"]["std"]),
            )
        )
    delta = summary["delta_affine_minus_baseline"]
    print("")
    print(
        "Delta output_affine - broad_process_v1: "
        f"test={_fmt(delta['rmse'])}, hot={_fmt(delta['hot_q90_rmse'])}, gradient={_fmt(delta['gradient_q90_rmse'])}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root containing outputs/.")
    parser.add_argument("--split", default="laser_power")
    parser.add_argument("--dataset-limit", type=int, default=12)
    parser.add_argument("--dataset-order", default="process_round_robin")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--json-output")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else [7, 1, 2]
    summary = collect_summary(Path(args.root), args.split, args.dataset_limit, args.dataset_order, seeds)
    print_markdown(summary)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote: {output}")
    if args.require_complete and summary["missing"]:
        for path in summary["missing"]:
            print(f"MISSING: {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
