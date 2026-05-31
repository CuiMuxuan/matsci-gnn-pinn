#!/usr/bin/env python3
"""Summarize Phase 25 process-conditioned FiLM metrics from server artifacts."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_SPLITS = ("line", "laser_power", "scan_speed", "spot_size", "process")

PINN_SERIES = (
    ("phase24_no_process", "{split}_holdout_a100_sxm4_40gb_v1", "no_process"),
    ("phase24_concat", "{split}_holdout_a100_sxm4_40gb_v1", "process_features"),
    ("film_v1_no_process", "{split}_film_a100_sxm4_40gb_v1", "no_process"),
    ("film_v1_train_minmax", "{split}_film_a100_sxm4_40gb_v1", "process_film"),
    ("film_v2_no_process", "{split}_film_global_standard_a100_sxm4_40gb_v1", "no_process"),
    (
        "film_v2_global_standard",
        "{split}_film_global_standard_a100_sxm4_40gb_v1",
        "process_film_global_standard",
    ),
    (
        "concat_global_standard",
        "{split}_concat_global_standard_a100_sxm4_40gb_v1",
        "process_concat_global_standard",
    ),
    (
        "concat_film_global_standard",
        "{split}_concat_film_global_standard_a100_sxm4_40gb_v1",
        "process_concat_film_global_standard",
    ),
    (
        "concat_film_0_25_global_standard",
        "{split}_concat_film_strength0_25_global_standard_a100_sxm4_40gb_v1",
        "process_concat_film_strength0_25_global_standard",
    ),
    (
        "routed_concat_prior_global_standard",
        "{split}_routed_concat_prior_global_standard_a100_sxm4_40gb_v1",
        "process_routed_concat_prior_global_standard",
    ),
    (
        "routed_film_prior_global_standard",
        "{split}_routed_film_prior_global_standard_a100_sxm4_40gb_v1",
        "process_routed_film_prior_global_standard",
    ),
    (
        "process_axis_profile",
        "{split}_process_axis_profile_a100_sxm4_40gb_v1",
        "process_axis_profile",
    ),
)

BASELINE_SERIES = (
    ("baseline_mean", "mean_constant"),
    ("baseline_knn_coords", "knn_coords"),
    ("baseline_knn_process", "knn_process"),
    ("baseline_extra_trees_coords", "extra_trees_coords"),
    ("baseline_extra_trees_process", "extra_trees_process"),
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


def _region_rmse(metrics: dict[str, Any], name: str) -> float | None:
    regions = metrics.get("region_metrics") or metrics.get("regions") or {}
    region = regions.get(name) or {}
    region_metrics = region.get("metrics") or region
    value = region_metrics.get("rmse")
    return float(value) if value is not None else None


def _metric_value(metrics: dict[str, Any], name: str) -> float | None:
    primary_metrics = metrics.get("metrics") or metrics
    value = primary_metrics.get(name)
    return float(value) if value is not None else None


def _metric_fields(metrics: dict[str, Any]) -> dict[str, float | None]:
    return {
        "rmse": _metric_value(metrics, "rmse"),
        "mae": _metric_value(metrics, "mae"),
        "relative_l2": _metric_value(metrics, "relative_l2"),
        "hot_q90_rmse": _region_rmse(metrics, "hot_q90"),
        "gradient_q90_rmse": _region_rmse(metrics, "gradient_q90"),
    }


def _baseline_paths(root: Path, split: str, tag: str) -> list[Path]:
    candidates = (
        f"ambench_multiline_process_temperature_{split}_process_axis_profile_a100_sxm4_40gb_v1",
        f"ambench_multiline_process_temperature_{split}_film_global_standard_a100_sxm4_40gb_v1",
        f"ambench_multiline_process_temperature_{split}_holdout_a100_sxm4_40gb_v1",
    )
    return [root / "outputs" / "baselines" / f"{run_id}_{tag}_regions_q90.json" for run_id in candidates]


def collect_rows(root: Path, splits: tuple[str, ...] = DEFAULT_SPLITS) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in splits:
        for method, run_template, tag in PINN_SERIES:
            run_id = "ambench_multiline_process_temperature_" + run_template.format(split=split)
            path = root / "outputs" / "runs" / f"{run_id}_macro_pinn_minmax_{tag}_v1" / "metrics.json"
            row: dict[str, Any] = {"split": split, "method": method, "path": str(path)}
            if not path.exists():
                row["missing"] = True
                rows.append(row)
                continue
            data = _read_json(path)
            features = data.get("input_features") or {}
            normalization = features.get("normalization") or {}
            route = features.get("route") or {}
            route_summary = route.get("summary") or {}
            profile = features.get("conditioning_profile") or {}
            selected_profile = profile.get("selected") or {}
            row.update(_metric_fields(_test_metrics(data)))
            row.update(
                {
                    "conditioning_mode": features.get("conditioning_mode"),
                    "film_strength": features.get("film_strength"),
                    "feature_normalization_mode": normalization.get("mode"),
                    "feature_normalization_fit_scope": normalization.get("fit_scope"),
                    "route_film_prior": route.get("film_prior"),
                    "route_trainable": route.get("trainable"),
                    "route_film_gate_mean": route_summary.get("film_gate_mean"),
                    "route_film_gate_min": route_summary.get("film_gate_min"),
                    "route_film_gate_max": route_summary.get("film_gate_max"),
                    "conditioning_profile": profile.get("profile"),
                    "conditioning_profile_group_key": profile.get("group_key"),
                    "conditioning_profile_selected_mode": selected_profile.get("conditioning_mode"),
                    "conditioning_profile_selected_norm": selected_profile.get("feature_normalization"),
                }
            )
            rows.append(row)

        for method, tag in BASELINE_SERIES:
            paths = _baseline_paths(root, split, tag)
            path = next((candidate for candidate in paths if candidate.exists()), paths[0])
            row = {"split": split, "method": method, "path": str(path)}
            if not path.exists():
                row["missing"] = True
                rows.append(row)
                continue
            data = _read_json(path)
            row.update(_metric_fields(_test_metrics(data)))
            rows.append(row)
    return rows


def collect_seed_rows(root: Path) -> list[dict[str, Any]]:
    specs = (
        (
            "laser_power",
            "no_process",
            "ambench_multiline_process_temperature_laser_power_process_axis_profile_a100_sxm4_40gb_v1",
            "no_process",
        ),
        (
            "laser_power",
            "process_axis_profile",
            "ambench_multiline_process_temperature_laser_power_process_axis_profile_a100_sxm4_40gb_v1",
            "process_axis_profile",
        ),
        (
            "scan_speed",
            "no_process",
            "ambench_multiline_process_temperature_scan_speed_concat_global_standard_a100_sxm4_40gb_v1",
            "no_process",
        ),
        (
            "scan_speed",
            "concat_global_standard",
            "ambench_multiline_process_temperature_scan_speed_concat_global_standard_a100_sxm4_40gb_v1",
            "process_concat_global_standard",
        ),
        (
            "spot_size",
            "no_process",
            "ambench_multiline_process_temperature_spot_size_film_global_standard_a100_sxm4_40gb_v1",
            "no_process",
        ),
        (
            "spot_size",
            "film_global_standard",
            "ambench_multiline_process_temperature_spot_size_film_global_standard_a100_sxm4_40gb_v1",
            "process_film_global_standard",
        ),
    )
    seed_paths = (
        (7, "{run_id}_macro_pinn_minmax_{tag}_v1"),
        (1, "{run_id}_seed1_macro_pinn_minmax_{tag}_v1"),
        (2, "{run_id}_seed2_macro_pinn_minmax_{tag}_v1"),
    )
    rows: list[dict[str, Any]] = []
    for split, method, run_id, tag in specs:
        values: list[dict[str, Any]] = []
        for seed, template in seed_paths:
            path = root / "outputs" / "runs" / template.format(run_id=run_id, tag=tag) / "metrics.json"
            if not path.exists():
                values.append({"seed": seed, "missing": True, "path": str(path)})
                continue
            metrics = _metric_fields(_test_metrics(_read_json(path)))
            values.append({"seed": seed, "path": str(path), **metrics})
        row: dict[str, Any] = {"split": split, "method": method, "seeds": values}
        present = [value for value in values if not value.get("missing")]
        for field in ("rmse", "hot_q90_rmse", "gradient_q90_rmse"):
            field_values = [float(value[field]) for value in present if value.get(field) is not None]
            if field_values:
                row[f"{field}_mean"] = statistics.fmean(field_values)
                row[f"{field}_std"] = statistics.pstdev(field_values)
        rows.append(row)
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def print_markdown(rows: list[dict[str, Any]]) -> None:
    print("| split | method | test RMSE | hot q90 RMSE | gradient q90 RMSE | mode | film strength | feature norm | profile | selected | route prior | route film gate |")
    print("|---|---|---:|---:|---:|---|---:|---|---|---|---:|---:|")
    for row in rows:
        if row.get("missing"):
            print(f"| {row['split']} | {row['method']} | MISSING |  |  |  |  | {row['path']} |  |  |  |  |")
            continue
        selected = ""
        if row.get("conditioning_profile_selected_mode"):
            selected = "{}/{}".format(
                row.get("conditioning_profile_selected_mode"),
                row.get("conditioning_profile_selected_norm"),
            )
        print(
            "| {split} | {method} | {rmse} | {hot} | {grad} | {mode} | {strength} | {norm} | {profile} | {selected} | {prior} | {gate} |".format(
                split=row["split"],
                method=row["method"],
                rmse=_fmt(row.get("rmse")),
                hot=_fmt(row.get("hot_q90_rmse")),
                grad=_fmt(row.get("gradient_q90_rmse")),
                mode=_fmt(row.get("conditioning_mode")),
                strength=_fmt(row.get("film_strength")),
                norm=_fmt(row.get("feature_normalization_mode")),
                profile=_fmt(row.get("conditioning_profile")),
                selected=selected,
                prior=_fmt(row.get("route_film_prior")),
                gate=_fmt(row.get("route_film_gate_mean")),
            )
        )


def print_seed_markdown(rows: list[dict[str, Any]]) -> None:
    print("| split | method | seeds | test RMSE mean +/- std | hot q90 mean +/- std | gradient q90 mean +/- std |")
    print("|---|---|---|---:|---:|---:|")
    for row in rows:
        seeds = ",".join(str(value["seed"]) for value in row["seeds"] if not value.get("missing"))
        print(
            "| {split} | {method} | {seeds} | {rmse} +/- {rmse_std} | {hot} +/- {hot_std} | {grad} +/- {grad_std} |".format(
                split=row["split"],
                method=row["method"],
                seeds=seeds,
                rmse=_fmt(row.get("rmse_mean")),
                rmse_std=_fmt(row.get("rmse_std")),
                hot=_fmt(row.get("hot_q90_rmse_mean")),
                hot_std=_fmt(row.get("hot_q90_rmse_std")),
                grad=_fmt(row.get("gradient_q90_rmse_mean")),
                grad_std=_fmt(row.get("gradient_q90_rmse_std")),
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Project root containing outputs/")
    parser.add_argument("--json-output", help="Optional path to write collected rows as JSON")
    parser.add_argument("--seed-check", action="store_true", help="Summarize focused Phase 25 seed-check runs")
    parser.add_argument(
        "--split",
        action="append",
        choices=DEFAULT_SPLITS,
        help="Limit the regular artifact summary to one or more process-axis splits.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    splits = tuple(args.split) if args.split else DEFAULT_SPLITS
    rows = collect_seed_rows(root) if args.seed_check else collect_rows(root, splits)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    if args.seed_check:
        print_seed_markdown(rows)
    else:
        print_markdown(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
