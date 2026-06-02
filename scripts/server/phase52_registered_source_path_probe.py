"""Phase 52 registered source-path feature gate.

The first responsibility of this probe is data compatibility. AM-Bench scan
strategy XYPT coordinates are commanded galvo positions in millimeters; the
current thermography field tables use camera pixel indices. A registered
source-path feature is only paper-facing if the scan object and coordinate
system can be aligned without using test labels.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gnnpinn.eval.baselines import regression_metric_table


Z90 = 1.6448536269514722


@dataclass(frozen=True)
class XyptPath:
    name: str
    x: np.ndarray
    y: np.ndarray
    power: np.ndarray
    trigger: np.ndarray
    power_on: np.ndarray


def _load_phase46_module():
    module_path = Path(__file__).with_name("phase46_bayesian_inverse_closure_probe.py")
    module_spec = importlib.util.spec_from_file_location("phase46_probe", module_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def _h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("h5py is required to inspect AM-Bench XYPT scan strategy files") from exc
    return h5py


def _scale01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = float(np.nanmax(values) - np.nanmin(values)) if values.size else 0.0
    if span <= 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.nanmin(values))) / span


def read_xypt_paths(path: Path) -> dict[str, XyptPath]:
    h5py = _h5py()
    paths: dict[str, XyptPath] = {}
    with h5py.File(path, "r") as handle:
        for group_name in ["XYPT/Xpad", "XYPT/Ypad"]:
            if group_name not in handle:
                continue
            group = handle[group_name]
            x = np.asarray(group["X"][()]).reshape(-1).astype(float)
            y = np.asarray(group["Y"][()]).reshape(-1).astype(float)
            power = np.asarray(group["P"][()]).reshape(-1).astype(float)
            trigger = np.asarray(group["T"][()]).reshape(-1).astype(float)
            power_on = power > 1.0
            paths[group_name] = XyptPath(
                name=group_name,
                x=x,
                y=y,
                power=power,
                trigger=trigger,
                power_on=power_on,
            )
    if not paths:
        raise ValueError(f"No XYPT/Xpad or XYPT/Ypad groups found in {path}")
    return paths


def xypt_summary(paths: dict[str, XyptPath]) -> dict[str, Any]:
    return {name: _xypt_path_summary(path) for name, path in paths.items()}


def _xypt_path_summary(path: XyptPath) -> dict[str, Any]:
    segments = _power_on_segments(path)
    return {
        "n_points": int(len(path.x)),
        "power_on_points": int(np.count_nonzero(path.power_on)),
        "x_range": [float(np.nanmin(path.x)), float(np.nanmax(path.x))],
        "y_range": [float(np.nanmin(path.y)), float(np.nanmax(path.y))],
        "power_values": [float(value) for value in sorted(np.unique(path.power).tolist())[:20]],
        "trigger_counts": {
            str(int(value)): int(np.count_nonzero(path.trigger == value))
            for value in sorted(np.unique(path.trigger).tolist())
        },
        "n_power_on_segments": len(segments),
        "first_power_on_segments": segments[:8],
    }


def _power_on_segments(path: XyptPath) -> list[dict[str, Any]]:
    indices = np.flatnonzero(path.power_on)
    if len(indices) == 0:
        return []
    breaks = np.where(np.diff(indices) > 1)[0]
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks], indices[-1]]
    segments = []
    for start, end in zip(starts, ends, strict=True):
        window = slice(int(start), int(end) + 1)
        segments.append(
            {
                "start": int(start),
                "end": int(end),
                "n": int(end - start + 1),
                "x_range": [float(np.nanmin(path.x[window])), float(np.nanmax(path.x[window]))],
                "y_range": [float(np.nanmin(path.y[window])), float(np.nanmax(path.y[window]))],
            }
        )
    return segments


def table_summary(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Table has no rows: {path}")
    numeric: dict[str, list[float]] = {name: [] for name in ["x", "y", "t", "frame_index", "row_index", "col_index"]}
    line_ids: set[str] = set()
    dataset_paths: set[str] = set()
    for row in rows:
        for name in numeric:
            if name in row and row[name] != "":
                numeric[name].append(float(row[name]))
        if row.get("line_id"):
            line_ids.add(row["line_id"])
        if row.get("dataset_path"):
            dataset_paths.add(row["dataset_path"])
    ranges = {}
    for name, values in numeric.items():
        if values:
            ranges[name] = {
                "min": float(min(values)),
                "max": float(max(values)),
                "n_unique": int(len(set(values))),
            }
    return {
        "n_rows": len(rows),
        "line_ids": sorted(line_ids),
        "dataset_paths": sorted(dataset_paths),
        "ranges": ranges,
    }


def select_xypt_path(table: dict[str, Any], paths: dict[str, XyptPath]) -> tuple[str | None, str]:
    joined = " ".join(table.get("line_ids", []) + table.get("dataset_paths", [])).lower()
    normalized = joined.replace("_", "").replace("-", "")
    if "xpad" in normalized and "XYPT/Xpad" in paths:
        return "XYPT/Xpad", "matched x-pad table metadata"
    if "ypad" in normalized and "XYPT/Ypad" in paths:
        return "XYPT/Ypad", "matched y-pad table metadata"
    if "line" in normalized:
        return None, "scan strategy file contains pad XYPT groups but table is a single-track Line_* dataset"
    return None, "table metadata does not identify X-pad or Y-pad"


def coordinate_compatibility(table: dict[str, Any], xypt_path: XyptPath) -> dict[str, Any]:
    x_range = table.get("ranges", {}).get("x") or table.get("ranges", {}).get("col_index")
    y_range = table.get("ranges", {}).get("y") or table.get("ranges", {}).get("row_index")
    if x_range is None or y_range is None:
        return {"compatible": False, "reason": "table lacks x/y coordinate ranges"}
    table_x = [float(x_range["min"]), float(x_range["max"])]
    table_y = [float(y_range["min"]), float(y_range["max"])]
    path_x = [float(np.nanmin(xypt_path.x)), float(np.nanmax(xypt_path.x))]
    path_y = [float(np.nanmin(xypt_path.y)), float(np.nanmax(xypt_path.y))]
    overlap_compatible = _ranges_overlap(table_x, path_x) and _ranges_overlap(table_y, path_y)
    span_compatible = _span_compatible(table_x, path_x) and _span_compatible(table_y, path_y)
    compatible = overlap_compatible and span_compatible
    return {
        "compatible": bool(compatible),
        "table_x_range": table_x,
        "table_y_range": table_y,
        "xypt_x_range": path_x,
        "xypt_y_range": path_y,
        "span_ratio_compatible": bool(span_compatible),
        "reason": "table x/y ranges overlap XYPT millimeter ranges with comparable spans"
        if compatible
        else (
            "table x/y ranges are not safely comparable to XYPT millimeter ranges; "
            "likely camera pixels vs galvo millimeters"
        ),
    }


def _ranges_overlap(left: list[float], right: list[float]) -> bool:
    span = max(right[1] - right[0], 1e-6)
    margin = 0.10 * span
    return not (left[1] < right[0] - margin or left[0] > right[1] + margin)


def _span_compatible(left: list[float], right: list[float]) -> bool:
    left_span = abs(left[1] - left[0])
    right_span = abs(right[1] - right[0])
    if left_span <= 1e-6 or right_span <= 1e-6:
        return True
    ratio = left_span / right_span
    return 0.25 <= ratio <= 4.0


def with_registered_path_features(data: Any, path: XyptPath) -> Any:
    phase46 = _load_phase46_module()
    active = np.flatnonzero(path.power_on)
    if len(active) == 0:
        raise ValueError(f"XYPT path {path.name} has no power-on points")
    path_x = _scale01(path.x[active])
    path_y = _scale01(path.y[active])
    path_progress = np.linspace(0.0, 1.0, len(active))
    x = _scale01(data.cols)
    y = _scale01(data.rows)
    t = _scale01(data.frames)
    features: list[np.ndarray] = []
    names: list[str] = []
    for lag in [0.0, 0.05, 0.15]:
        source_progress = np.clip(t - lag, 0.0, 1.0)
        source_x = np.interp(source_progress, path_progress, path_x)
        source_y = np.interp(source_progress, path_progress, path_y)
        distance2 = (x - source_x) ** 2 + (y - source_y) ** 2
        for width in [0.08, 0.16, 0.32]:
            features.append(np.exp(-0.5 * distance2 / max(width**2, 1e-8)))
            names.append(f"registered_source_lag{lag:g}_w{width:g}")
    min_distance = np.sqrt(
        np.minimum.reduce(
            [
                (x - np.interp(np.clip(t - lag, 0.0, 1.0), path_progress, path_x)) ** 2
                + (y - np.interp(np.clip(t - lag, 0.0, 1.0), path_progress, path_y)) ** 2
                for lag in [0.0, 0.05, 0.15]
            ]
        )
    )
    features.append(1.0 - _scale01(min_distance))
    names.append("registered_source_inverse_distance")
    source_prior = np.maximum(data.source_prior_score, np.max(np.column_stack(features), axis=1))
    return phase46.ProbeData(
        label=f"{data.label}_registered_source_path",
        features=np.column_stack([data.features, *features]),
        target=data.target,
        feature_names=[*data.feature_names, *names],
        splits=data.splits,
        rows=data.rows,
        cols=data.cols,
        frames=data.frames,
        source_prior_score=source_prior,
        true_theta=None,
    )


def _fit_all_train(data: Any, *, prior_variance: float, noise_floor: float) -> dict[str, Any]:
    phase46 = _load_phase46_module()
    train = np.asarray(data.splits["train"], dtype=int)
    posterior = phase46.fit_bayesian_linear(
        data.features[train],
        data.target[train],
        prior_variance=prior_variance,
        noise_variance=None,
        noise_floor=noise_floor,
    )
    mean, std = phase46.posterior_predict(posterior, data.features)
    std, calibration = phase46._conformal_std(data, mean, std, mode="conformal90")
    splits = {
        split: phase46._split_metrics(data, np.asarray(indices, dtype=int), mean, std)
        for split, indices in data.splits.items()
    }
    return {
        "n_features": int(data.features.shape[1]),
        "feature_names": data.feature_names,
        "noise_variance": posterior.noise_variance,
        "calibration": calibration,
        "splits": splits,
    }


def _metric_summary(rows: dict[str, Any]) -> dict[str, float]:
    test = rows["splits"]["test"]
    return {
        "test_rmse": float(test["metrics"]["rmse"]),
        "test_hot_q90_rmse": float(test["region_metrics"]["hot_q90"]["metrics"]["rmse"]),
        "test_gradient_q90_rmse": float(test["region_metrics"]["gradient_q90"]["metrics"]["rmse"]),
        "test_coverage90": float(test["coverage_90"]),
    }


def _gains(base: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    return {
        "test_rmse_gain": base["test_rmse"] - candidate["test_rmse"],
        "test_hot_q90_rmse_gain": base["test_hot_q90_rmse"] - candidate["test_hot_q90_rmse"],
        "test_gradient_q90_rmse_gain": base["test_gradient_q90_rmse"] - candidate["test_gradient_q90_rmse"],
        "test_coverage90_gain": candidate["test_coverage90"] - base["test_coverage90"],
    }


def _decision_from_gains(gain: dict[str, float], registered: dict[str, float]) -> dict[str, Any]:
    region_ok = (
        gain["test_rmse_gain"] >= 0.0
        and gain["test_hot_q90_rmse_gain"] >= 0.0
        and gain["test_gradient_q90_rmse_gain"] >= 0.0
    )
    coverage_ok = 0.75 <= registered["test_coverage90"] <= 1.0
    return {
        "status": "positive" if region_ok and coverage_ok else "negative",
        "region_ok": bool(region_ok),
        "coverage_ok": bool(coverage_ok),
        "interpretation": (
            "Registered source-path features pass the local gate."
            if region_ok and coverage_ok
            else "Registered source-path features do not pass the local gate."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = read_xypt_paths(args.scan_strategy)
    table = table_summary(args.table)
    selected_name, match_reason = select_xypt_path(table, paths)
    compatibility: dict[str, Any] = {
        "selected_xypt_path": selected_name,
        "match_reason": match_reason,
    }
    if selected_name is None:
        return {
            "mode": "registered_source_path",
            "scan_strategy": str(args.scan_strategy),
            "table": str(args.table),
            "xypt_summary": xypt_summary(paths),
            "table_summary": table,
            "compatibility": compatibility,
            "summary": {},
            "decision": {
                "status": "negative",
                "reason": match_reason,
                "interpretation": "Cannot build a paper-facing registered source-path feature for this table.",
            },
        }
    coord = coordinate_compatibility(table, paths[selected_name])
    compatibility["coordinate"] = coord
    if not coord["compatible"] and not args.allow_independent_rescale:
        return {
            "mode": "registered_source_path",
            "scan_strategy": str(args.scan_strategy),
            "table": str(args.table),
            "xypt_summary": xypt_summary(paths),
            "table_summary": table,
            "compatibility": compatibility,
            "summary": {},
            "decision": {
                "status": "negative",
                "reason": coord["reason"],
                "interpretation": "Coordinate systems are not safely registered for a source-path feature gate.",
            },
        }

    phase46 = _load_phase46_module()
    data = phase46.make_table_data(
        table=args.table,
        target=args.target,
        split_manifest=args.split_manifest,
    )
    registered = with_registered_path_features(data, paths[selected_name])
    base_rows = _fit_all_train(data, prior_variance=args.prior_variance, noise_floor=args.noise_floor)
    registered_rows = _fit_all_train(registered, prior_variance=args.prior_variance, noise_floor=args.noise_floor)
    base_summary = _metric_summary(base_rows)
    registered_summary = _metric_summary(registered_rows)
    gain = _gains(base_summary, registered_summary)
    return {
        "mode": "registered_source_path",
        "scan_strategy": str(args.scan_strategy),
        "table": str(args.table),
        "xypt_summary": xypt_summary(paths),
        "table_summary": table,
        "compatibility": compatibility,
        "feature_names": registered.feature_names,
        "summary": {
            "base": base_summary,
            "registered_source_path": registered_summary,
            "gains_vs_base": gain,
        },
        "runs": {
            "base": base_rows,
            "registered_source_path": registered_rows,
        },
        "decision": _decision_from_gains(gain, registered_summary),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-strategy", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--allow-independent-rescale",
        action="store_true",
        help="Allow a diagnostic feature fit after independently scaling table and XYPT coordinates.",
    )
    parser.add_argument("--prior-variance", type=float, default=1e6)
    parser.add_argument("--noise-floor", type=float, default=1.0)
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")
        print(f"Wrote: {args.json_output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
