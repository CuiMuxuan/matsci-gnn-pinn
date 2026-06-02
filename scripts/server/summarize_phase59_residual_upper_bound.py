#!/usr/bin/env python3
"""No-test-leakage residual upper-bound probe for Phase 59."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_GROUP_FIELDS = (
    "line_id",
    "laser_power_W",
    "scan_speed_mm_s",
    "spot_size_um",
    "process_tuple",
    "time_bin",
    "frame_bin",
)
DEFAULT_BLEND_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _is_float(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _read_prediction_csv(path: Path, target: str) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")
    records: dict[int, dict[str, Any]] = {}
    predictions: dict[int, float] = {}
    for row in rows:
        if target not in row:
            raise ValueError(f"Target column {target!r} missing in {path}")
        index = int(row["row_index"])
        records[index] = row
        predictions[index] = float(row["prediction"])
    return {
        "method": rows[0].get("method") or path.stem,
        "path": str(path),
        "records": records,
        "predictions": predictions,
    }


def _aligned_records(prediction_sets: dict[str, dict[str, Any]], target: str) -> list[dict[str, Any]]:
    labels = list(prediction_sets)
    first = prediction_sets[labels[0]]
    row_indices = sorted(first["records"])
    for label in labels[1:]:
        mismatch = set(row_indices).symmetric_difference(prediction_sets[label]["records"])
        if mismatch:
            raise ValueError(f"Prediction set {label!r} is not row-aligned; mismatch count={len(mismatch)}")
    rows: list[dict[str, Any]] = []
    for index in row_indices:
        row = dict(first["records"][index])
        row["row_index"] = int(row["row_index"])
        row[target] = float(row[target])
        for label in labels:
            row[f"prediction:{label}"] = prediction_sets[label]["predictions"][index]
        rows.append(row)
    return rows


def _quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute quantile of empty values")
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    value = ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    return min(max(value, ordered[0]), ordered[-1])


def _axis_neighbors(values: list[float]) -> dict[float, list[float]]:
    output: dict[float, list[float]] = {}
    for index, value in enumerate(values):
        neighbors: list[float] = []
        if index > 0:
            neighbors.append(values[index - 1])
        if index + 1 < len(values):
            neighbors.append(values[index + 1])
        output[value] = neighbors
    return output


def _spatial_gradient_scores(rows: list[dict[str, Any]], target: str) -> dict[int, float]:
    if not rows:
        return {}
    frame_column = "frame_index" if "frame_index" in rows[0] else "t" if "t" in rows[0] else "time"
    row_column = "metadata_row_index" if "metadata_row_index" in rows[0] else "y"
    col_column = "col_index" if "col_index" in rows[0] else "x"
    if any(column not in rows[0] for column in (frame_column, row_column, col_column)):
        return {int(row["row_index"]): 0.0 for row in rows}
    groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _is_float(row.get(frame_column)) and _is_float(row.get(row_column)) and _is_float(row.get(col_column)):
            groups[float(row[frame_column])].append(row)
    scores = {int(row["row_index"]): 0.0 for row in rows}
    for group_rows in groups.values():
        rr_values = sorted({float(row[row_column]) for row in group_rows})
        cc_values = sorted({float(row[col_column]) for row in group_rows})
        rr_neighbors = _axis_neighbors(rr_values)
        cc_neighbors = _axis_neighbors(cc_values)
        by_position = {(float(row[row_column]), float(row[col_column])): row for row in group_rows}
        for row in group_rows:
            rr = float(row[row_column])
            cc = float(row[col_column])
            local: list[float] = []
            for neighbor_rr in rr_neighbors.get(rr, []):
                neighbor = by_position.get((neighbor_rr, cc))
                if neighbor is not None:
                    distance = abs(neighbor_rr - rr) or 1.0
                    local.append(abs(float(row[target]) - float(neighbor[target])) / distance)
            for neighbor_cc in cc_neighbors.get(cc, []):
                neighbor = by_position.get((rr, neighbor_cc))
                if neighbor is not None:
                    distance = abs(neighbor_cc - cc) or 1.0
                    local.append(abs(float(row[target]) - float(neighbor[target])) / distance)
            if local:
                scores[int(row["row_index"])] = max(local)
    return scores


def _bin_label(value: Any, thresholds: tuple[float, float] | None) -> str:
    if thresholds is None or not _is_float(value):
        return "missing"
    number = float(value)
    if number <= thresholds[0]:
        return "low"
    if number <= thresholds[1]:
        return "mid"
    return "high"


def _tertile_thresholds(rows: list[dict[str, Any]], column: str) -> tuple[float, float] | None:
    values = [float(row[column]) for row in rows if column in row and _is_float(row[column])]
    if not values:
        return None
    return (_quantile(values, 1.0 / 3.0), _quantile(values, 2.0 / 3.0))


def _enrich_rows(rows: list[dict[str, Any]], target: str) -> None:
    time_column = "t" if rows and "t" in rows[0] else "time"
    time_thresholds = _tertile_thresholds(rows, time_column)
    frame_thresholds = _tertile_thresholds(rows, "frame_index")
    gradient_scores = _spatial_gradient_scores(rows, target)
    for row in rows:
        row["process_tuple"] = "__".join(
            [
                f"laser_power_W={row.get('laser_power_W', 'NA')}",
                f"scan_speed_mm_s={row.get('scan_speed_mm_s', 'NA')}",
                f"spot_size_um={row.get('spot_size_um', 'NA')}",
            ]
        )
        row["time_bin"] = _bin_label(row.get(time_column), time_thresholds)
        row["frame_bin"] = _bin_label(row.get("frame_index"), frame_thresholds)
        row["gradient_score"] = gradient_scores.get(int(row["row_index"]), 0.0)


def _metric_payload(rows: list[dict[str, Any]], target: str, prediction_by_index: dict[int, float]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "rmse": None,
            "mae": None,
            "bias": None,
            "hot_q90_rmse": None,
            "gradient_q90_rmse": None,
        }
    truth = [float(row[target]) for row in rows]
    predictions = [float(prediction_by_index[int(row["row_index"])]) for row in rows]
    errors = [pred - y for y, pred in zip(truth, predictions)]
    payload = {
        "n": len(rows),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "bias": sum(errors) / len(errors),
    }
    hot_threshold = _quantile(truth, 0.9)
    gradient_threshold = _quantile([float(row["gradient_score"]) for row in rows], 0.9)
    hot_rows = [row for row in rows if float(row[target]) >= hot_threshold]
    gradient_rows = [row for row in rows if float(row["gradient_score"]) >= gradient_threshold]
    payload["hot_q90_rmse"] = _rmse(hot_rows, target, prediction_by_index)
    payload["gradient_q90_rmse"] = _rmse(gradient_rows, target, prediction_by_index)
    return payload


def _rmse(rows: list[dict[str, Any]], target: str, prediction_by_index: dict[int, float]) -> float | None:
    if not rows:
        return None
    errors = [
        float(prediction_by_index[int(row["row_index"])]) - float(row[target])
        for row in rows
    ]
    return math.sqrt(sum(error * error for error in errors) / len(errors))


def _group_value(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if value in (None, ""):
        return "missing"
    if _is_float(value):
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:.6g}"
    return str(value)


def _candidate_predictions(rows: list[dict[str, Any]], label: str) -> dict[int, float]:
    return {int(row["row_index"]): float(row[f"prediction:{label}"]) for row in rows}


def _fit_global_bias(
    rows: list[dict[str, Any]],
    target: str,
    candidate: str,
) -> float:
    if not rows:
        return 0.0
    residuals = [float(row[target]) - float(row[f"prediction:{candidate}"]) for row in rows]
    return sum(residuals) / len(residuals)


def _variant_global_bias(
    rows: list[dict[str, Any]],
    target: str,
    candidate: str,
    fit_rows: list[dict[str, Any]],
) -> dict[int, float]:
    correction = _fit_global_bias(fit_rows, target, candidate)
    return {
        int(row["row_index"]): float(row[f"prediction:{candidate}"]) + correction
        for row in rows
    }


def _variant_group_bias(
    rows: list[dict[str, Any]],
    target: str,
    candidate: str,
    fit_rows: list[dict[str, Any]],
    field: str,
    min_fit_n: int,
    shrinkage: float,
) -> dict[int, float]:
    global_correction = _fit_global_bias(fit_rows, target, candidate)
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in fit_rows:
        residual = float(row[target]) - float(row[f"prediction:{candidate}"])
        grouped[_group_value(row, field)].append(residual)
    corrections: dict[str, float] = {}
    for group, residuals in grouped.items():
        if len(residuals) < min_fit_n:
            continue
        mean_residual = sum(residuals) / len(residuals)
        weight = len(residuals) / (len(residuals) + shrinkage)
        corrections[group] = weight * mean_residual + (1.0 - weight) * global_correction
    return {
        int(row["row_index"]): float(row[f"prediction:{candidate}"])
        + corrections.get(_group_value(row, field), global_correction)
        for row in rows
    }


def _variant_blend(
    rows: list[dict[str, Any]],
    candidate: str,
    reference: str,
    alpha: float,
) -> dict[int, float]:
    return {
        int(row["row_index"]): (1.0 - alpha) * float(row[f"prediction:{candidate}"])
        + alpha * float(row[f"prediction:{reference}"])
        for row in rows
    }


def _split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("split")) == split]


def _evaluate_variant(
    name: str,
    rows_by_split: dict[str, list[dict[str, Any]]],
    target: str,
    predictions: dict[int, float],
) -> dict[str, Any]:
    return {
        "name": name,
        "metrics": {
            split: _metric_payload(split_rows, target, predictions)
            for split, split_rows in rows_by_split.items()
        },
    }


def collect_summary(args: argparse.Namespace) -> dict[str, Any]:
    labels = list(args.label or [])
    if labels and len(labels) != len(args.prediction):
        raise ValueError("--label must be provided once per --prediction")
    prediction_sets = {}
    for index, path in enumerate(args.prediction):
        label = labels[index] if labels else path.stem
        prediction_sets[label] = _read_prediction_csv(path, args.target)
    for label in (args.candidate, args.reference, args.secondary_reference):
        if label and label not in prediction_sets:
            raise ValueError(f"Unknown method label: {label}")

    rows = _aligned_records(prediction_sets, args.target)
    _enrich_rows(rows, args.target)
    splits = sorted({str(row.get("split", "all")) for row in rows})
    rows_by_split = {split: _split_rows(rows, split) for split in splits}
    fit_rows = rows_by_split.get(args.fit_split, [])
    selection_rows = rows_by_split.get(args.selection_split, [])
    if not fit_rows:
        raise ValueError(f"No rows for fit split: {args.fit_split}")
    if not selection_rows:
        raise ValueError(f"No rows for selection split: {args.selection_split}")

    variants: list[dict[str, Any]] = []
    variants.append(
        _evaluate_variant(
            f"{args.candidate}:identity",
            rows_by_split,
            args.target,
            _candidate_predictions(rows, args.candidate),
        )
    )
    variants.append(
        _evaluate_variant(
            f"{args.candidate}:train_global_bias",
            rows_by_split,
            args.target,
            _variant_global_bias(rows, args.target, args.candidate, fit_rows),
        )
    )
    for field in args.group_field or list(DEFAULT_GROUP_FIELDS):
        variants.append(
            _evaluate_variant(
                f"{args.candidate}:train_group_bias:{field}",
                rows_by_split,
                args.target,
                _variant_group_bias(
                    rows,
                    args.target,
                    args.candidate,
                    fit_rows,
                    field,
                    min_fit_n=args.min_fit_n,
                    shrinkage=args.shrinkage,
                ),
            )
        )
    for alpha in args.blend_alpha or list(DEFAULT_BLEND_ALPHAS):
        variants.append(
            _evaluate_variant(
                f"blend:{args.candidate}->{args.reference}:alpha={alpha:g}",
                rows_by_split,
                args.target,
                _variant_blend(rows, args.candidate, args.reference, float(alpha)),
            )
        )

    variants.sort(
        key=lambda item: (
            item["metrics"][args.selection_split][args.selection_metric]
            if item["metrics"][args.selection_split][args.selection_metric] is not None
            else float("inf")
        )
    )
    selected = variants[0]
    baselines = {
        label: _evaluate_variant(label, rows_by_split, args.target, _candidate_predictions(rows, label))
        for label in prediction_sets
    }
    return {
        "phase": 59,
        "objective": "No-test-leakage residual correction upper-bound probe.",
        "uses_test_for_selection": False,
        "fit_split": args.fit_split,
        "selection_split": args.selection_split,
        "analysis_split": args.analysis_split,
        "selection_metric": args.selection_metric,
        "candidate": args.candidate,
        "reference": args.reference,
        "secondary_reference": args.secondary_reference,
        "prediction_files": {label: payload["path"] for label, payload in prediction_sets.items()},
        "baseline_metrics": {label: item["metrics"] for label, item in baselines.items()},
        "variants": variants,
        "selected_variant": selected,
        "decision": _decision(selected, baselines, args),
    }


def _decision(selected: dict[str, Any], baselines: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    analysis = selected["metrics"][args.analysis_split]
    reference = baselines[args.reference]["metrics"][args.analysis_split]
    candidate = baselines[args.candidate]["metrics"][args.analysis_split]
    selected_rmse = analysis["rmse"]
    reference_rmse = reference["rmse"]
    candidate_rmse = candidate["rmse"]
    selected_beats_reference = (
        selected_rmse is not None and reference_rmse is not None and selected_rmse < reference_rmse
    )
    selected_beats_candidate = (
        selected_rmse is not None and candidate_rmse is not None and selected_rmse < candidate_rmse
    )
    return {
        "selected_variant": selected["name"],
        "analysis_rmse": selected_rmse,
        "reference_rmse": reference_rmse,
        "candidate_rmse": candidate_rmse,
        "selected_beats_reference_rmse": selected_beats_reference,
        "selected_improves_candidate_rmse": selected_beats_candidate,
        "interpretation": (
            "validation-visible correction can beat the reference on the analysis split"
            if selected_beats_reference
            else "validation-visible correction does not beat the reference on the analysis split"
        ),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def markdown_text(summary: dict[str, Any]) -> str:
    selected = summary["selected_variant"]
    lines = [
        "# Phase 59 Residual Upper-Bound Probe",
        "",
        f"Uses test for selection: `{str(summary['uses_test_for_selection']).lower()}`",
        f"Fit split: `{summary['fit_split']}`",
        f"Selection split: `{summary['selection_split']}`",
        f"Analysis split: `{summary['analysis_split']}`",
        f"Selected variant: `{selected['name']}`",
        "",
        "## Selected Variant Metrics",
        "",
        "| split | n | RMSE | hot q90 RMSE | gradient q90 RMSE | MAE | bias |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, row in selected["metrics"].items():
        lines.append(
            "| {split} | {n} | {rmse} | {hot} | {grad} | {mae} | {bias} |".format(
                split=split,
                n=row["n"],
                rmse=_fmt(row["rmse"]),
                hot=_fmt(row["hot_q90_rmse"]),
                grad=_fmt(row["gradient_q90_rmse"]),
                mae=_fmt(row["mae"]),
                bias=_fmt(row["bias"]),
            )
        )
    lines.extend(["", "## Validation-Ranked Variants", ""])
    lines.append("| rank | variant | val RMSE | test RMSE | test hot q90 | test gradient q90 |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for rank, variant in enumerate(summary["variants"][:10], start=1):
        val = variant["metrics"][summary["selection_split"]]
        test = variant["metrics"][summary["analysis_split"]]
        lines.append(
            f"| {rank} | {variant['name']} | {_fmt(val['rmse'])} | {_fmt(test['rmse'])} | {_fmt(test['hot_q90_rmse'])} | {_fmt(test['gradient_q90_rmse'])} |"
        )
    lines.extend(["", "## Decision", "", summary["decision"]["interpretation"], ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, action="append", type=Path)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--target", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--secondary-reference")
    parser.add_argument("--fit-split", default="train")
    parser.add_argument("--selection-split", default="val")
    parser.add_argument("--analysis-split", default="test")
    parser.add_argument("--selection-metric", default="rmse")
    parser.add_argument("--group-field", action="append")
    parser.add_argument("--min-fit-n", type=int, default=20)
    parser.add_argument("--shrinkage", type=float, default=20.0)
    parser.add_argument("--blend-alpha", action="append", type=float)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)

    summary = collect_summary(args)
    markdown = markdown_text(summary)
    print(markdown)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote: {args.json_output}")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")
        print(f"Wrote: {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
