#!/usr/bin/env python3
"""Summarize row-level residual anatomy from prediction CSV exports."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


RESERVED = {
    "row_index",
    "split",
    "sample_id",
    "method",
    "prediction",
    "error",
    "abs_error",
    "x",
    "y",
    "z",
    "t",
    "time",
}
METRICS = ("rmse", "mae", "bias")
DEFAULT_GROUP_FIELDS = (
    "laser_power_W",
    "scan_speed_mm_s",
    "spot_size_um",
    "process_tuple",
    "line_id",
    "time_bin",
    "frame_bin",
    "hot_region",
    "gradient_region",
    "region",
)


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
    predictions: dict[int, float] = {}
    records: dict[int, dict[str, Any]] = {}
    for row in rows:
        row_index = int(row["row_index"])
        predictions[row_index] = float(row["prediction"])
        if target not in row:
            raise ValueError(f"Target column {target!r} missing in {path}")
        records[row_index] = row
    method = rows[0].get("method") or path.stem
    return {"path": str(path), "method": method, "predictions": predictions, "records": records}


def _metric_payload(truth: list[float], pred: list[float]) -> dict[str, float | int | None]:
    n = len(truth)
    if not truth:
        return {"n": 0, "rmse": None, "mae": None, "bias": None}
    errors = [float(p) - float(t) for t, p in zip(truth, pred)]
    return {
        "n": n,
        "rmse": math.sqrt(sum(error * error for error in errors) / n),
        "mae": sum(abs(error) for error in errors) / n,
        "bias": sum(errors) / n,
    }


def _quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute quantile of an empty list")
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    value = ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    return min(max(value, ordered[0]), ordered[-1])


def _bin_label(value: Any, thresholds: tuple[float, float] | None) -> str:
    if thresholds is None or not _is_float(value):
        return "missing"
    number = float(value)
    low, high = thresholds
    if number <= low:
        return "low"
    if number <= high:
        return "mid"
    return "high"


def _tertile_thresholds(records: Iterable[dict[str, Any]], column: str) -> tuple[float, float] | None:
    values = [float(row[column]) for row in records if column in row and _is_float(row[column])]
    if not values:
        return None
    return (_quantile(values, 1.0 / 3.0), _quantile(values, 2.0 / 3.0))


def _spatial_gradient_scores(records: list[dict[str, Any]], target: str) -> dict[int, float]:
    if not records:
        return {}
    frame_column = "frame_index" if "frame_index" in records[0] else "t" if "t" in records[0] else "time"
    row_column = "metadata_row_index" if "metadata_row_index" in records[0] else "y"
    col_column = "col_index" if "col_index" in records[0] else "x"
    if any(column not in records[0] for column in (frame_column, row_column, col_column)):
        return {int(row["row_index"]): 0.0 for row in records}

    groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if _is_float(row.get(frame_column)) and _is_float(row.get(row_column)) and _is_float(row.get(col_column)):
            groups[float(row[frame_column])].append(row)

    scores = {int(row["row_index"]): 0.0 for row in records}
    for group in groups.values():
        rows = sorted({float(row[row_column]) for row in group})
        cols = sorted({float(row[col_column]) for row in group})
        row_neighbors = _axis_neighbors(rows)
        col_neighbors = _axis_neighbors(cols)
        by_position = {(float(row[row_column]), float(row[col_column])): row for row in group}
        for row in group:
            row_index = int(row["row_index"])
            rr = float(row[row_column])
            cc = float(row[col_column])
            local: list[float] = []
            for neighbor_row in row_neighbors.get(rr, []):
                neighbor = by_position.get((neighbor_row, cc))
                if neighbor is not None:
                    distance = abs(neighbor_row - rr) or 1.0
                    local.append(abs(float(row[target]) - float(neighbor[target])) / distance)
            for neighbor_col in col_neighbors.get(cc, []):
                neighbor = by_position.get((rr, neighbor_col))
                if neighbor is not None:
                    distance = abs(neighbor_col - cc) or 1.0
                    local.append(abs(float(row[target]) - float(neighbor[target])) / distance)
            if local:
                scores[row_index] = max(local)
    return scores


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


def _enrich_records(records: list[dict[str, Any]], target: str, hot_quantile: float, gradient_quantile: float) -> None:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gradient_scores = _spatial_gradient_scores(records, target)
    for row in records:
        by_split[str(row.get("split", "all"))].append(row)
        row["process_tuple"] = "__".join(
            [
                f"laser_power_W={row.get('laser_power_W', 'NA')}",
                f"scan_speed_mm_s={row.get('scan_speed_mm_s', 'NA')}",
                f"spot_size_um={row.get('spot_size_um', 'NA')}",
            ]
        )
        row["gradient_score"] = gradient_scores.get(int(row["row_index"]), 0.0)

    time_thresholds = _tertile_thresholds(records, "t" if "t" in records[0] else "time")
    frame_thresholds = _tertile_thresholds(records, "frame_index")
    for row in records:
        row["time_bin"] = _bin_label(row.get("t", row.get("time")), time_thresholds)
        row["frame_bin"] = _bin_label(row.get("frame_index"), frame_thresholds)

    for split_rows in by_split.values():
        target_values = [float(row[target]) for row in split_rows]
        gradient_values = [float(row["gradient_score"]) for row in split_rows]
        hot_threshold = _quantile(target_values, hot_quantile)
        gradient_threshold = _quantile(gradient_values, gradient_quantile)
        for row in split_rows:
            regions = []
            if float(row[target]) >= hot_threshold:
                regions.append("hot_q90")
                row["hot_region"] = "hot_q90"
            else:
                row["hot_region"] = "not_hot_q90"
            if float(row["gradient_score"]) >= gradient_threshold:
                regions.append("gradient_q90")
                row["gradient_region"] = "gradient_q90"
            else:
                row["gradient_region"] = "not_gradient_q90"
            row["region"] = "+".join(regions) if regions else "background"


def _aligned_records(prediction_sets: dict[str, dict[str, Any]], target: str) -> list[dict[str, Any]]:
    labels = list(prediction_sets)
    first = prediction_sets[labels[0]]
    row_indices = sorted(first["records"])
    for label in labels[1:]:
        missing = set(row_indices).symmetric_difference(prediction_sets[label]["records"])
        if missing:
            raise ValueError(f"Prediction set {label!r} is not row-aligned; mismatch count={len(missing)}")
    records: list[dict[str, Any]] = []
    for row_index in row_indices:
        base = dict(first["records"][row_index])
        base[target] = float(base[target])
        base["row_index"] = int(base["row_index"])
        for label in labels:
            prediction = prediction_sets[label]["predictions"][row_index]
            base[f"prediction:{label}"] = prediction
            base[f"error:{label}"] = prediction - float(base[target])
            base[f"abs_error:{label}"] = abs(prediction - float(base[target]))
        records.append(base)
    return records


def _metrics_for(records: list[dict[str, Any]], target: str, methods: list[str]) -> dict[str, dict[str, Any]]:
    truth = [float(row[target]) for row in records]
    return {
        method: _metric_payload(truth, [float(row[f"prediction:{method}"]) for row in records])
        for method in methods
    }


def _group_value(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if value in (None, ""):
        return "missing"
    if _is_float(value):
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:.6g}"
    return str(value)


def _group_summary(
    records: list[dict[str, Any]],
    *,
    field: str,
    target: str,
    methods: list[str],
    candidate: str,
    reference: str,
    secondary_reference: str | None,
    min_group_n: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[_group_value(row, field)].append(row)
    rows: list[dict[str, Any]] = []
    for value, group_rows in grouped.items():
        if len(group_rows) < min_group_n:
            continue
        metrics = _metrics_for(group_rows, target, methods)
        candidate_rmse = metrics[candidate]["rmse"]
        reference_rmse = metrics[reference]["rmse"]
        secondary_rmse = metrics[secondary_reference]["rmse"] if secondary_reference else None
        rows.append(
            {
                "field": field,
                "value": value,
                "n": len(group_rows),
                "metrics": metrics,
                "delta_candidate_minus_reference_rmse": (
                    float(candidate_rmse) - float(reference_rmse)
                    if candidate_rmse is not None and reference_rmse is not None
                    else None
                ),
                "delta_candidate_minus_secondary_rmse": (
                    float(candidate_rmse) - float(secondary_rmse)
                    if candidate_rmse is not None and secondary_rmse is not None
                    else None
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            row["delta_candidate_minus_reference_rmse"]
            if row["delta_candidate_minus_reference_rmse"] is not None
            else float("-inf")
        ),
        reverse=True,
    )
    return rows


def collect_summary(args: argparse.Namespace) -> dict[str, Any]:
    labels = list(args.label or [])
    if labels and len(labels) != len(args.prediction):
        raise ValueError("--label must be provided once per --prediction")
    prediction_sets = {}
    for index, path in enumerate(args.prediction):
        payload = _read_prediction_csv(path, args.target)
        label = labels[index] if labels else str(payload["method"])
        prediction_sets[label] = payload
    methods = list(prediction_sets)
    for required in [args.candidate, args.reference, args.secondary_reference]:
        if required and required not in prediction_sets:
            raise ValueError(f"Unknown method label: {required}")

    records = _aligned_records(prediction_sets, args.target)
    _enrich_records(records, args.target, args.hot_quantile, args.gradient_quantile)
    analysis_records = [row for row in records if row.get("split") == args.analysis_split]
    if not analysis_records:
        raise ValueError(f"No rows found for analysis split: {args.analysis_split}")

    split_summary = {}
    for split in sorted({str(row.get("split", "all")) for row in records}):
        split_rows = [row for row in records if row.get("split") == split]
        split_summary[split] = _metrics_for(split_rows, args.target, methods)

    group_fields = args.group_field or list(DEFAULT_GROUP_FIELDS)
    group_summaries = {
        field: _group_summary(
            analysis_records,
            field=field,
            target=args.target,
            methods=methods,
            candidate=args.candidate,
            reference=args.reference,
            secondary_reference=args.secondary_reference,
            min_group_n=args.min_group_n,
        )
        for field in group_fields
    }
    worst = sorted(
        [row for rows in group_summaries.values() for row in rows],
        key=lambda row: (
            row["delta_candidate_minus_reference_rmse"]
            if row["delta_candidate_minus_reference_rmse"] is not None
            else float("-inf")
        ),
        reverse=True,
    )[: args.top_n]
    return {
        "phase": 59,
        "objective": "Residual anatomy of route-guard failures before model expansion.",
        "prediction_files": {
            label: payload["path"] for label, payload in prediction_sets.items()
        },
        "methods": methods,
        "candidate": args.candidate,
        "reference": args.reference,
        "secondary_reference": args.secondary_reference,
        "target": args.target,
        "analysis_split": args.analysis_split,
        "n_records": len(records),
        "n_analysis_records": len(analysis_records),
        "split_summary": split_summary,
        "group_summaries": group_summaries,
        "worst_candidate_vs_reference": worst,
        "decision": _decision(split_summary, args.analysis_split, args.candidate, args.reference, args.secondary_reference),
    }


def _decision(
    split_summary: dict[str, Any],
    analysis_split: str,
    candidate: str,
    reference: str,
    secondary_reference: str | None,
) -> dict[str, Any]:
    metrics = split_summary[analysis_split]
    candidate_rmse = metrics[candidate]["rmse"]
    reference_rmse = metrics[reference]["rmse"]
    secondary_rmse = metrics[secondary_reference]["rmse"] if secondary_reference else None
    return {
        "candidate_beats_reference_rmse": candidate_rmse is not None and reference_rmse is not None and candidate_rmse < reference_rmse,
        "candidate_beats_secondary_rmse": (
            candidate_rmse is not None and secondary_rmse is not None and candidate_rmse < secondary_rmse
            if secondary_reference
            else None
        ),
        "interpretation": (
            "candidate loses to reference on the analysis split; inspect worst groups before model expansion"
            if candidate_rmse is not None and reference_rmse is not None and candidate_rmse >= reference_rmse
            else "candidate beats the reference on the analysis split; residual anatomy should focus on remaining local failures"
        ),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def markdown_text(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 59 Residual Anatomy",
        "",
        f"Analysis split: `{summary['analysis_split']}`",
        f"Candidate: `{summary['candidate']}`",
        f"Reference: `{summary['reference']}`",
        f"Secondary reference: `{summary.get('secondary_reference') or ''}`",
        "",
        "## Split Summary",
        "",
        "| split | method | n | RMSE | MAE | bias |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split, metrics in summary["split_summary"].items():
        for method, row in metrics.items():
            lines.append(
                f"| {split} | {method} | {row['n']} | {_fmt(row['rmse'])} | {_fmt(row['mae'])} | {_fmt(row['bias'])} |"
            )
    lines.extend(["", "## Worst Candidate-vs-Reference Groups", ""])
    lines.append("| field | value | n | candidate RMSE | reference RMSE | delta | secondary delta |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    candidate = summary["candidate"]
    reference = summary["reference"]
    secondary = summary.get("secondary_reference")
    for row in summary["worst_candidate_vs_reference"]:
        metrics = row["metrics"]
        lines.append(
            "| {field} | {value} | {n} | {cand} | {ref} | {delta} | {secondary_delta} |".format(
                field=row["field"],
                value=row["value"],
                n=row["n"],
                cand=_fmt(metrics[candidate]["rmse"]),
                ref=_fmt(metrics[reference]["rmse"]),
                delta=_fmt(row["delta_candidate_minus_reference_rmse"]),
                secondary_delta=_fmt(row["delta_candidate_minus_secondary_rmse"]) if secondary else "",
            )
        )
    lines.extend(["", "## Decision", "", summary["decision"]["interpretation"], ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True, type=Path)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--target", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--secondary-reference")
    parser.add_argument("--analysis-split", default="test")
    parser.add_argument("--group-field", action="append")
    parser.add_argument("--min-group-n", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--hot-quantile", type=float, default=0.9)
    parser.add_argument("--gradient-quantile", type=float, default=0.9)
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
