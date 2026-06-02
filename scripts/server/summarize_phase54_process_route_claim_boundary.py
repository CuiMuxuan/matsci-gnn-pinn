#!/usr/bin/env python3
"""Summarize Phase 54 paper-facing claim boundaries for broad process routing.

The input is the JSON produced by
``summarize_phase30_broad_process_selector_smoke.py``.  This script does not
read or overwrite training artifacts directly; it classifies already comparable
broad12/broad21 summaries into paper-facing positives, route-guard positives,
diagnostic negatives, and incomplete-metric cases.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TARGET_METHOD = "broad_process_v1"
STRONG_BASELINE_METHODS = (
    "mean",
    "knn_coords",
    "knn_process",
    "extra_trees_coords",
    "extra_trees_process",
)
NEURAL_REFERENCE_METHODS = ("no_process", "process_axis_v1")
DEFAULT_REQUIRED_METRICS = ("rmse", "hot_q90_rmse", "gradient_q90_rmse")
METRIC_LABELS = {
    "rmse": "test RMSE",
    "hot_q90_rmse": "hot q90 RMSE",
    "gradient_q90_rmse": "gradient q90 RMSE",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _method_value(methods: dict[str, Any], method: str, metric: str) -> float | None:
    row = methods.get(method) or {}
    if row.get("missing") or row.get("comparison_status") != "comparable":
        return None
    return _float_or_none(row.get(metric))


def _strictly_better(candidate: float | None, reference: float | None, tolerance: float) -> bool:
    if candidate is None or reference is None:
        return False
    return candidate < reference - tolerance


def _non_worse(candidate: float | None, reference: float | None, tolerance: float) -> bool:
    if candidate is None or reference is None:
        return False
    return candidate <= reference + tolerance


def _best_method(
    methods: dict[str, Any],
    candidates: tuple[str, ...],
    metric: str,
) -> dict[str, Any] | None:
    values: list[tuple[str, float]] = []
    for method in candidates:
        value = _method_value(methods, method, metric)
        if value is not None:
            values.append((method, value))
    if not values:
        return None
    method, value = min(values, key=lambda item: item[1])
    return {"method": method, "value": value}


def _route_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_conditioning_mode": row.get("selected_conditioning_mode"),
        "selected_feature_normalization": row.get("selected_feature_normalization"),
        "effective_conditioning_mode": row.get("effective_conditioning_mode"),
        "effective_feature_columns": row.get("effective_feature_columns") or [],
    }


def _available_classification(
    metric_rows: dict[str, dict[str, Any]],
    available_metrics: list[str],
    tolerance: float,
) -> str:
    if not available_metrics:
        return "diagnostic_negative"
    strong_pass = all(metric_rows[m]["candidate_beats_best_strong_baseline"] for m in available_metrics)
    no_process_safe = all(metric_rows[m]["candidate_non_worse_than_no_process"] for m in available_metrics)
    if strong_pass and no_process_safe:
        return "paper_claim_positive"
    improves_no_process = any(metric_rows[m]["candidate_beats_no_process"] for m in available_metrics)
    no_process_non_worse = all(metric_rows[m]["candidate_non_worse_than_no_process"] for m in available_metrics)
    old_profile_avoided = any(metric_rows[m]["candidate_beats_process_axis_v1"] for m in available_metrics)
    if (improves_no_process and no_process_non_worse) or (old_profile_avoided and no_process_non_worse):
        return "route_guard_positive"
    return "diagnostic_negative"


def classify_split(
    split_name: str,
    split_payload: dict[str, Any],
    required_metrics: tuple[str, ...] = DEFAULT_REQUIRED_METRICS,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    methods = split_payload.get("methods") or {}
    target = methods.get(TARGET_METHOD) or {}
    result: dict[str, Any] = {
        "split": split_name,
        "target_method": TARGET_METHOD,
        "all_methods_comparable": bool(split_payload.get("all_methods_comparable")),
        "route": _route_summary(target),
        "required_metrics": list(required_metrics),
        "available_metrics": [],
        "missing_metrics": [],
        "metrics": {},
        "notes": [],
    }
    if not result["all_methods_comparable"]:
        result["classification"] = "incomparable"
        result["notes"].append("Manifest/split comparability failed in the input summary.")
        return result
    if target.get("missing") or target.get("comparison_status") != "comparable":
        result["classification"] = "incomparable"
        result["notes"].append(f"{TARGET_METHOD} is missing or not comparable.")
        return result

    for metric in required_metrics:
        candidate = _method_value(methods, TARGET_METHOD, metric)
        no_process = _method_value(methods, "no_process", metric)
        process_axis = _method_value(methods, "process_axis_v1", metric)
        best_strong = _best_method(methods, STRONG_BASELINE_METHODS, metric)
        best_neural = _best_method(methods, (TARGET_METHOD, *NEURAL_REFERENCE_METHODS), metric)
        available = candidate is not None and no_process is not None and best_strong is not None
        if available:
            result["available_metrics"].append(metric)
        else:
            result["missing_metrics"].append(metric)
        best_strong_value = None if best_strong is None else best_strong["value"]
        metric_row = {
            "candidate": candidate,
            "best_strong_baseline": best_strong,
            "no_process": no_process,
            "process_axis_v1": process_axis,
            "best_neural_reference": best_neural,
            "candidate_delta_vs_best_strong": (
                None if candidate is None or best_strong_value is None else candidate - best_strong_value
            ),
            "candidate_delta_vs_no_process": (
                None if candidate is None or no_process is None else candidate - no_process
            ),
            "candidate_delta_vs_process_axis_v1": (
                None if candidate is None or process_axis is None else candidate - process_axis
            ),
            "candidate_beats_best_strong_baseline": _strictly_better(
                candidate, best_strong_value, tolerance
            ),
            "candidate_non_worse_than_no_process": _non_worse(candidate, no_process, tolerance),
            "candidate_beats_no_process": _strictly_better(candidate, no_process, tolerance),
            "candidate_beats_process_axis_v1": _strictly_better(candidate, process_axis, tolerance),
        }
        result["metrics"][metric] = metric_row

    available_metrics = result["available_metrics"]
    available_class = _available_classification(result["metrics"], available_metrics, tolerance)
    result["available_metric_classification"] = available_class
    if result["missing_metrics"]:
        result["classification"] = "incomplete_metric"
        result["notes"].append(
            "At least one required metric is missing; do not make a full three-metric paper claim."
        )
    else:
        result["classification"] = available_class

    if result["classification"] == "paper_claim_positive":
        result["notes"].append(
            f"{TARGET_METHOD} beats the best strong baseline and is non-worse than no-process on all required metrics."
        )
    elif result["classification"] == "route_guard_positive":
        result["notes"].append(
            f"{TARGET_METHOD} supports route-guard evidence but does not beat the best strong baseline on all required metrics."
        )
    elif result["classification"] == "diagnostic_negative":
        result["notes"].append(
            f"{TARGET_METHOD} is weaker than the strong-baseline/no-process boundary for this split."
        )

    neural_gaps = []
    for metric in available_metrics:
        best_neural = result["metrics"][metric].get("best_neural_reference")
        if best_neural and best_neural.get("method") != TARGET_METHOD:
            neural_gaps.append(f"{metric}:{best_neural['method']}")
    if neural_gaps:
        result["notes"].append(
            "A neural reference is better on at least one metric: " + ", ".join(neural_gaps) + "."
        )
    if target.get("selected_conditioning_mode") == "none":
        result["notes"].append("The selected route is the no-process fallback; do not claim process conditioning improves this split.")
    return result


def _dataset_label(input_path: Path, payload: dict[str, Any]) -> str:
    limit = payload.get("dataset_limit")
    if limit is not None:
        return f"broad{limit}"
    stem = input_path.stem
    if "broad12" in stem:
        return "broad12"
    if "broad21" in stem:
        return "broad21"
    return stem


def summarize_input(
    input_path: Path,
    required_metrics: tuple[str, ...] = DEFAULT_REQUIRED_METRICS,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    payload = _read_json(input_path)
    dataset = {
        "label": _dataset_label(input_path, payload),
        "input_path": str(input_path),
        "dataset_limit": payload.get("dataset_limit"),
        "dataset_order": payload.get("dataset_order"),
        "splits": {},
        "counts": {},
    }
    for split_name, split_payload in (payload.get("splits") or {}).items():
        row = classify_split(split_name, split_payload, required_metrics, tolerance)
        dataset["splits"][split_name] = row
        classification = row["classification"]
        dataset["counts"][classification] = dataset["counts"].get(classification, 0) + 1
    return dataset


def collect_summary(
    input_paths: list[Path],
    required_metrics: tuple[str, ...] = DEFAULT_REQUIRED_METRICS,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    datasets = [summarize_input(path, required_metrics, tolerance) for path in input_paths]
    overall_counts: dict[str, int] = {}
    lists: dict[str, list[str]] = {
        "paper_claim_positive": [],
        "route_guard_positive": [],
        "diagnostic_negative": [],
        "incomplete_metric": [],
        "incomparable": [],
    }
    for dataset in datasets:
        for split, row in dataset["splits"].items():
            classification = row["classification"]
            overall_counts[classification] = overall_counts.get(classification, 0) + 1
            lists.setdefault(classification, []).append(f"{dataset['label']}:{split}")
    return {
        "phase": 54,
        "target_method": TARGET_METHOD,
        "strong_baseline_methods": list(STRONG_BASELINE_METHODS),
        "neural_reference_methods": list(NEURAL_REFERENCE_METHODS),
        "required_metrics": list(required_metrics),
        "metric_labels": METRIC_LABELS,
        "classification_counts": overall_counts,
        "classification_lists": lists,
        "datasets": datasets,
        "claim_boundary": {
            "paper_claim_positive": lists.get("paper_claim_positive", []),
            "route_guard_positive": lists.get("route_guard_positive", []),
            "incomplete_metric": lists.get("incomplete_metric", []),
            "diagnostic_negative": lists.get("diagnostic_negative", []),
            "incomparable": lists.get("incomparable", []),
            "seed_expansion_candidates": [
                item for item in lists.get("paper_claim_positive", []) if not item.endswith(":line")
            ],
        },
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _route_text(route: dict[str, Any]) -> str:
    selected = route.get("selected_conditioning_mode") or ""
    norm = route.get("selected_feature_normalization") or ""
    if selected and norm:
        return f"{selected}/{norm}"
    return selected or ""


def _metric_cell(metric_row: dict[str, Any]) -> str:
    candidate = metric_row.get("candidate")
    best = metric_row.get("best_strong_baseline") or {}
    delta = metric_row.get("candidate_delta_vs_best_strong")
    if candidate is None:
        return ""
    best_text = ""
    if best:
        best_text = f" vs {best.get('method')} {_fmt(best.get('value'))}"
    delta_text = "" if delta is None else f" d={_fmt(delta)}"
    return f"{_fmt(candidate)}{best_text}{delta_text}"


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Phase 54 Broad Process Route Claim Boundary")
    lines.append("")
    lines.append("## Classification Counts")
    lines.append("")
    lines.append("| classification | count | splits |")
    lines.append("|---|---:|---|")
    for key in ("paper_claim_positive", "route_guard_positive", "incomplete_metric", "diagnostic_negative", "incomparable"):
        items = summary["classification_lists"].get(key, [])
        lines.append(f"| {key} | {len(items)} | {', '.join(items)} |")
    lines.append("")
    lines.append("## Split Table")
    lines.append("")
    lines.append(
        "| dataset | split | class | route | test RMSE | hot q90 RMSE | gradient q90 RMSE | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for dataset in summary["datasets"]:
        for split, row in dataset["splits"].items():
            metrics = row.get("metrics") or {}
            notes = " ".join(row.get("notes") or [])
            lines.append(
                "| {dataset} | {split} | {klass} | {route} | {rmse} | {hot} | {grad} | {notes} |".format(
                    dataset=dataset["label"],
                    split=split,
                    klass=row.get("classification"),
                    route=_route_text(row.get("route") or {}),
                    rmse=_metric_cell(metrics.get("rmse") or {}),
                    hot=_metric_cell(metrics.get("hot_q90_rmse") or {}),
                    grad=_metric_cell(metrics.get("gradient_q90_rmse") or {}),
                    notes=notes,
                )
            )
    lines.append("")
    lines.append("## Claim Boundary")
    lines.append("")
    lines.append(
        "- `paper_claim_positive`: full required metrics beat the best strong baseline and are non-worse than no-process."
    )
    lines.append(
        "- `route_guard_positive`: useful negative-transfer guard evidence, but not a strong-baseline paper claim."
    )
    lines.append("- `incomplete_metric`: at least one required metric is missing; rerun metric extraction before a full claim.")
    lines.append("- Source-inversion / registered source-path branches remain diagnostic unless aligned scan-path data appears.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Phase 30-style broad process summary JSON. Repeat for broad12 and broad21.",
    )
    parser.add_argument("--json-output", help="Optional path to write the Phase 54 JSON summary.")
    parser.add_argument("--markdown-output", help="Optional path to write a Markdown summary table.")
    parser.add_argument(
        "--required-metric",
        action="append",
        choices=tuple(DEFAULT_REQUIRED_METRICS),
        help="Required metric for full paper-claim classification. Defaults to all three metrics.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Numerical tolerance for lower-is-better comparisons.",
    )
    parser.add_argument(
        "--require-comparable",
        action="store_true",
        help="Exit non-zero if any split is incomparable in the claim-boundary summary.",
    )
    args = parser.parse_args()

    required_metrics = tuple(args.required_metric or DEFAULT_REQUIRED_METRICS)
    input_paths = [Path(path) for path in args.input]
    summary = collect_summary(input_paths, required_metrics, args.tolerance)
    markdown = render_markdown(summary)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown_output:
        _write_text(Path(args.markdown_output), markdown)
    print(markdown)
    if args.require_comparable and summary["classification_lists"].get("incomparable"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
