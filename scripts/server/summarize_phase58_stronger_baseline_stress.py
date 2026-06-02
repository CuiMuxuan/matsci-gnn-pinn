#!/usr/bin/env python3
"""Summarize Phase 58 stronger-baseline stress tests.

The stress gate asks a narrow question: after adding low-risk stronger tabular
baselines, does the frozen Phase 55 `spot_size` claim still beat the strongest
baseline on broad12 and broad21 for RMSE, hot q90 RMSE, and gradient q90 RMSE?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = ("rmse", "hot_q90_rmse", "gradient_q90_rmse")
DEFAULT_STRESS_METHODS = (
    "random_forest:coords",
    "random_forest:process",
    "hist_gradient_boosting:coords",
    "hist_gradient_boosting:process",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


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


def _baseline_path(root: Path, base_run_id: str, method: str, tag: str) -> Path:
    return root / "outputs" / "baselines" / f"{base_run_id}_{method}_{tag}_regions_q90.json"


def _baseline_row(root: Path, base_run_id: str, method_spec: str) -> dict[str, Any]:
    method, tag = method_spec.split(":", 1)
    path = _baseline_path(root, base_run_id, method, tag)
    row: dict[str, Any] = {
        "method": method,
        "tag": tag,
        "name": f"{method}_{tag}",
        "path": str(path),
    }
    if not path.exists():
        row["missing"] = True
        return row
    row.update(_extract_metric_fields(_read_json(path)))
    row["missing"] = any(row.get(metric) is None for metric in METRIC_KEYS)
    return row


def _best(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    candidates = [
        {"method": row["name"], "value": row.get(metric), "path": row.get("path")}
        for row in rows
        if not row.get("missing") and row.get(metric) is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: float(item["value"]))


def _phase55_floor(dataset: dict[str, Any], metric: str) -> float | None:
    value = (
        ((dataset.get("aggregates") or {}).get("broad_process_v1") or {})
        .get(metric, {})
        .get("mean")
    )
    return float(value) if value is not None else None


def _phase55_best_strong(dataset: dict[str, Any], metric: str) -> dict[str, Any] | None:
    row = ((dataset.get("aggregate_gate") or {}).get("metrics") or {}).get(metric) or {}
    value = row.get("best_strong_baseline")
    if value is None:
        return None
    return {
        "method": row.get("best_strong_baseline_method"),
        "value": float(value),
    }


def collect_summary(
    root: Path,
    phase55_summary: Path,
    stress_methods: tuple[str, ...] = DEFAULT_STRESS_METHODS,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    phase55 = _read_json(phase55_summary)
    datasets = []
    for dataset in phase55.get("datasets", []):
        base_run_id = dataset["base_run_id"]
        rows = [_baseline_row(root, base_run_id, method) for method in stress_methods]
        missing = [row["name"] for row in rows if row.get("missing")]
        metric_rows: dict[str, Any] = {}
        for metric in METRIC_KEYS:
            frozen = _phase55_floor(dataset, metric)
            prior = _phase55_best_strong(dataset, metric)
            stress = _best(rows, metric)
            prior_value = prior.get("value") if prior else None
            stress_value = stress.get("value") if stress else None
            strongest_value = None
            strongest_method = None
            if prior_value is not None:
                strongest_value = prior_value
                strongest_method = prior.get("method")
            if stress_value is not None and (
                strongest_value is None or stress_value < strongest_value
            ):
                strongest_value = stress_value
                strongest_method = stress.get("method")
            passes = (
                frozen is not None
                and strongest_value is not None
                and frozen < strongest_value - tolerance
            )
            metric_rows[metric] = {
                "frozen_broad_process_v1": frozen,
                "prior_best_strong": prior,
                "best_stress_baseline": stress,
                "best_baseline_after_stress": {
                    "method": strongest_method,
                    "value": strongest_value,
                },
                "delta_vs_best_after_stress": None
                if frozen is None or strongest_value is None
                else frozen - strongest_value,
                "frozen_beats_best_after_stress": passes,
            }
        dataset_pass = not missing and all(
            row["frozen_beats_best_after_stress"] for row in metric_rows.values()
        )
        datasets.append(
            {
                "label": dataset["label"],
                "base_run_id": base_run_id,
                "missing_stress_baselines": missing,
                "stress_rows": rows,
                "metrics": metric_rows,
                "pass": dataset_pass,
            }
        )
    overall_pass = all(dataset["pass"] for dataset in datasets)
    return {
        "phase": 58,
        "objective": "stronger_baseline_stress",
        "phase55_summary": str(phase55_summary),
        "stress_methods": list(stress_methods),
        "required_metrics": list(METRIC_KEYS),
        "datasets": datasets,
        "stress_gate": {
            "pass": overall_pass,
            "status": "claim_survives_stronger_baselines"
            if overall_pass
            else "claim_challenged_by_stronger_baseline",
        },
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 58 Stronger-Baseline Stress Summary",
        "",
        f"Stress gate: `{summary['stress_gate']['status']}`",
        "",
        "| dataset | metric | frozen broad_process_v1 | best baseline after stress | delta | pass |",
        "|---|---|---:|---:|---:|---|",
    ]
    for dataset in summary["datasets"]:
        for metric, row in dataset["metrics"].items():
            best = row["best_baseline_after_stress"]
            lines.append(
                "| {dataset} | {metric} | {frozen} | {best_value} ({best_method}) | {delta} | {passed} |".format(
                    dataset=dataset["label"],
                    metric=metric,
                    frozen=_fmt(row["frozen_broad_process_v1"]),
                    best_value=_fmt(best.get("value")),
                    best_method=best.get("method") or "",
                    delta=_fmt(row["delta_vs_best_after_stress"]),
                    passed=_fmt(row["frozen_beats_best_after_stress"]),
                )
            )
    lines.extend(["", "## Missing Stress Baselines", ""])
    any_missing = False
    for dataset in summary["datasets"]:
        if dataset["missing_stress_baselines"]:
            any_missing = True
            lines.append(
                f"- {dataset['label']}: {', '.join(dataset['missing_stress_baselines'])}"
            )
    if not any_missing:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--phase55-summary",
        type=Path,
        default=Path("outputs/reports/phase55_spot_size_route_seed_check_summary.json"),
    )
    parser.add_argument("--stress-method", action="append", dest="stress_methods")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    root = args.root
    methods = tuple(args.stress_methods or DEFAULT_STRESS_METHODS)
    summary = collect_summary(root, args.phase55_summary, methods)
    markdown = render_markdown(summary)
    print(markdown)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")
    if args.require_complete and any(
        dataset["missing_stress_baselines"] for dataset in summary["datasets"]
    ):
        return 3
    if args.require_pass and not summary["stress_gate"]["pass"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
