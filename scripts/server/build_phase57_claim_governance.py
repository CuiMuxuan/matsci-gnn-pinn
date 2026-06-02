#!/usr/bin/env python3
"""Build the Phase 57 claim-governance ledger and gate contract.

This script consumes already summarized Phase 54/55/56 artifacts. It does not
inspect or mutate training outputs. The intent is to freeze the paper-facing
floor before any new model branch is attempted.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_DATASETS = ("broad12", "broad21")
REQUIRED_METRICS = ("rmse", "hot_q90_rmse", "gradient_q90_rmse")
METRIC_LABELS = {
    "rmse": "test RMSE",
    "hot_q90_rmse": "hot q90 RMSE",
    "gradient_q90_rmse": "gradient q90 RMSE",
}
COMPARISON_METHODS = (
    "mean",
    "knn_coords",
    "knn_process",
    "extra_trees_coords",
    "extra_trees_process",
    "no_process",
    "process_axis_v1",
    "broad_process_v1",
)
LEDGER_FIELDS = (
    "kind",
    "dataset",
    "split_or_branch",
    "status",
    "process_conditioning_claim",
    "route",
    "metrics_gate",
    "seed_gate",
    "paper_placement",
    "current_evidence",
    "next_action",
    "notes",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in LEDGER_FIELDS})


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _route_label(route: dict[str, Any]) -> str:
    selected = route.get("selected_conditioning_mode") or ""
    norm = route.get("selected_feature_normalization") or ""
    if selected and norm:
        return f"{selected}/{norm}"
    return selected or norm or ""


def _phase55_by_dataset(phase55: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(dataset.get("label")): dataset
        for dataset in phase55.get("datasets", [])
        if isinstance(dataset, dict) and dataset.get("label")
    }


def extract_frozen_floor(phase55: dict[str, Any]) -> dict[str, Any]:
    """Extract the current paper-facing floor from Phase 55 seed aggregates."""
    floor: dict[str, Any] = {
        "method": "broad_process_v1",
        "split": phase55.get("split") or "spot_size",
        "required_datasets": list(REQUIRED_DATASETS),
        "seeds": phase55.get("seeds") or [7, 1, 2],
        "required_metrics": list(REQUIRED_METRICS),
        "datasets": {},
    }
    for dataset in phase55.get("datasets", []):
        label = dataset.get("label")
        if label not in REQUIRED_DATASETS:
            continue
        aggregates = dataset.get("aggregates") or {}
        candidate = aggregates.get("broad_process_v1") or {}
        route = dataset.get("route") or {}
        metrics: dict[str, Any] = {}
        for metric in REQUIRED_METRICS:
            values = candidate.get(metric) or {}
            metrics[metric] = {
                "mean": values.get("mean"),
                "pstdev": values.get("pstdev"),
            }
        floor["datasets"][label] = {
            "status": dataset.get("status"),
            "route": route,
            "metrics": metrics,
        }
    return floor


def evaluate_candidate_against_floor(
    candidate_metrics: dict[str, dict[str, float | int | None]],
    floor_metrics: dict[str, dict[str, float | int | None]],
    required_datasets: tuple[str, ...] = REQUIRED_DATASETS,
    required_metrics: tuple[str, ...] = REQUIRED_METRICS,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Evaluate whether a candidate may proceed to A100 seed expansion.

    Lower metrics are better. A candidate passes this narrow seed-expansion gate
    only when every required metric is non-worse than the frozen floor on every
    required dataset. This helper intentionally does not replace strong-baseline
    comparison; it is the route-guard preservation gate.
    """
    checks: list[dict[str, Any]] = []
    passed = True
    for dataset in required_datasets:
        candidate_dataset = candidate_metrics.get(dataset) or {}
        floor_dataset = floor_metrics.get(dataset) or {}
        for metric in required_metrics:
            candidate = candidate_dataset.get(metric)
            floor = floor_dataset.get(metric)
            metric_pass = (
                candidate is not None
                and floor is not None
                and float(candidate) <= float(floor) + tolerance
            )
            if not metric_pass:
                passed = False
            checks.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "candidate": candidate,
                    "frozen_floor": floor,
                    "delta_vs_floor": None
                    if candidate is None or floor is None
                    else float(candidate) - float(floor),
                    "pass": metric_pass,
                }
            )
    return {
        "pass": passed,
        "decision": "allow_seed_expansion" if passed else "block_seed_expansion",
        "checks": checks,
    }


def _floor_metric_means(floor: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = {}
    for dataset, payload in (floor.get("datasets") or {}).items():
        output[dataset] = {
            metric: ((payload.get("metrics") or {}).get(metric) or {}).get("mean")
            for metric in REQUIRED_METRICS
        }
    return output


def build_contract(phase54: dict[str, Any], phase55: dict[str, Any]) -> dict[str, Any]:
    floor = extract_frozen_floor(phase55)
    missing_floor = [
        dataset for dataset in REQUIRED_DATASETS if dataset not in (floor.get("datasets") or {})
    ]
    return {
        "phase": 57,
        "objective": "Freeze claim boundaries and future branch gates after Phase 56.",
        "frozen_floor": floor,
        "missing_floor_datasets": missing_floor,
        "required_metrics": list(REQUIRED_METRICS),
        "metric_labels": METRIC_LABELS,
        "comparison_methods_required_for_future_branches": list(COMPARISON_METHODS),
        "no_test_leakage_rules": [
            "Route selection, hyperparameters, feature-family selection, and seed expansion must be decided from train/validation evidence only.",
            "Test metrics may be used only once for reporting the predeclared candidate.",
            "New candidates must preserve manifest/split comparability with the route guard before comparison.",
            "Focused seed-7 gates may only justify seeds 1/2; they do not justify manuscript claims.",
        ],
        "future_branch_seed_expansion_gate": {
            "decision": "block unless all required checks pass",
            "route_guard_method": "broad_process_v1",
            "required_datasets": list(REQUIRED_DATASETS),
            "required_metrics": list(REQUIRED_METRICS),
            "rule": "candidate <= frozen broad_process_v1 floor on every required dataset/metric",
        },
        "future_manuscript_claim_gate": {
            "requires_seed_aggregate": "seeds 7/1/2 at minimum",
            "requires_strong_baseline_comparison": True,
            "requires_route_guard_preservation": True,
            "allowed_main_claim_status": "paper_positive_seed_robust",
        },
        "current_claim_boundary_counts": phase54.get("classification_counts") or {},
        "current_transfer_gate": phase55.get("transfer_gate") or {},
    }


def _split_ledger_row(
    dataset_label: str,
    split_name: str,
    split_payload: dict[str, Any],
    phase55_dataset: dict[str, Any] | None,
    phase54_path: Path,
    phase55_path: Path,
) -> dict[str, Any]:
    classification = split_payload.get("classification")
    route = split_payload.get("route") or {}
    selected = route.get("selected_conditioning_mode")
    route_label = _route_label(route)
    notes = " ".join(split_payload.get("notes") or [])
    if split_name == "spot_size" and phase55_dataset:
        seed_status = str(phase55_dataset.get("status"))
        aggregate_pass = bool((phase55_dataset.get("aggregate_gate") or {}).get("pass"))
        paired_pass = bool((phase55_dataset.get("paired_seed_gate") or {}).get("pass"))
        if seed_status == "seed_robust_transfer_positive":
            status = "paper_positive_seed_robust"
            process_claim = "yes"
            paper_placement = "main_table_and_main_figure"
            next_action = "Use as frozen floor; future candidates must beat or preserve this route."
        elif aggregate_pass:
            status = "aggregate_positive_seed_mixed"
            process_claim = "not_yet"
            paper_placement = "appendix_or_pending"
            next_action = "Do not claim until paired seed gate passes."
        else:
            status = "seed_unstable_or_negative"
            process_claim = "no"
            paper_placement = "appendix_diagnostic"
            next_action = "Do not expand without a new focused gate."
        return {
            "kind": "process_axis",
            "dataset": dataset_label,
            "split_or_branch": split_name,
            "status": status,
            "process_conditioning_claim": process_claim,
            "route": route_label,
            "metrics_gate": f"aggregate_pass={aggregate_pass}",
            "seed_gate": f"paired_pass={paired_pass}; status={seed_status}",
            "paper_placement": paper_placement,
            "current_evidence": str(phase55_path),
            "next_action": next_action,
            "notes": notes,
        }
    if classification == "paper_claim_positive" and selected == "none":
        return {
            "kind": "process_axis",
            "dataset": dataset_label,
            "split_or_branch": split_name,
            "status": "route_guard_no_process_positive",
            "process_conditioning_claim": "no",
            "route": route_label,
            "metrics_gate": "strong-baseline-positive via no-process fallback",
            "seed_gate": "not_seed_expanded_for_process_claim",
            "paper_placement": "route_guard_boundary_table",
            "current_evidence": str(phase54_path),
            "next_action": "Use only as fallback evidence, not process-conditioning improvement.",
            "notes": notes,
        }
    if classification == "paper_claim_positive":
        status = "paper_positive_single_summary"
        placement = "candidate_table_pending_seed_gate"
        process_claim = "pending_seed_gate"
        next_action = "Run seed gate before any manuscript claim."
    elif classification == "route_guard_positive":
        status = "route_guard_only"
        placement = "route_guard_boundary_table"
        process_claim = "no"
        next_action = "Keep as boundary evidence unless a new branch beats strong baselines."
    elif classification == "incomplete_metric":
        status = "incomplete_metric"
        placement = "not_claimable"
        process_claim = "no"
        next_action = "Regenerate missing metrics before classification."
    elif classification == "incomparable":
        status = "incomparable"
        placement = "not_claimable"
        process_claim = "no"
        next_action = "Fix manifest/split comparability before classification."
    else:
        status = "diagnostic_negative"
        placement = "appendix_diagnostic"
        process_claim = "no"
        next_action = "Do not continue without a new predeclared hypothesis."
    return {
        "kind": "process_axis",
        "dataset": dataset_label,
        "split_or_branch": split_name,
        "status": status,
        "process_conditioning_claim": process_claim,
        "route": route_label,
        "metrics_gate": classification,
        "seed_gate": "not_seed_expanded",
        "paper_placement": placement,
        "current_evidence": str(phase54_path),
        "next_action": next_action,
        "notes": notes,
    }


def _diagnostic_status(result: str) -> str:
    lowered = result.lower()
    if "data-incompatible" in lowered or "blocked" in lowered:
        return "blocked_by_data"
    if "negative" in lowered or "unstable" in lowered or "non-transferable" in lowered:
        return "diagnostic_negative"
    if "synthetic-positive" in lowered:
        return "diagnostic_negative"
    return "appendix_diagnostic"


def _diagnostic_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result = row.get("result") or ""
            rows.append(
                {
                    "kind": "diagnostic_branch",
                    "dataset": "",
                    "split_or_branch": f"Phase {row.get('phase', '')}: {row.get('branch', '')}",
                    "status": _diagnostic_status(result),
                    "process_conditioning_claim": "no",
                    "route": "",
                    "metrics_gate": result,
                    "seed_gate": "closed_or_blocked",
                    "paper_placement": "appendix_diagnostic_or_limitation",
                    "current_evidence": row.get("evidence") or str(path),
                    "next_action": "Keep as negative evidence unless the documented blocker is removed.",
                    "notes": row.get("paper_use") or "",
                }
            )
    return rows


def build_ledger(
    phase54: dict[str, Any],
    phase55: dict[str, Any],
    phase54_path: Path,
    phase55_path: Path,
    negative_table_path: Path | None = None,
) -> list[dict[str, Any]]:
    phase55_map = _phase55_by_dataset(phase55)
    rows: list[dict[str, Any]] = []
    for dataset in phase54.get("datasets", []):
        label = str(dataset.get("label"))
        for split_name, split_payload in sorted((dataset.get("splits") or {}).items()):
            rows.append(
                _split_ledger_row(
                    label,
                    split_name,
                    split_payload,
                    phase55_map.get(label) if split_name == "spot_size" else None,
                    phase54_path,
                    phase55_path,
                )
            )
    rows.extend(_diagnostic_rows(negative_table_path))
    return rows


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key, _ in columns) + " |")
    return "\n".join(lines)


def build_markdown(contract: dict[str, Any], ledger_rows: list[dict[str, Any]]) -> str:
    floor = contract["frozen_floor"]
    process_rows = [row for row in ledger_rows if row.get("kind") == "process_axis"]
    diagnostic_counts = _count_by(
        [row for row in ledger_rows if row.get("kind") == "diagnostic_branch"], "status"
    )
    floor_rows: list[dict[str, Any]] = []
    for dataset, payload in (floor.get("datasets") or {}).items():
        metrics = payload.get("metrics") or {}
        floor_rows.append(
            {
                "dataset": dataset,
                "route": _route_label(payload.get("route") or {}),
                "status": payload.get("status"),
                "rmse": (metrics.get("rmse") or {}).get("mean"),
                "hot_q90_rmse": (metrics.get("hot_q90_rmse") or {}).get("mean"),
                "gradient_q90_rmse": (metrics.get("gradient_q90_rmse") or {}).get("mean"),
            }
        )
    lines = [
        "# AM-Bench Phase 57 Claim Governance",
        "",
        "## Frozen floor",
        "",
        "The frozen paper-facing floor is `broad_process_v1` on the `spot_size` holdout with seeds 7/1/2. Future branches must beat or preserve this floor before seed expansion.",
        "",
        _markdown_table(
            floor_rows,
            [
                ("dataset", "Dataset"),
                ("route", "Route"),
                ("status", "Seed status"),
                ("rmse", "RMSE"),
                ("hot_q90_rmse", "Hot q90 RMSE"),
                ("gradient_q90_rmse", "Gradient q90 RMSE"),
            ],
        ),
        "",
        "## Future branch contract",
        "",
        "- Compare every candidate against mean, kNN, ExtraTrees, no-process Macro PINN, `process_axis_v1`, and `broad_process_v1` when artifacts are comparable.",
        "- Select routes, hyperparameters, and feature families from train/validation evidence only.",
        "- Do not use test metrics to decide whether to seed-expand a branch.",
        "- A focused candidate may seed-expand only if it is non-worse than the frozen `broad_process_v1` floor on broad12 and broad21 for RMSE, hot q90 RMSE, and gradient q90 RMSE.",
        "- A manuscript model claim requires at least seeds 7/1/2 and strong-baseline comparison, not a single focused run.",
        "",
        "## Current process-axis ledger",
        "",
        _markdown_table(
            process_rows,
            [
                ("dataset", "Dataset"),
                ("split_or_branch", "Split"),
                ("status", "Status"),
                ("process_conditioning_claim", "Process claim"),
                ("route", "Route"),
                ("paper_placement", "Paper placement"),
            ],
        ),
        "",
        "## Diagnostic branch counts",
        "",
        _markdown_table(
            [
                {"status": status, "count": count}
                for status, count in sorted(diagnostic_counts.items())
            ],
            [("status", "Status"), ("count", "Count")],
        ),
        "",
        "## Claim wording boundary",
        "",
        "The main process-conditioning claim is limited to the seed-robust `spot_size` route. `line` supports the no-process fallback route guard, not a process-conditioning improvement. `laser_power`, `scan_speed`, and full `process` remain route-guard-only unless a future candidate passes the Phase 57 gate.",
        "",
    ]
    return "\n".join(lines)


def build_governance_package(
    root: Path,
    phase54_path: Path,
    phase55_path: Path,
    negative_table_path: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    phase54 = _read_json(phase54_path)
    phase55 = _read_json(phase55_path)
    contract = build_contract(phase54, phase55)
    ledger_rows = build_ledger(phase54, phase55, phase54_path, phase55_path, negative_table_path)
    seed_gate_eval = evaluate_candidate_against_floor(
        _floor_metric_means(contract["frozen_floor"]),
        _floor_metric_means(contract["frozen_floor"]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "phase57_claim_contract.json"
    ledger_path = output_dir / "phase57_claim_ledger.csv"
    markdown_path = output_dir / "phase57_claim_governance.md"
    manifest_path = output_dir / "phase57_claim_governance_manifest.json"

    _write_json(contract_path, contract)
    _write_csv(ledger_path, ledger_rows)
    markdown_path.write_text(build_markdown(contract, ledger_rows), encoding="utf-8")

    manifest = {
        "phase": 57,
        "objective": "long_term_claim_and_experiment_governance",
        "root": str(root),
        "inputs": {
            "phase54": str(phase54_path),
            "phase55": str(phase55_path),
            "phase56_negative_table": str(negative_table_path)
            if negative_table_path is not None
            else None,
        },
        "outputs": {
            "contract": str(contract_path),
            "ledger": str(ledger_path),
            "markdown": str(markdown_path),
            "manifest": str(manifest_path),
        },
        "ledger_counts": _count_by(ledger_rows, "status"),
        "seed_expansion_self_check": seed_gate_eval,
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--phase54-summary",
        type=Path,
        default=Path("outputs/reports/phase54_process_route_claim_boundary_summary.json"),
    )
    parser.add_argument(
        "--phase55-summary",
        type=Path,
        default=Path("outputs/reports/phase55_spot_size_route_seed_check_summary.json"),
    )
    parser.add_argument(
        "--phase56-negative-table",
        type=Path,
        default=Path(
            "docs/results/phase56_manuscript_package/"
            "phase56_negative_diagnostic_appendix_table.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase57_claim_governance"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    manifest = build_governance_package(
        root=root,
        phase54_path=(root / args.phase54_summary).resolve()
        if not args.phase54_summary.is_absolute()
        else args.phase54_summary,
        phase55_path=(root / args.phase55_summary).resolve()
        if not args.phase55_summary.is_absolute()
        else args.phase55_summary,
        negative_table_path=(root / args.phase56_negative_table).resolve()
        if args.phase56_negative_table and not args.phase56_negative_table.is_absolute()
        else args.phase56_negative_table,
        output_dir=(root / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
