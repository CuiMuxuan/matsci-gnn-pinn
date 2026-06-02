#!/usr/bin/env python3
"""Build a post-Phase-59 manuscript evidence package.

This package does not add training evidence. It consolidates the current
paper-facing floor, route boundaries, stress tests, and residual-anatomy gate
into auditable manuscript tables before another model branch is attempted.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = (
    ("rmse", "Test RMSE"),
    ("hot_q90_rmse", "Hot q90 RMSE"),
    ("gradient_q90_rmse", "Gradient q90 RMSE"),
)

STRESS_FIELDS = (
    "scenario",
    "dataset",
    "split",
    "metric",
    "status",
    "candidate",
    "comparator",
    "delta_vs_comparator",
    "selected_variant",
    "manuscript_use",
    "evidence",
)

NEXT_GATE_FIELDS = (
    "branch",
    "status",
    "entry_condition",
    "focused_validation",
    "seed_expansion_gate",
    "manuscript_rule",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Expected at least one data row in {path}")
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _metric_label(metric: str) -> str:
    return dict(METRICS).get(metric, metric)


def _metric_keys(payload: dict[str, Any]) -> list[str]:
    required = payload.get("required_metrics")
    if isinstance(required, list) and required:
        return [str(metric) for metric in required]
    return [metric for metric, _ in METRICS]


def _default_paths(root: Path) -> dict[str, Path]:
    return {
        "phase56_main": root
        / "docs/results/phase56_manuscript_package/phase56_main_spot_size_seed_positive_table.csv",
        "phase56_route": root
        / "docs/results/phase56_manuscript_package/phase56_route_guard_boundary_table.csv",
        "phase56_appendix": root
        / "docs/results/phase56_manuscript_package/phase56_negative_diagnostic_appendix_table.csv",
        "phase57_contract": root / "docs/results/phase57_claim_governance/phase57_claim_contract.json",
        "phase57_ledger": root / "docs/results/phase57_claim_governance/phase57_claim_ledger.csv",
        "phase58_stronger": root
        / "docs/results/phase58_stronger_baseline_stress/phase58_stronger_baseline_stress_summary.json",
        "phase58_density": root
        / "docs/results/phase58_sampling_panel_stress/phase58_sampling_density_stress_summary.json",
        "phase58_panel": root
        / "docs/results/phase58_sampling_panel_stress/phase58_process_panel_stress_summary.json",
        "phase59_anatomy": root
        / "docs/results/phase59_residual_anatomy/phase59_broad21_density_residual_anatomy.json",
        "phase59_upper": root
        / "docs/results/phase59_residual_anatomy/phase59_broad21_density_residual_upper_bound.json",
    }


def _stress_rows_from_stronger(
    path: Path, payload: dict[str, Any], root: Path | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in payload.get("datasets", []):
        label = str(dataset.get("label") or "")
        for metric in _metric_keys(payload):
            metrics = (dataset.get("metrics") or {}).get(metric) or {}
            comparator = metrics.get("best_baseline_after_stress") or {}
            rows.append(
                {
                    "scenario": "stronger_baseline_stress",
                    "dataset": label,
                    "split": "spot_size",
                    "metric": _metric_label(metric),
                    "status": "pass" if metrics.get("frozen_beats_best_after_stress") else "fail",
                    "candidate": metrics.get("frozen_broad_process_v1"),
                    "comparator": _method_value(comparator),
                    "delta_vs_comparator": metrics.get("delta_vs_best_after_stress"),
                    "selected_variant": "",
                    "manuscript_use": "supports fixed-sampling Phase 55 floor",
                    "evidence": _display_path(path, root),
                }
            )
    return rows


def _stress_rows_from_seed_summary(
    path: Path,
    payload: dict[str, Any],
    scenario: str,
    pass_use: str,
    fail_use: str,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split = str(payload.get("split") or "")
    for dataset in payload.get("datasets", []):
        label = str(dataset.get("label") or "")
        gate = dataset.get("aggregate_gate") or {}
        gate_pass = bool(gate.get("pass"))
        metrics = gate.get("metrics") or {}
        for metric in _metric_keys(payload):
            metric_payload = metrics.get(metric) or {}
            rows.append(
                {
                    "scenario": scenario,
                    "dataset": label,
                    "split": split,
                    "metric": _metric_label(metric),
                    "status": "pass" if gate_pass and metric_payload.get("beats_best_strong_baseline") else "boundary",
                    "candidate": metric_payload.get("candidate"),
                    "comparator": _method_value(
                        {
                            "method": metric_payload.get("best_strong_baseline_method"),
                            "value": metric_payload.get("best_strong_baseline"),
                        }
                    ),
                    "delta_vs_comparator": metric_payload.get("delta_vs_best_strong"),
                    "selected_variant": "",
                    "manuscript_use": pass_use
                    if gate_pass and metric_payload.get("beats_best_strong_baseline")
                    else fail_use,
                    "evidence": _display_path(path, root),
                }
            )
    return rows


def _stress_rows_from_residual_gate(
    anatomy_path: Path,
    anatomy: dict[str, Any],
    upper_path: Path,
    upper: dict[str, Any],
    root: Path | None = None,
) -> list[dict[str, Any]]:
    selected = upper.get("selected_variant") or {}
    decision = upper.get("decision") or {}
    rows = [
        {
            "scenario": "residual_upper_bound_gate",
            "dataset": "broad21_density",
            "split": str(anatomy.get("analysis_split") or "test"),
            "metric": "Test RMSE",
            "status": "blocks_model_expansion",
            "candidate": decision.get("candidate_rmse"),
            "comparator": f"{upper.get('reference')}: {_fmt(decision.get('reference_rmse'))}",
            "delta_vs_comparator": None
            if decision.get("candidate_rmse") is None or decision.get("reference_rmse") is None
            else float(decision["candidate_rmse"]) - float(decision["reference_rmse"]),
            "selected_variant": selected.get("name") or decision.get("selected_variant"),
            "manuscript_use": "route-boundary evidence; do not claim density-invariant robustness",
            "evidence": _display_path(upper_path, root),
        }
    ]
    worst = anatomy.get("worst_candidate_vs_reference") or []
    for item in worst[:3]:
        rows.append(
            {
                "scenario": "residual_anatomy_slice",
                "dataset": "broad21_density",
                "split": str(anatomy.get("analysis_split") or "test"),
                "metric": f"{item.get('field')}={item.get('value')}",
                "status": "boundary",
                "candidate": ((item.get("metrics") or {}).get(anatomy.get("candidate")) or {}).get("rmse"),
                "comparator": _method_value(
                    {
                        "method": anatomy.get("reference"),
                        "value": ((item.get("metrics") or {}).get(anatomy.get("reference")) or {}).get("rmse"),
                    }
                ),
                "delta_vs_comparator": item.get("delta_candidate_minus_reference_rmse"),
                "selected_variant": "",
                "manuscript_use": "appendix residual-slice evidence",
                "evidence": _display_path(anatomy_path, root),
            }
        )
    return rows


def _method_value(payload: dict[str, Any]) -> str:
    method = payload.get("method")
    value = payload.get("value")
    if method and value is not None:
        return f"{method}: {_fmt(value)}"
    if value is not None:
        return _fmt(value)
    return str(method or "")


def build_stress_boundary_rows(paths: dict[str, Path], root: Path | None = None) -> list[dict[str, Any]]:
    stronger = _read_json(paths["phase58_stronger"])
    density = _read_json(paths["phase58_density"])
    panel = _read_json(paths["phase58_panel"])
    anatomy = _read_json(paths["phase59_anatomy"])
    upper = _read_json(paths["phase59_upper"])
    rows: list[dict[str, Any]] = []
    rows.extend(_stress_rows_from_stronger(paths["phase58_stronger"], stronger, root))
    rows.extend(
        _stress_rows_from_seed_summary(
            paths["phase58_density"],
            density,
            "alternate_density_stress",
            "density stress support",
            "density-sensitive boundary",
            root,
        )
    )
    rows.extend(
        _stress_rows_from_seed_summary(
            paths["phase58_panel"],
            panel,
            "auxiliary_process_panel",
            "auxiliary process-panel support",
            "auxiliary process-panel boundary",
            root,
        )
    )
    rows.extend(
        _stress_rows_from_residual_gate(
            paths["phase59_anatomy"], anatomy, paths["phase59_upper"], upper, root
        )
    )
    return rows


def build_appendix_rows(
    phase56_appendix_rows: list[dict[str, str]],
    stress_rows: list[dict[str, Any]],
    upper: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = list(phase56_appendix_rows)
    rows.append(
        {
            "phase": "58",
            "branch": "Alternate-density broad21 spot_size stress",
            "target": "broad21 spot_size, seed 7, denser fixed sampling",
            "result": "Density-sensitive boundary",
            "paper_use": "Appendix/route-boundary: fixed-sampling claim is not density-invariant.",
            "evidence": "docs/results/phase58_sampling_panel_stress/phase58_sampling_density_stress_summary.md",
        }
    )
    rows.append(
        {
            "phase": "59",
            "branch": "Residual anatomy and no-test-leakage upper-bound probe",
            "target": "Phase 58 broad21 density failure",
            "result": "Negative model-expansion gate",
            "paper_use": "Appendix/route-boundary: validation-visible correction falls back to mean baseline.",
            "evidence": "docs/results/phase59_residual_anatomy/phase59_broad21_density_residual_upper_bound.md",
        }
    )
    if any(row.get("scenario") == "stronger_baseline_stress" and row.get("status") != "pass" for row in stress_rows):
        rows.append(
            {
                "phase": "58",
                "branch": "Stronger-baseline stress",
                "target": "Phase 55 spot_size floor",
                "result": "Negative stress gate",
                "paper_use": "Would demote main claim if present.",
                "evidence": "docs/results/phase58_stronger_baseline_stress/phase58_stronger_baseline_stress_summary.md",
            }
        )
    decision = upper.get("decision") or {}
    if decision.get("selected_beats_reference_rmse"):
        rows.append(
            {
                "phase": "59",
                "branch": "Residual correction upper-bound",
                "target": "Phase 58 broad21 density failure",
                "result": "Positive upper-bound diagnostic",
                "paper_use": "Would justify a new validation-visible correction branch.",
                "evidence": "docs/results/phase59_residual_anatomy/phase59_broad21_density_residual_upper_bound.md",
            }
        )
    return rows


def build_next_gate_rows(contract: dict[str, Any], upper: dict[str, Any]) -> list[dict[str, str]]:
    required = ", ".join(contract.get("required_metrics") or [metric for metric, _ in METRICS])
    datasets = ", ".join((contract.get("frozen_floor") or {}).get("required_datasets") or ["broad12", "broad21"])
    upper_decision = upper.get("decision") or {}
    density_gate_status = (
        "blocked by Phase 59 density gate"
        if not upper_decision.get("selected_beats_reference_rmse")
        else "eligible for a narrow residual-correction design"
    )
    common_seed_gate = (
        f"Seed-expand only if seed 7 is non-worse than broad_process_v1 on {datasets} for {required}."
    )
    return [
        {
            "branch": "Candidate A: physically constrained spot-size conditioning",
            "status": "paused",
            "entry_condition": "Requires a new validation-visible spot-size signal, not the Phase 58 density failure alone.",
            "focused_validation": f"Run broad12 and broad21 spot_size seed-7 gates against the frozen floor on {required}.",
            "seed_expansion_gate": common_seed_gate,
            "manuscript_rule": "Promote only if it beats or preserves the Phase 55 floor; otherwise keep Phase 55 as the main claim.",
        },
        {
            "branch": "Candidate B: validation-auditable route policy",
            "status": density_gate_status,
            "entry_condition": "Route choice must be selected from train/validation evidence and existing comparable artifacts.",
            "focused_validation": "First test as a non-trainable policy among mean/no-process/process routes before adding capacity.",
            "seed_expansion_gate": common_seed_gate,
            "manuscript_rule": "May appear in main text only if it preserves spot_size and improves a boundary axis without test leakage.",
        },
        {
            "branch": "Candidate C: heat-kernel or Green's-function source features",
            "status": "blocked by registration data",
            "entry_condition": "Requires aligned single-track scan-path metadata or a defensible pad thermography target.",
            "focused_validation": "Start with no-training-change feature gates before Macro PINN integration.",
            "seed_expansion_gate": common_seed_gate,
            "manuscript_rule": "Do not claim source-path physics on broad12/broad21 until registration is resolved.",
        },
        {
            "branch": "External robustness / second dataset branch",
            "status": "deferred until manuscript package is draft-ready",
            "entry_condition": "Use after the current AM-Bench claim package is internally complete.",
            "focused_validation": "Apply the same local feasibility, A100 focused gate, broad transfer, and seed protocol.",
            "seed_expansion_gate": "Define a new frozen floor before seed expansion on the added dataset.",
            "manuscript_rule": "Use only as external robustness unless it passes the same strong-baseline and route-guard gates.",
        },
    ]


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_fmt(row.get(key)).replace("\n", " ") for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def build_markdown(
    main_rows: list[dict[str, Any]],
    route_rows: list[dict[str, Any]],
    stress_rows: list[dict[str, Any]],
    appendix_rows: list[dict[str, Any]],
    next_gate_rows: list[dict[str, Any]],
    contract: dict[str, Any],
    upper: dict[str, Any],
) -> str:
    main_preview = [
        row for row in main_rows if row.get("metric") in {"Test RMSE", "Hot q90 RMSE", "Gradient q90 RMSE"}
    ]
    route_preview = [
        row
        for row in route_rows
        if row.get("classification") != "paper_claim_positive"
        or "no-process" in str(row.get("claim_use") or "")
    ]
    stress_preview = [
        row
        for row in stress_rows
        if row.get("scenario")
        in {"stronger_baseline_stress", "alternate_density_stress", "auxiliary_process_panel", "residual_upper_bound_gate"}
    ]
    transfer_gate = (contract.get("current_transfer_gate") or {}).get("paper_claim_status")
    upper_decision = upper.get("decision") or {}
    lines = [
        "# AM-Bench Phase 60 Manuscript Evidence Package",
        "",
        "## Purpose",
        "",
        "Phase 60 consolidates the post-Phase-59 evidence boundary for manuscript drafting. It does not add training evidence and does not reopen a model branch from the density failure.",
        "",
        "## Main Claim Floor",
        "",
        f"Current transfer gate: `{transfer_gate}`. The main paper-positive evidence remains the fixed-sampling Phase 55 `spot_size` result under `broad_process_v1` with seeds 7/1/2.",
        "",
        _markdown_table(
            main_preview,
            [
                ("dataset", "Dataset"),
                ("metric", "Metric"),
                ("broad_process_v1_mean", "broad_process_v1"),
                ("best_strong_baseline", "Best strong baseline"),
                ("delta_vs_best_strong", "Delta"),
                ("gate", "Gate"),
            ],
        ),
        "",
        "## Route Guard Boundaries",
        "",
        "Laser power, scan speed, full process, and no-process fallback axes remain boundary evidence unless a future branch passes the frozen-floor gate.",
        "",
        _markdown_table(
            route_preview,
            [
                ("dataset", "Dataset"),
                ("split", "Split"),
                ("classification", "Classification"),
                ("route", "Route"),
                ("claim_use", "Use"),
            ],
        ),
        "",
        "## Stress and Residual Boundaries",
        "",
        "Stronger-baseline stress supports the fixed-sampling floor, while alternate-density broad21 is a density-sensitive boundary. Phase 59 selected a mean fallback from validation, so the density failure is not a model-expansion signal.",
        "",
        _markdown_table(
            stress_preview,
            [
                ("scenario", "Scenario"),
                ("dataset", "Dataset"),
                ("metric", "Metric"),
                ("status", "Status"),
                ("candidate", "Candidate"),
                ("comparator", "Comparator"),
                ("delta_vs_comparator", "Delta"),
                ("manuscript_use", "Use"),
            ],
        ),
        "",
        "## Next Branch Gates",
        "",
        _markdown_table(
            next_gate_rows,
            [
                ("branch", "Branch"),
                ("status", "Status"),
                ("entry_condition", "Entry condition"),
                ("seed_expansion_gate", "Seed gate"),
            ],
        ),
        "",
        "## Manuscript Placement",
        "",
        "- Main table: `phase60_main_spot_size_seed_positive_table.csv`.",
        "- Route-guard table: `phase60_route_guard_boundary_table.csv`.",
        "- Stress/boundary table: `phase60_stress_boundary_table.csv`.",
        "- Appendix negative/boundary diagnostics: `phase60_appendix_negative_diagnostic_table.csv`.",
        "",
        "## Claim Guardrail",
        "",
        f"Phase 59 selected `{upper_decision.get('selected_variant')}` with `uses_test_for_selection={upper.get('uses_test_for_selection')}`. This blocks density-failure-driven model expansion until a new validation-visible signal appears.",
        "",
        f"Appendix rows after Phase 60: `{len(appendix_rows)}`.",
        "",
    ]
    return "\n".join(lines)


def build_package(
    root: Path,
    output_dir: Path,
    paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    resolved = _default_paths(root)
    if paths:
        resolved.update(paths)

    main_rows = _read_csv(resolved["phase56_main"])
    route_rows = _read_csv(resolved["phase56_route"])
    phase56_appendix_rows = _read_csv(resolved["phase56_appendix"])
    contract = _read_json(resolved["phase57_contract"])
    ledger_rows = _read_csv(resolved["phase57_ledger"])
    upper = _read_json(resolved["phase59_upper"])
    stress_rows = build_stress_boundary_rows(resolved, root=root)
    appendix_rows = build_appendix_rows(phase56_appendix_rows, stress_rows, upper)
    next_gate_rows = build_next_gate_rows(contract, upper)

    output_dir.mkdir(parents=True, exist_ok=True)
    main_csv = output_dir / "phase60_main_spot_size_seed_positive_table.csv"
    route_csv = output_dir / "phase60_route_guard_boundary_table.csv"
    stress_csv = output_dir / "phase60_stress_boundary_table.csv"
    appendix_csv = output_dir / "phase60_appendix_negative_diagnostic_table.csv"
    next_gate_csv = output_dir / "phase60_next_branch_gate_table.csv"
    markdown_path = output_dir / "phase60_manuscript_evidence_package.md"
    manifest_path = output_dir / "phase60_manuscript_evidence_package_manifest.json"

    _write_csv(main_csv, main_rows, list(main_rows[0].keys()))
    _write_csv(route_csv, route_rows, list(route_rows[0].keys()))
    _write_csv(stress_csv, stress_rows, STRESS_FIELDS)
    _write_csv(appendix_csv, appendix_rows, list(appendix_rows[0].keys()))
    _write_csv(next_gate_csv, next_gate_rows, NEXT_GATE_FIELDS)
    markdown_path.write_text(
        build_markdown(main_rows, route_rows, stress_rows, appendix_rows, next_gate_rows, contract, upper),
        encoding="utf-8",
    )

    stress_counts: dict[str, int] = {}
    for row in stress_rows:
        key = str(row.get("status") or "")
        stress_counts[key] = stress_counts.get(key, 0) + 1
    ledger_counts: dict[str, int] = {}
    for row in ledger_rows:
        key = str(row.get("status") or "")
        ledger_counts[key] = ledger_counts.get(key, 0) + 1

    model_expansion_gate = {
        "decision": "block_density_failure_driven_model_expansion"
        if not ((upper.get("decision") or {}).get("selected_beats_reference_rmse"))
        else "allow_narrow_residual_branch_design",
        "reason": (upper.get("decision") or {}).get("interpretation"),
        "selected_variant": ((upper.get("selected_variant") or {}).get("name")),
        "uses_test_for_selection": upper.get("uses_test_for_selection"),
    }
    manifest = {
        "phase": 60,
        "objective": "post_phase59_manuscript_evidence_package",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "main_table": _display_path(main_csv, root),
            "route_guard_table": _display_path(route_csv, root),
            "stress_boundary_table": _display_path(stress_csv, root),
            "appendix_table": _display_path(appendix_csv, root),
            "next_branch_gate_table": _display_path(next_gate_csv, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "main_rows": len(main_rows),
            "route_guard_rows": len(route_rows),
            "stress_boundary_rows": len(stress_rows),
            "appendix_rows": len(appendix_rows),
            "next_branch_gate_rows": len(next_gate_rows),
            "claim_ledger_status_counts": ledger_counts,
            "stress_status_counts": stress_counts,
        },
        "claim_boundary": {
            "main_claim": "fixed-sampling broad12/broad21 spot_size under broad_process_v1, seeds 7/1/2",
            "excluded_claims": [
                "density-invariant robustness",
                "universal process-conditioning success",
                "laser_power, scan_speed, or full-process strong-baseline wins",
                "source-path or Green's-function broad12/broad21 success under current data registration",
            ],
        },
        "model_expansion_gate": model_expansion_gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase60_manuscript_evidence_package"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    manifest = build_package(root=root, output_dir=output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
