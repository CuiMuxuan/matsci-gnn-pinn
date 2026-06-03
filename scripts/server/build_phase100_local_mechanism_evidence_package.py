#!/usr/bin/env python3
"""Build the Phase 100 local mechanism evidence package.

Phase 100 packages the Phase 96/98/99 fixed Green's-function / heat-kernel
evidence as local mechanism evidence. It does not unlock AM-Bench transfer or
A100 training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EVIDENCE_FIELDS = (
    "row_id",
    "source_phase",
    "evidence_type",
    "artifact",
    "mechanism",
    "claim_scope",
    "status",
    "key_metric_summary",
    "paper_use",
    "notes",
)

BOUNDARY_FIELDS = (
    "boundary_id",
    "source_phase",
    "boundary_type",
    "evidence",
    "blocked_claim",
    "required_before_unlock",
    "status",
)

CLAIM_FIELDS = (
    "claim_id",
    "claim_text",
    "claim_location",
    "supporting_evidence",
    "allowed_now",
    "status",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _default_paths(root: Path) -> dict[str, Path]:
    phase96 = root / "docs/results/phase96_pfhub_local_smoke_gate"
    phase98 = root / "docs/results/phase98_registered_target_unlock_gate"
    phase99 = root / "docs/results/phase99_registered_surrogate_smoke_gate"
    return {
        "phase96_gate": phase96 / "phase96_pfhub_local_smoke_gate.json",
        "phase96_metric_table": phase96 / "phase96_local_smoke_metric_table.csv",
        "phase96_mechanism_table": phase96 / "phase96_mechanism_decision_table.csv",
        "phase96_target_manifest": phase96 / "phase96_pfhub_style_target_manifest.json",
        "phase98_gate": phase98 / "phase98_registered_target_unlock_gate.json",
        "phase98_surrogate_card": phase98 / "phase98_registered_surrogate_data_card.json",
        "phase98_unlock_table": phase98 / "phase98_unlock_candidate_table.csv",
        "phase99_gate": phase99 / "phase99_registered_surrogate_smoke_gate.json",
        "phase99_metric_table": phase99 / "phase99_registered_surrogate_metric_table.csv",
        "phase99_comparison_table": phase99 / "phase99_registered_surrogate_comparison_table.csv",
    }


def _row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise ValueError(f"Missing row where {key}={value}")


def _count(rows: list[dict[str, str]], key: str, value: str) -> int:
    return sum(1 for row in rows if row.get(key) == value)


def _metric_summary(*, candidate: dict[str, str], baseline: dict[str, str]) -> str:
    return (
        f"test_rmse {baseline['test_rmse']}->{candidate['test_rmse']}; "
        f"pde_residual {baseline.get('test_pde_residual_rmse', '')}->{candidate.get('test_pde_residual_rmse', '')}; "
        f"hot_q90 {baseline['test_hot_q90_rmse']}->{candidate['test_hot_q90_rmse']}; "
        f"gradient_q90 {baseline['test_gradient_q90_rmse']}->{candidate['test_gradient_q90_rmse']}"
    )


def build_evidence_rows(
    *,
    paths: dict[str, Path],
    phase96_gate: dict[str, Any],
    phase96_metrics: list[dict[str, str]],
    phase96_mechanisms: list[dict[str, str]],
    phase96_target: dict[str, Any],
    phase98_gate: dict[str, Any],
    phase98_card: dict[str, Any],
    phase98_unlock_rows: list[dict[str, str]],
    phase99_gate: dict[str, Any],
    phase99_metrics: list[dict[str, str]],
    phase99_comparisons: list[dict[str, str]],
    root: Path,
) -> list[dict[str, Any]]:
    phase96_fixed = _row_by(phase96_metrics, "method_id", "fixed_green_function_features")
    phase96_vanilla = _row_by(phase96_metrics, "method_id", "vanilla_deterministic_surrogate")
    phase96_bayes = _row_by(phase96_metrics, "method_id", "bayesian_adaptive_collocation")
    phase96_random = _row_by(phase96_metrics, "method_id", "random_collocation_same_budget")
    phase99_fixed = _row_by(phase99_metrics, "method_id", "fixed_green_function_features_full")
    phase99_vanilla = _row_by(phase99_metrics, "method_id", "vanilla_deterministic_surrogate_full")
    phase99_random = _row_by(phase99_metrics, "method_id", "random_collocation_same_budget")
    full_grid_passes = [
        row
        for row in phase99_comparisons
        if row.get("scope") == "full-grid baselines" and row.get("pass") == "true"
    ]
    boundary_failures = [
        row
        for row in phase99_comparisons
        if row.get("scope") == "same-budget boundary audit" and row.get("pass") == "false"
    ]
    open_surrogate_rows = [
        row
        for row in phase98_unlock_rows
        if row.get("phase99_local_smoke_allowed") == "true"
    ]

    return [
        {
            "row_id": "P100-EVID-001",
            "source_phase": 96,
            "evidence_type": "deterministic_target_manifest",
            "artifact": _display_path(paths["phase96_target_manifest"], root),
            "mechanism": "manufactured_heat_diffusion_with_moving_source",
            "claim_scope": "local_physics_benchmark",
            "status": phase96_gate.get("status"),
            "key_metric_summary": (
                f"target={phase96_target.get('target_id')}; "
                f"train={phase96_target.get('splits', {}).get('train_grid')}; "
                f"validation={phase96_target.get('splits', {}).get('validation_grid')}; "
                f"test={phase96_target.get('splits', {}).get('test_grid')}"
            ),
            "paper_use": "appendix target and reproducibility description",
            "notes": "Generated target is registered by construction and is not AM-Bench evidence.",
        },
        {
            "row_id": "P100-EVID-002",
            "source_phase": 96,
            "evidence_type": "mechanism_positive",
            "artifact": _display_path(paths["phase96_metric_table"], root),
            "mechanism": "fixed_green_function_features",
            "claim_scope": "local_mechanism_positive",
            "status": _row_by(
                phase96_mechanisms, "mechanism", "fixed_green_function_features"
            ).get("next_action"),
            "key_metric_summary": _metric_summary(candidate=phase96_fixed, baseline=phase96_vanilla),
            "paper_use": "appendix mechanism evidence or methods ablation",
            "notes": "Validation-selected fixed-kernel features pass Phase 96 local smoke.",
        },
        {
            "row_id": "P100-EVID-003",
            "source_phase": 96,
            "evidence_type": "mechanism_negative_control",
            "artifact": _display_path(paths["phase96_metric_table"], root),
            "mechanism": "bayesian_adaptive_collocation",
            "claim_scope": "diagnostic_only",
            "status": _row_by(
                phase96_mechanisms, "mechanism", "bayesian_adaptive_collocation"
            ).get("next_action"),
            "key_metric_summary": _metric_summary(candidate=phase96_bayes, baseline=phase96_random),
            "paper_use": "appendix negative diagnostic",
            "notes": "Adaptive collocation improved hot-region error but failed global/residual guards.",
        },
        {
            "row_id": "P100-EVID-004",
            "source_phase": 98,
            "evidence_type": "registered_surrogate_data_card",
            "artifact": _display_path(paths["phase98_surrogate_card"], root),
            "mechanism": "analytic_registration",
            "claim_scope": "registered_surrogate_only",
            "status": phase98_gate.get("status"),
            "key_metric_summary": (
                f"candidate={phase98_card.get('candidate_id')}; "
                f"registration={phase98_card.get('registration_story')}"
            ),
            "paper_use": "appendix data-card evidence",
            "notes": f"{len(open_surrogate_rows)} generated surrogate row opened Phase 99; AM-Bench rows remain locked.",
        },
        {
            "row_id": "P100-EVID-005",
            "source_phase": 99,
            "evidence_type": "baseline_first_positive",
            "artifact": _display_path(paths["phase99_metric_table"], root),
            "mechanism": "fixed_green_function_features_full",
            "claim_scope": "full_grid_registered_surrogate_positive",
            "status": phase99_gate.get("status"),
            "key_metric_summary": _metric_summary(candidate=phase99_fixed, baseline=phase99_vanilla),
            "paper_use": "appendix mechanism table",
            "notes": f"{len(full_grid_passes)} full-grid comparison rows passed.",
        },
        {
            "row_id": "P100-EVID-006",
            "source_phase": 99,
            "evidence_type": "same_budget_boundary",
            "artifact": _display_path(paths["phase99_comparison_table"], root),
            "mechanism": "fixed_green_function_features_full_vs_random_collocation",
            "claim_scope": "boundary_disclosed",
            "status": "focused_boundary_failure_present",
            "key_metric_summary": _metric_summary(candidate=phase99_fixed, baseline=phase99_random),
            "paper_use": "appendix boundary note",
            "notes": f"{len(boundary_failures)} same-budget boundary failure rows; failing metric(s): "
            + ", ".join(sorted({row.get("metric", "") for row in boundary_failures})),
        },
    ]


def build_boundary_rows(
    *,
    phase98_gate: dict[str, Any],
    phase98_unlock_rows: list[dict[str, str]],
    phase99_gate: dict[str, Any],
) -> list[dict[str, Any]]:
    blocked_registered_rows = [
        row
        for row in phase98_unlock_rows
        if row.get("phase99_local_smoke_allowed") == "false"
        and row.get("candidate_family") != "synthetic_benchmark_appendix"
    ]
    return [
        {
            "boundary_id": "P100-BND-001",
            "source_phase": 97,
            "boundary_type": "transfer_registration_blocker",
            "evidence": "Phase 97/98 AM-Bench and external rows lack paper-facing registration or data cards.",
            "blocked_claim": "AM-Bench transfer for fixed heat-kernel / Green's-function features",
            "required_before_unlock": "public registered AM-Bench or external target with source manifest and split plan",
            "status": "active",
        },
        {
            "boundary_id": "P100-BND-002",
            "source_phase": 98,
            "boundary_type": "registered_target_lock",
            "evidence": (
                f"am_bench_transfer_unlocked={str(phase98_gate.get('am_bench_transfer_unlocked')).lower()}; "
                f"blocked_registered_rows={len(blocked_registered_rows)}"
            ),
            "blocked_claim": "real-data registered-source transfer",
            "required_before_unlock": "camera-to-galvo registration or external registered target data card",
            "status": "active",
        },
        {
            "boundary_id": "P100-BND-003",
            "source_phase": 99,
            "boundary_type": "same_budget_focused_boundary",
            "evidence": (
                f"same_budget_boundary_failures={phase99_gate.get('same_budget_boundary_failures')}; "
                "fixed full-grid gradient q90 is worse than random same-budget collocation"
            ),
            "blocked_claim": "budget-matched focused-region superiority",
            "required_before_unlock": "new local gate where same-budget global/hot/gradient comparisons all pass",
            "status": "active",
        },
        {
            "boundary_id": "P100-BND-004",
            "source_phase": 100,
            "boundary_type": "a100_training_lock",
            "evidence": (
                f"a100_training_allowed_now={str(phase99_gate.get('a100_training_allowed_now')).lower()}; "
                f"a100_80gb_request_now={str(phase99_gate.get('a100_80gb_request_now')).lower()}"
            ),
            "blocked_claim": "A100 broad validation or A100-SXM4-80GB request",
            "required_before_unlock": "registered target, baseline-first smoke, non-worse metrics, pushed-commit server validation",
            "status": "active",
        },
    ]


def build_claim_rows() -> list[dict[str, Any]]:
    return [
        {
            "claim_id": "P100-CLAIM-001",
            "claim_text": "Fixed Green's-function / heat-kernel features are effective on a registered analytic heat-source surrogate.",
            "claim_location": "appendix or method ablation",
            "supporting_evidence": "P100-EVID-001;P100-EVID-002;P100-EVID-004;P100-EVID-005",
            "allowed_now": True,
            "status": "allowed_local_mechanism_claim",
        },
        {
            "claim_id": "P100-CLAIM-002",
            "claim_text": "The fixed-kernel mechanism transfers to AM-Bench broad12/broad21 thermal/process prediction.",
            "claim_location": "main paper",
            "supporting_evidence": "P100-BND-001;P100-BND-002",
            "allowed_now": False,
            "status": "blocked_no_registered_transfer_target",
        },
        {
            "claim_id": "P100-CLAIM-003",
            "claim_text": "The fixed-kernel mechanism is superior under all same-budget focused-region comparisons.",
            "claim_location": "main paper or appendix",
            "supporting_evidence": "P100-EVID-006;P100-BND-003",
            "allowed_now": False,
            "status": "blocked_same_budget_gradient_boundary",
        },
        {
            "claim_id": "P100-CLAIM-004",
            "claim_text": "New A100 broad validation or A100-SXM4-80GB training is justified now.",
            "claim_location": "execution plan",
            "supporting_evidence": "P100-BND-004",
            "allowed_now": False,
            "status": "blocked_training_lock",
        },
    ]


def build_gate(
    *,
    phase96_gate: dict[str, Any],
    phase98_gate: dict[str, Any],
    phase99_gate: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase99_allows_package = bool(phase99_gate.get("phase100_local_mechanism_package_allowed"))
    fixed_positive = "fixed_green_function_features" in phase96_gate.get("positive_mechanisms", [])
    transfer_locked = not bool(phase98_gate.get("am_bench_transfer_unlocked"))
    boundary_disclosed = any(
        row.get("boundary_type") == "same_budget_focused_boundary" for row in boundary_rows
    )
    if phase99_allows_package and fixed_positive and transfer_locked and boundary_disclosed:
        status = "local_mechanism_package_ready_transfer_locked"
        appendix_ready = True
        next_action = "enter Phase 101 registered target acquisition gate before any A100 training"
    else:
        status = "blocked_local_mechanism_package_incomplete"
        appendix_ready = False
        next_action = "repair Phase 96/98/99 evidence before packaging"

    return {
        "status": status,
        "appendix_local_mechanism_ready": appendix_ready,
        "main_paper_transfer_claim_ready": False,
        "phase101_registered_target_acquisition_allowed": appendix_ready,
        "am_bench_transfer_unlocked": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "submission_ready": False,
        "source_phase96_status": phase96_gate.get("status"),
        "source_phase98_status": phase98_gate.get("status"),
        "source_phase99_status": phase99_gate.get("status"),
        "evidence_rows": len(evidence_rows),
        "boundary_rows": len(boundary_rows),
        "claim_rows": len(claim_rows),
        "allowed_claim_rows": _count(
            [{**row, "allowed_now": _csv_value(row.get("allowed_now"))} for row in claim_rows],
            "allowed_now",
            "true",
        ),
        "blocked_claim_rows": _count(
            [{**row, "allowed_now": _csv_value(row.get("allowed_now"))} for row in claim_rows],
            "allowed_now",
            "false",
        ),
        "same_budget_boundary_disclosed": boundary_disclosed,
        "next_action": next_action,
        "required_before_a100_training": [
            "public registered AM-Bench/external target",
            "baseline-first local smoke on that registered target",
            "non-worse global/hot/gradient metrics",
            "server validation from a pushed commit",
        ],
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_csv_value(row.get(key)).replace("\n", " ") for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def build_markdown(
    *,
    gate: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Phase 100 Local Mechanism Evidence Package",
            "",
            "## Gate Decision",
            "",
            f"Status: `{gate['status']}`.",
            f"Appendix/local mechanism ready: `{str(gate['appendix_local_mechanism_ready']).lower()}`.",
            f"Main-paper transfer claim ready: `{str(gate['main_paper_transfer_claim_ready']).lower()}`.",
            f"AM-Bench transfer unlocked: `{str(gate['am_bench_transfer_unlocked']).lower()}`.",
            f"A100 training allowed now: `{str(gate['a100_training_allowed_now']).lower()}`.",
            f"A100-SXM4-80GB request now: `{str(gate['a100_80gb_request_now']).lower()}`.",
            "",
            "Phase 100 packages local fixed-kernel mechanism evidence. It does not provide AM-Bench transfer evidence.",
            "",
            "## Evidence Rows",
            "",
            _markdown_table(
                evidence_rows,
                [
                    ("row_id", "Row"),
                    ("source_phase", "Phase"),
                    ("mechanism", "Mechanism"),
                    ("claim_scope", "Scope"),
                    ("status", "Status"),
                    ("key_metric_summary", "Summary"),
                ],
            ),
            "",
            "## Boundaries",
            "",
            _markdown_table(
                boundary_rows,
                [
                    ("boundary_id", "Boundary"),
                    ("boundary_type", "Type"),
                    ("blocked_claim", "Blocked claim"),
                    ("required_before_unlock", "Required before unlock"),
                    ("status", "Status"),
                ],
            ),
            "",
            "## Claim Use",
            "",
            _markdown_table(
                claim_rows,
                [
                    ("claim_id", "Claim"),
                    ("claim_location", "Location"),
                    ("allowed_now", "Allowed"),
                    ("status", "Status"),
                ],
            ),
            "",
            "## Next Action",
            "",
            gate["next_action"],
            "",
        ]
    )


def build_package(
    root: Path,
    output_dir: Path,
    paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    resolved = _default_paths(root)
    if paths:
        resolved.update(paths)

    phase96_gate = _read_json(resolved["phase96_gate"])
    phase96_metrics = _read_csv(resolved["phase96_metric_table"])
    phase96_mechanisms = _read_csv(resolved["phase96_mechanism_table"])
    phase96_target = _read_json(resolved["phase96_target_manifest"])
    phase98_gate = _read_json(resolved["phase98_gate"])
    phase98_card = _read_json(resolved["phase98_surrogate_card"])
    phase98_unlock_rows = _read_csv(resolved["phase98_unlock_table"])
    phase99_gate = _read_json(resolved["phase99_gate"])
    phase99_metrics = _read_csv(resolved["phase99_metric_table"])
    phase99_comparisons = _read_csv(resolved["phase99_comparison_table"])

    evidence_rows = build_evidence_rows(
        paths=resolved,
        phase96_gate=phase96_gate,
        phase96_metrics=phase96_metrics,
        phase96_mechanisms=phase96_mechanisms,
        phase96_target=phase96_target,
        phase98_gate=phase98_gate,
        phase98_card=phase98_card,
        phase98_unlock_rows=phase98_unlock_rows,
        phase99_gate=phase99_gate,
        phase99_metrics=phase99_metrics,
        phase99_comparisons=phase99_comparisons,
        root=root,
    )
    boundary_rows = build_boundary_rows(
        phase98_gate=phase98_gate,
        phase98_unlock_rows=phase98_unlock_rows,
        phase99_gate=phase99_gate,
    )
    claim_rows = build_claim_rows()
    gate = build_gate(
        phase96_gate=phase96_gate,
        phase98_gate=phase98_gate,
        phase99_gate=phase99_gate,
        evidence_rows=evidence_rows,
        boundary_rows=boundary_rows,
        claim_rows=claim_rows,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "phase100_local_mechanism_evidence_table.csv"
    boundary_path = output_dir / "phase100_local_mechanism_boundary_table.csv"
    claim_path = output_dir / "phase100_local_mechanism_claim_use_table.csv"
    gate_path = output_dir / "phase100_local_mechanism_evidence_package.json"
    markdown_path = output_dir / "phase100_local_mechanism_evidence_package.md"
    manifest_path = output_dir / "phase100_local_mechanism_evidence_package_manifest.json"

    _write_csv(evidence_path, evidence_rows, EVIDENCE_FIELDS)
    _write_csv(boundary_path, boundary_rows, BOUNDARY_FIELDS)
    _write_csv(claim_path, claim_rows, CLAIM_FIELDS)
    _write_json(gate_path, gate)
    markdown_path.write_text(
        build_markdown(
            gate=gate,
            evidence_rows=evidence_rows,
            boundary_rows=boundary_rows,
            claim_rows=claim_rows,
        ),
        encoding="utf-8",
    )

    manifest = {
        "phase": 100,
        "objective": "local_mechanism_evidence_packaging",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "evidence_table": _display_path(evidence_path, root),
            "boundary_table": _display_path(boundary_path, root),
            "claim_use_table": _display_path(claim_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "evidence_rows": len(evidence_rows),
            "boundary_rows": len(boundary_rows),
            "claim_rows": len(claim_rows),
            "allowed_claim_rows": gate["allowed_claim_rows"],
            "blocked_claim_rows": gate["blocked_claim_rows"],
        },
        "gate": gate,
        "source_gates": {
            "phase96_status": phase96_gate.get("status"),
            "phase98_status": phase98_gate.get("status"),
            "phase99_status": phase99_gate.get("status"),
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase100_local_mechanism_evidence_package"),
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
