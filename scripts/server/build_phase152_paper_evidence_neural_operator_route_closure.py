#!/usr/bin/env python3
"""Build Phase 152 paper-evidence refresh for the closed neural-operator route."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path(
    "docs/results/phase152_paper_evidence_neural_operator_route_closure"
)

PHASE_INPUTS = {
    "phase146_gate": Path(
        "docs/results/phase146_paper_evidence_refresh/"
        "phase146_paper_evidence_refresh_gate.json"
    ),
    "phase149_gate": Path(
        "docs/results/phase149_neural_operator_readiness_gate/"
        "phase149_neural_operator_readiness_gate.json"
    ),
    "phase149_readiness_table": Path(
        "docs/results/phase149_neural_operator_readiness_gate/"
        "phase149_neural_operator_readiness_table.csv"
    ),
    "phase150_gate": Path(
        "docs/results/phase150_dense_tensorization_inventory_gate/"
        "phase150_dense_tensorization_inventory_gate.json"
    ),
    "phase150_inventory_table": Path(
        "docs/results/phase150_dense_tensorization_inventory_gate/"
        "phase150_dense_source_inventory_table.csv"
    ),
    "phase151_gate": Path(
        "docs/results/phase151_fixed_grid_dense_baseline_review/"
        "phase151_fixed_grid_dense_baseline_gate.json"
    ),
    "phase151_review_table": Path(
        "docs/results/phase151_fixed_grid_dense_baseline_review/"
        "phase151_dense_baseline_review_table.csv"
    ),
    "phase151_split_table": Path(
        "docs/results/phase151_fixed_grid_dense_baseline_review/"
        "phase151_split_contract_table.csv"
    ),
}

ROUTE_FIELDS = (
    "route_id",
    "source_phase",
    "artifact",
    "route_status",
    "evidence_summary",
    "paper_use",
    "operator_training_allowed",
    "model_training_allowed",
    "a100_training_allowed_now",
    "blocks_neural_operator_success_claim",
    "next_action",
)

CLAIM_FIELDS = (
    "claim_id",
    "claim_area",
    "claim_status",
    "paper_boundary",
    "evidence_anchor",
    "allowed_final_use",
)

DECISION_FIELDS = (
    "decision_id",
    "decision",
    "status",
    "rationale",
    "evidence_anchor",
    "next_action",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is None:
        return str(path).replace("\\", "/")
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _is_false(value: Any) -> bool:
    if value is False or value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "0", "false", "none", "no"}
    return False


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _count_truthy(rows: list[dict[str, str]], column: str) -> int:
    return sum(1 for row in rows if _is_true(row.get(column)))


def _count_status(rows: list[dict[str, str]], column: str, status: str) -> int:
    return sum(1 for row in rows if row.get(column) == status)


def build_route_rows(
    *,
    root: Path,
    phase146_gate: dict[str, Any],
    phase149_gate: dict[str, Any],
    phase150_gate: dict[str, Any],
    phase151_gate: dict[str, Any],
    phase149_readiness_rows: list[dict[str, str]],
    phase150_inventory_rows: list[dict[str, str]],
    phase151_review_rows: list[dict[str, str]],
    phase151_split_rows: list[dict[str, str]],
    phase_inputs: dict[str, Path],
) -> list[dict[str, Any]]:
    readiness_blockers = _count_truthy(phase149_readiness_rows, "blocks_operator_training")
    present_dense_sources = _count_truthy(phase150_inventory_rows, "present")
    tensorizable_sources = _count_status(
        phase150_inventory_rows,
        "tensorization_status",
        "candidate_indexed_dense_csv_needs_split_and_operator_baseline",
    )
    diagnostic_splits = _count_status(
        phase151_split_rows, "split_contract_status", "diagnostic_frame_block_split_only"
    )
    leakage_safe_splits = _count_truthy(phase151_split_rows, "leakage_safe_split")
    strong_baseline_solved = _count_truthy(phase151_review_rows, "strong_baseline_solved")

    return [
        {
            "route_id": "P152-ROUTE-001",
            "source_phase": "phase146_paper_evidence_refresh",
            "artifact": _display_path(phase_inputs["phase146_gate"], root),
            "route_status": phase146_gate.get("status", "missing"),
            "evidence_summary": (
                "First-paper floor remains "
                f"{phase146_gate.get('main_paper_floor', 'unknown')}."
            ),
            "paper_use": "main_text_narrow_floor_only",
            "operator_training_allowed": False,
            "model_training_allowed": phase146_gate.get("phase146_model_training_allowed", False),
            "a100_training_allowed_now": phase146_gate.get(
                "a100_training_allowed_now", False
            ),
            "blocks_neural_operator_success_claim": True,
            "next_action": "preserve paper-one floor without relabeling it as an operator result",
        },
        {
            "route_id": "P152-ROUTE-002",
            "source_phase": "phase149_neural_operator_readiness_gate",
            "artifact": _display_path(phase_inputs["phase149_gate"], root),
            "route_status": phase149_gate.get("status", "missing"),
            "evidence_summary": (
                f"Readiness blockers: {phase149_gate.get('blocker_rows', readiness_blockers)}; "
                "operator training remained closed."
            ),
            "paper_use": "appendix_or_limitations_diagnostic",
            "operator_training_allowed": phase149_gate.get(
                "operator_training_allowed_now", False
            ),
            "model_training_allowed": phase149_gate.get("phase149_model_training_allowed", False),
            "a100_training_allowed_now": phase149_gate.get(
                "a100_training_allowed_now", False
            ),
            "blocks_neural_operator_success_claim": True,
            "next_action": "do not train FNO or neural operators from readiness-only evidence",
        },
        {
            "route_id": "P152-ROUTE-003",
            "source_phase": "phase150_dense_tensorization_inventory_gate",
            "artifact": _display_path(phase_inputs["phase150_gate"], root),
            "route_status": phase150_gate.get("status", "missing"),
            "evidence_summary": (
                f"Present dense sources: {phase150_gate.get('present_source_rows', present_dense_sources)}; "
                f"tensorizable candidates: {phase150_gate.get('tensorizable_candidate_rows', tensorizable_sources)}; "
                f"operator-gap-ready rows: {phase150_gate.get('operator_gap_ready_rows', 0)}."
            ),
            "paper_use": "appendix_or_limitations_diagnostic",
            "operator_training_allowed": phase150_gate.get(
                "operator_training_allowed_now", False
            ),
            "model_training_allowed": phase150_gate.get("phase150_model_training_allowed", False),
            "a100_training_allowed_now": phase150_gate.get(
                "a100_training_allowed_now", False
            ),
            "blocks_neural_operator_success_claim": True,
            "next_action": "require a leakage-safe fixed-grid baseline review before model work",
        },
        {
            "route_id": "P152-ROUTE-004",
            "source_phase": "phase151_fixed_grid_dense_baseline_review",
            "artifact": _display_path(phase_inputs["phase151_gate"], root),
            "route_status": phase151_gate.get("status", "missing"),
            "evidence_summary": (
                f"Split contracts: {phase151_gate.get('split_contract_rows', len(phase151_split_rows))}; "
                f"diagnostic-only splits: {diagnostic_splits}; "
                f"leakage-safe splits: {phase151_gate.get('leakage_safe_source_rows', leakage_safe_splits)}; "
                f"strong-baseline-solved targets: {strong_baseline_solved}; "
                "low-capacity dense design candidates: "
                f"{phase151_gate.get('phase152_low_capacity_dense_design_candidates', 0)}."
            ),
            "paper_use": "appendix_or_limitations_diagnostic",
            "operator_training_allowed": phase151_gate.get(
                "operator_training_allowed_now", False
            ),
            "model_training_allowed": phase151_gate.get("phase151_model_training_allowed", False),
            "a100_training_allowed_now": phase151_gate.get(
                "a100_training_allowed_now", False
            ),
            "blocks_neural_operator_success_claim": True,
            "next_action": "close the neural-operator route unless a new dense target/split source is added",
        },
    ]


def build_claim_boundary_rows(*, root: Path, phase_inputs: dict[str, Path]) -> list[dict[str, Any]]:
    phase151_anchor = _display_path(phase_inputs["phase151_gate"], root)
    return [
        {
            "claim_id": "P152-CLAIM-001",
            "claim_area": "first_paper_positive_floor",
            "claim_status": "allowed_narrow_claim",
            "paper_boundary": (
                "The first paper may continue around fixed-sampling broad12/broad21 "
                "spot_size under broad_process_v1 with seeds 7/1/2."
            ),
            "evidence_anchor": _display_path(phase_inputs["phase146_gate"], root),
            "allowed_final_use": "main_text_core_result",
        },
        {
            "claim_id": "P152-CLAIM-002",
            "claim_area": "neural_operator_or_fno",
            "claim_status": "blocked_success_claim",
            "paper_boundary": (
                "Do not write neural-operator, FNO, operator-learning, or dense-field "
                "operator success. Phase 149-151 close this route as diagnostic."
            ),
            "evidence_anchor": phase151_anchor,
            "allowed_final_use": "appendix_or_limitations_only",
        },
        {
            "claim_id": "P152-CLAIM-003",
            "claim_area": "dense_fixed_grid_targets",
            "claim_status": "diagnostic_only",
            "paper_boundary": (
                "Fixed-grid dense summaries may be cited only to explain why the "
                "operator route is not trained: single-line splits are diagnostic, "
                "and the leakage-safe multiline split is solved by non-neural baselines."
            ),
            "evidence_anchor": phase151_anchor,
            "allowed_final_use": "appendix_or_limitations_only",
        },
        {
            "claim_id": "P152-CLAIM-004",
            "claim_area": "compute_need",
            "claim_status": "blocked_80gb_claim",
            "paper_boundary": (
                "Do not claim A100-SXM4-80GB is needed. No seed-positive route has "
                "hit a measured 40GB memory/runtime bottleneck."
            ),
            "evidence_anchor": phase151_anchor,
            "allowed_final_use": "project_boundary_note",
        },
        {
            "claim_id": "P152-CLAIM-005",
            "claim_area": "overbroad_model_framing",
            "claim_status": "blocked_success_claim",
            "paper_boundary": (
                "Do not write complete GNN-PINN, general process-condition modeling, "
                "density-invariant robustness, source-path/Green success, "
                "microstructure GNN success, CAPL/path-contact success, or "
                "MAM-PhyGNN/FNO success."
            ),
            "evidence_anchor": "task_plan.md; findings.md; phase149-151 gates",
            "allowed_final_use": "claim_guardrail",
        },
    ]


def build_decision_rows(
    *,
    route_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    gate: dict[str, Any],
    root: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    closure_path = _display_path(output_dir / "phase152_neural_operator_route_closure_table.csv", root)
    claim_path = _display_path(output_dir / "phase152_claim_boundary_refresh_table.csv", root)
    return [
        {
            "decision_id": "P152-DECISION-001",
            "decision": "refresh_first_paper_boundary",
            "status": "ready" if gate["first_paper_draft_allowed_now"] else "blocked",
            "rationale": (
                "The positive floor remains narrow and unchanged after the "
                "neural-operator diagnostics."
            ),
            "evidence_anchor": claim_path,
            "next_action": "continue first-paper writing/refinement around the route-guarded spot_size floor",
        },
        {
            "decision_id": "P152-DECISION-002",
            "decision": "close_neural_operator_route",
            "status": "closed_diagnostic"
            if gate["neural_operator_route_closed_as_diagnostic"]
            else "incomplete_review",
            "rationale": (
                "Phase 149-151 did not produce a leakage-safe dense target that "
                "strong baselines leave unsolved."
            ),
            "evidence_anchor": closure_path,
            "next_action": "do not train FNO/neural operators from the current dense candidates",
        },
        {
            "decision_id": "P152-DECISION-003",
            "decision": "next_research_route",
            "status": "fresh_no_training_intake_only",
            "rationale": (
                f"{len(route_rows)} route rows and {len(claim_rows)} claim rows keep "
                "all training and compute escalation locks false."
            ),
            "evidence_anchor": closure_path,
            "next_action": "open a fresh baseline-first source intake only after this closure is recorded",
        },
    ]


def build_gate(
    *,
    phase146_gate: dict[str, Any],
    phase149_gate: dict[str, Any],
    phase150_gate: dict[str, Any],
    phase151_gate: dict[str, Any],
    route_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    operator_training_locked = all(
        _is_false(row.get("operator_training_allowed")) for row in route_rows
    )
    model_training_locked = all(_is_false(row.get("model_training_allowed")) for row in route_rows)
    a100_training_locked = all(
        _is_false(row.get("a100_training_allowed_now")) for row in route_rows
    )
    phase146_ready = _is_true(phase146_gate.get("first_paper_draft_allowed_now"))
    phase149_closed = (
        phase149_gate.get("status")
        == "phase149_neural_operator_readiness_closed_not_ready_for_operator_training"
    )
    phase151_candidates = int(phase151_gate.get("phase152_low_capacity_dense_design_candidates", 0))
    phase151_closed = (
        phase151_gate.get("status")
        == "phase151_fixed_grid_dense_baseline_closed_no_operator_gap"
        and phase151_candidates == 0
    )
    route_closed = (
        phase146_ready
        and phase149_closed
        and phase151_closed
        and operator_training_locked
        and model_training_locked
        and a100_training_locked
    )
    status = (
        "phase152_paper_evidence_refresh_ready_first_paper_narrow_claims_neural_operator_closed"
        if route_closed
        else "phase152_paper_evidence_refresh_incomplete_neural_operator_closure"
    )
    return {
        "status": status,
        "main_paper_floor": phase146_gate.get(
            "main_paper_floor",
            "fixed-sampling broad12/broad21 spot_size under broad_process_v1, seeds 7/1/2",
        ),
        "first_paper_draft_allowed_now": bool(phase146_ready),
        "first_paper_submission_ready": False,
        "latest_operator_diagnostics_complete": bool(phase149_closed and phase151_closed),
        "latest_operator_training_locks_verified": bool(
            operator_training_locked and model_training_locked and a100_training_locked
        ),
        "neural_operator_route_closed_as_diagnostic": bool(route_closed),
        "new_neural_operator_model_claim_ready": False,
        "new_external_model_claim_ready": False,
        "phase149_gate_status": phase149_gate.get("status"),
        "phase150_gate_status": phase150_gate.get("status"),
        "phase151_gate_status": phase151_gate.get("status"),
        "phase152_low_capacity_dense_design_candidates": phase151_candidates,
        "route_closure_rows": len(route_rows),
        "claim_boundary_rows": len(claim_rows),
        "phase152_model_mechanism_allowed": False,
        "phase152_model_training_allowed": False,
        "operator_training_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": (
            "continue first-paper refinement around the route-guarded spot_size floor; "
            "do not write neural-operator/FNO success; open only a fresh no-training "
            "source intake if continuing model discovery"
        ),
    }


def _markdown_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[str]:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(_csv_value(row.get(field, "")) for field in fields) + " |"
        for row in rows
    ]
    return [header, sep, *body]


def build_markdown(
    *,
    gate: dict[str, Any],
    route_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# Phase 152 Paper Evidence Refresh: Neural-Operator Route Closure",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- First-paper draft allowed now: `{_csv_value(gate['first_paper_draft_allowed_now'])}`",
        f"- Neural-operator route closed as diagnostic: `{_csv_value(gate['neural_operator_route_closed_as_diagnostic'])}`",
        f"- New neural-operator model claim ready: `{_csv_value(gate['new_neural_operator_model_claim_ready'])}`",
        f"- Phase 152 model training allowed: `{_csv_value(gate['phase152_model_training_allowed'])}`",
        f"- Operator training allowed now: `{_csv_value(gate['operator_training_allowed_now'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "Phase 149-151 should be cited only as a diagnostic closure: readiness was "
            "blocked, dense tensor candidates needed a fixed-grid split review, and the "
            "final leakage-safe multiline dense target was solved by non-neural strong "
            "baselines. Do not write FNO/neural-operator success from this route."
        ),
        "",
        "## Route Closure Table",
        *_markdown_table(route_rows, ROUTE_FIELDS),
        "",
        "## Claim Boundary Table",
        *_markdown_table(claim_rows, CLAIM_FIELDS),
        "",
        "## Next Decision Table",
        *_markdown_table(decision_rows, DECISION_FIELDS),
        "",
    ]
    return "\n".join(lines)


def build_package(*, root: Path, output_dir: Path, phase_inputs: dict[str, Path]) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    resolved = {
        name: path if path.is_absolute() else root / path for name, path in phase_inputs.items()
    }

    phase146_gate = _read_json(resolved["phase146_gate"])
    phase149_gate = _read_json(resolved["phase149_gate"])
    phase150_gate = _read_json(resolved["phase150_gate"])
    phase151_gate = _read_json(resolved["phase151_gate"])
    phase149_readiness_rows = _read_csv(resolved["phase149_readiness_table"])
    phase150_inventory_rows = _read_csv(resolved["phase150_inventory_table"])
    phase151_review_rows = _read_csv(resolved["phase151_review_table"])
    phase151_split_rows = _read_csv(resolved["phase151_split_table"])

    route_rows = build_route_rows(
        root=root,
        phase146_gate=phase146_gate,
        phase149_gate=phase149_gate,
        phase150_gate=phase150_gate,
        phase151_gate=phase151_gate,
        phase149_readiness_rows=phase149_readiness_rows,
        phase150_inventory_rows=phase150_inventory_rows,
        phase151_review_rows=phase151_review_rows,
        phase151_split_rows=phase151_split_rows,
        phase_inputs=resolved,
    )
    claim_rows = build_claim_boundary_rows(root=root, phase_inputs=resolved)
    gate = build_gate(
        phase146_gate=phase146_gate,
        phase149_gate=phase149_gate,
        phase150_gate=phase150_gate,
        phase151_gate=phase151_gate,
        route_rows=route_rows,
        claim_rows=claim_rows,
    )
    decision_rows = build_decision_rows(
        route_rows=route_rows,
        claim_rows=claim_rows,
        gate=gate,
        root=root,
        output_dir=output_dir,
    )

    closure_path = output_dir / "phase152_neural_operator_route_closure_table.csv"
    claim_path = output_dir / "phase152_claim_boundary_refresh_table.csv"
    decision_path = output_dir / "phase152_next_decision_table.csv"
    gate_path = output_dir / "phase152_paper_evidence_refresh_gate.json"
    markdown_path = output_dir / "phase152_paper_evidence_refresh.md"
    manifest_path = output_dir / "phase152_paper_evidence_refresh_manifest.json"

    _write_csv(closure_path, route_rows, ROUTE_FIELDS)
    _write_csv(claim_path, claim_rows, CLAIM_FIELDS)
    _write_csv(decision_path, decision_rows, DECISION_FIELDS)
    _write_json(gate_path, gate)
    markdown_path.write_text(
        build_markdown(
            gate=gate,
            route_rows=route_rows,
            claim_rows=claim_rows,
            decision_rows=decision_rows,
        ),
        encoding="utf-8",
    )

    manifest = {
        "phase": 152,
        "description": "paper evidence refresh and neural-operator route closure",
        "inputs": {name: _display_path(path, root) for name, path in resolved.items()},
        "outputs": {
            "route_closure_table": _display_path(closure_path, root),
            "claim_boundary_table": _display_path(claim_path, root),
            "next_decision_table": _display_path(decision_path, root),
            "gate": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "route_closure_rows": len(route_rows),
            "claim_boundary_rows": len(claim_rows),
            "decision_rows": len(decision_rows),
            "blocked_neural_operator_claim_rows": sum(
                1 for row in claim_rows if row["claim_status"] == "blocked_success_claim"
            ),
            "operator_training_allowed_route_rows": sum(
                1 for row in route_rows if not _is_false(row["operator_training_allowed"])
            ),
            "model_training_allowed_route_rows": sum(
                1 for row in route_rows if not _is_false(row["model_training_allowed"])
            ),
            "a100_training_allowed_route_rows": sum(
                1 for row in route_rows if not _is_false(row["a100_training_allowed_now"])
            ),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    for name, default in PHASE_INPUTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase_inputs = {name: getattr(args, name) for name in PHASE_INPUTS}
    manifest = build_package(
        root=args.root,
        output_dir=args.output_dir,
        phase_inputs=phase_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
