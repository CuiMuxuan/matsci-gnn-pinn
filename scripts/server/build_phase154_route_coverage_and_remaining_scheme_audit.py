#!/usr/bin/env python3
"""Build Phase 154 route coverage and remaining-scheme audit package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("docs/results/phase154_route_coverage_and_remaining_scheme_audit")

PHASE_INPUTS = {
    "phase153_gate": Path(
        "docs/results/phase153_first_paper_contribution_refinement/"
        "phase153_first_paper_contribution_refinement_gate.json"
    ),
    "phase153_contribution_table": Path(
        "docs/results/phase153_first_paper_contribution_refinement/"
        "phase153_contribution_refinement_table.csv"
    ),
    "phase153_phrasing_guard_table": Path(
        "docs/results/phase153_first_paper_contribution_refinement/"
        "phase153_claim_phrasing_guard_table.csv"
    ),
    "phase153_open_gap_table": Path(
        "docs/results/phase153_first_paper_contribution_refinement/"
        "phase153_open_gap_table.csv"
    ),
    "phase152_route_closure_table": Path(
        "docs/results/phase152_paper_evidence_neural_operator_route_closure/"
        "phase152_neural_operator_route_closure_table.csv"
    ),
    "phase116_claim_status_table": Path(
        "docs/results/phase116_paper_evidence_consolidation/"
        "phase116_manuscript_claim_status_table.csv"
    ),
}

ROUTE_FIELDS = (
    "route_id",
    "route_family",
    "current_status",
    "verification_scope",
    "evidence_anchor",
    "currently_executable_resolved",
    "future_scheme_space_exhausted",
    "model_training_allowed_now",
    "a100_training_allowed_now",
    "missing_precondition",
    "next_action",
)

REMAINING_FIELDS = (
    "remaining_id",
    "scheme_or_need",
    "status",
    "why_not_done_now",
    "required_precondition",
    "can_start_without_new_input",
    "recommended_next_gate",
)

DECISION_FIELDS = (
    "decision_id",
    "question",
    "answer",
    "evidence_anchor",
    "project_action",
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
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
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


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _is_false(value: Any) -> bool:
    if value is False or value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "0", "false", "none", "no"}
    return False


def _count_status(rows: list[dict[str, str]], column: str, status: str) -> int:
    return sum(1 for row in rows if row.get(column) == status)


def build_route_rows(*, root: Path, phase_inputs: dict[str, Path]) -> list[dict[str, Any]]:
    phase153_anchor = _display_path(phase_inputs["phase153_gate"], root)
    phase152_anchor = _display_path(phase_inputs["phase152_route_closure_table"], root)
    claim_anchor = _display_path(phase_inputs["phase116_claim_status_table"], root)
    phrasing_anchor = _display_path(phase_inputs["phase153_phrasing_guard_table"], root)
    return [
        {
            "route_id": "P154-ROUTE-001",
            "route_family": "first_paper_main_floor",
            "current_status": "verified_positive_narrow_scope",
            "verification_scope": "fixed-sampling broad12/broad21 spot_size under broad_process_v1, seeds 7/1/2",
            "evidence_anchor": phase153_anchor,
            "currently_executable_resolved": True,
            "future_scheme_space_exhausted": False,
            "model_training_allowed_now": False,
            "a100_training_allowed_now": False,
            "missing_precondition": "none for first-paper draft; venue/literature still needed for submission",
            "next_action": "write/refine first paper around the narrow floor",
        },
        {
            "route_id": "P154-ROUTE-002",
            "route_family": "neural_operator_fno_dense_field",
            "current_status": "closed_diagnostic_no_operator_gap",
            "verification_scope": "Phase 149 readiness, Phase 150 dense inventory, Phase 151 fixed-grid baseline review",
            "evidence_anchor": phase152_anchor,
            "currently_executable_resolved": True,
            "future_scheme_space_exhausted": False,
            "model_training_allowed_now": False,
            "a100_training_allowed_now": False,
            "missing_precondition": "new leakage-safe dense target/split that strong baselines do not solve",
            "next_action": "do not train FNO/neural operators from current dense candidates",
        },
        {
            "route_id": "P154-ROUTE-003",
            "route_family": "registered_source_path_green_capl",
            "current_status": "closed_diagnostic_under_current_registration",
            "verification_scope": "source-path, Green proxy, registered Layer Camera, G-code, and path-contact audits",
            "evidence_anchor": phrasing_anchor,
            "currently_executable_resolved": True,
            "future_scheme_space_exhausted": False,
            "model_training_allowed_now": False,
            "a100_training_allowed_now": False,
            "missing_precondition": "stronger camera-to-galvo registration or new source-target target with guarded gap",
            "next_action": "do not tune the closed scalar proxy/path-contact branch",
        },
        {
            "route_id": "P154-ROUTE-004",
            "route_family": "microstructure_gnn_or_image_encoder",
            "current_status": "closed_diagnostic_alignment_or_stability_limit",
            "verification_scope": "real-micro, region, patch embedding, and manuscript claim-boundary packages",
            "evidence_anchor": claim_anchor,
            "currently_executable_resolved": True,
            "future_scheme_space_exhausted": False,
            "model_training_allowed_now": False,
            "a100_training_allowed_now": False,
            "missing_precondition": "physically stronger microstructure/thermal alignment or new benchmark source",
            "next_action": "do not claim microstructure GNN success from current evidence",
        },
        {
            "route_id": "P154-ROUTE-005",
            "route_family": "external_baseline_first_sources",
            "current_status": "sampled_sources_verified_or_closed_as_diagnostics",
            "verification_scope": "Battery, Matbench, MPEA, glass/is-metal/perovskite-style baseline-first intakes",
            "evidence_anchor": claim_anchor,
            "currently_executable_resolved": True,
            "future_scheme_space_exhausted": False,
            "model_training_allowed_now": False,
            "a100_training_allowed_now": False,
            "missing_precondition": "fresh public source with leakage-safe splits and strong-baseline-visible gap",
            "next_action": "new baseline-first source intake is allowed only as a new gated branch",
        },
        {
            "route_id": "P154-ROUTE-006",
            "route_family": "first_paper_submission",
            "current_status": "draft_allowed_submission_not_ready",
            "verification_scope": "Phase 153 contribution package plus open venue/literature gaps",
            "evidence_anchor": phase153_anchor,
            "currently_executable_resolved": False,
            "future_scheme_space_exhausted": False,
            "model_training_allowed_now": False,
            "a100_training_allowed_now": False,
            "missing_precondition": "target venue/author guide and benchmark literature verification",
            "next_action": "resolve writing evidence gaps before submission polish",
        },
        {
            "route_id": "P154-ROUTE-007",
            "route_family": "large_gpu_training_or_80gb",
            "current_status": "blocked_no_measured_need",
            "verification_scope": "all active gates keep A100/80GB locks false",
            "evidence_anchor": phase153_anchor,
            "currently_executable_resolved": True,
            "future_scheme_space_exhausted": False,
            "model_training_allowed_now": False,
            "a100_training_allowed_now": False,
            "missing_precondition": "seed-positive branch with measured 40GB memory/runtime bottleneck",
            "next_action": "do not request A100-SXM4-80GB",
        },
    ]


def build_remaining_rows() -> list[dict[str, Any]]:
    return [
        {
            "remaining_id": "P154-REMAIN-001",
            "scheme_or_need": "target venue / author guide / benchmark papers",
            "status": "open_non_model_blocker",
            "why_not_done_now": "This requires user-selected venue or verified benchmark-paper set.",
            "required_precondition": "target venue, author guide, or 3-10 accepted benchmark papers",
            "can_start_without_new_input": False,
            "recommended_next_gate": "literature/venue verification and manuscript alignment package",
        },
        {
            "remaining_id": "P154-REMAIN-002",
            "scheme_or_need": "fresh leakage-safe dense operator target",
            "status": "future_preconditioned_route",
            "why_not_done_now": "Current fixed-grid dense candidates either lack leakage-safe splits or are solved by strong baselines.",
            "required_precondition": "new dense target/split source with a strong-baseline-visible gap",
            "can_start_without_new_input": False,
            "recommended_next_gate": "no-training dense target intake and baseline-gap audit",
        },
        {
            "remaining_id": "P154-REMAIN-003",
            "scheme_or_need": "stronger scan-path/camera registration",
            "status": "future_preconditioned_route",
            "why_not_done_now": "Current source-path/CAPL/path-contact diagnostics did not clear route guards.",
            "required_precondition": "defensible camera-pixel to galvo-mm registration or new registered target",
            "can_start_without_new_input": False,
            "recommended_next_gate": "registered target/source intake before any mechanism",
        },
        {
            "remaining_id": "P154-REMAIN-004",
            "scheme_or_need": "fresh baseline-first source intake",
            "status": "allowed_if_opened_as_new_branch",
            "why_not_done_now": "Existing source branches are closed or diagnostic under current gates.",
            "required_precondition": "small public source or server-local source with manifest, leakage-safe split, and strong baselines",
            "can_start_without_new_input": True,
            "recommended_next_gate": "baseline-first source intake with all training locks false",
        },
        {
            "remaining_id": "P154-REMAIN-005",
            "scheme_or_need": "large GPU / A100-SXM4-80GB training",
            "status": "blocked",
            "why_not_done_now": "No seed-positive branch has a measured 40GB bottleneck.",
            "required_precondition": "passed seed-positive gate plus measured A800/A100-40GB memory/runtime failure",
            "can_start_without_new_input": False,
            "recommended_next_gate": "none until a positive branch proves compute need",
        },
    ]


def build_decision_rows(*, route_rows: list[dict[str, Any]], remaining_rows: list[dict[str, Any]], root: Path, output_dir: Path) -> list[dict[str, Any]]:
    route_anchor = _display_path(output_dir / "phase154_route_coverage_table.csv", root)
    remaining_anchor = _display_path(output_dir / "phase154_remaining_scheme_table.csv", root)
    current_unresolved = [
        row for row in route_rows
        if not _is_true(row["currently_executable_resolved"])
        and row["route_family"] != "first_paper_submission"
    ]
    return [
        {
            "decision_id": "P154-DECISION-001",
            "question": "Are all currently executable model/research branches verified?",
            "answer": "yes_under_current_data_and_gate_conditions" if not current_unresolved else "no",
            "evidence_anchor": route_anchor,
            "project_action": "do not reopen closed branches without a new baseline-first gate",
        },
        {
            "decision_id": "P154-DECISION-002",
            "question": "Are all possible future schemes exhausted?",
            "answer": "no_future_preconditioned_routes_remain",
            "evidence_anchor": remaining_anchor,
            "project_action": "treat future schemes as requiring new data, registration, venue evidence, or fresh source intake",
        },
        {
            "decision_id": "P154-DECISION-003",
            "question": "Can the first paper proceed?",
            "answer": "draft_yes_submission_not_ready",
            "evidence_anchor": _display_path(output_dir / "phase154_route_coverage_gate.json", root),
            "project_action": "continue writing around the narrow floor and resolve venue/literature blockers",
        },
        {
            "decision_id": "P154-DECISION-004",
            "question": "Should A100-SXM4-80GB be requested now?",
            "answer": "no",
            "evidence_anchor": route_anchor,
            "project_action": "keep A800/A100-40GB as sufficient until a measured positive-branch bottleneck exists",
        },
    ]


def build_gate(
    *,
    phase153_gate: dict[str, Any],
    route_rows: list[dict[str, Any]],
    remaining_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase153_ready = (
        phase153_gate.get("status")
        == "phase153_first_paper_contribution_refinement_ready_narrow_claims"
    )
    locks_false = (
        _is_false(phase153_gate.get("phase153_model_training_allowed"))
        and _is_false(phase153_gate.get("operator_training_allowed_now"))
        and _is_false(phase153_gate.get("a100_training_allowed_now"))
        and _is_false(phase153_gate.get("a100_80gb_request_now"))
    )
    model_route_rows = [
        row for row in route_rows
        if row["route_family"] not in {"first_paper_submission"}
    ]
    current_model_routes_resolved = all(
        _is_true(row["currently_executable_resolved"]) for row in model_route_rows
    )
    future_preconditioned_rows = [
        row for row in remaining_rows
        if row["status"] in {"future_preconditioned_route", "allowed_if_opened_as_new_branch"}
    ]
    status = (
        "phase154_route_coverage_audit_ready_current_routes_verified_future_not_exhausted"
        if phase153_ready and locks_false and current_model_routes_resolved
        else "phase154_route_coverage_audit_incomplete"
    )
    return {
        "status": status,
        "route_coverage_rows": len(route_rows),
        "remaining_scheme_rows": len(remaining_rows),
        "decision_rows": len(decision_rows),
        "currently_executable_model_routes_verified": bool(current_model_routes_resolved),
        "all_possible_future_schemes_exhausted": False,
        "future_preconditioned_route_rows": len(future_preconditioned_rows),
        "first_paper_draft_allowed_now": bool(
            _is_true(phase153_gate.get("first_paper_draft_allowed_now"))
        ),
        "first_paper_submission_ready": False,
        "new_model_claim_ready": False,
        "phase154_model_mechanism_allowed": False,
        "phase154_model_training_allowed": False,
        "operator_training_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": (
            "do not treat future scheme space as exhausted; either resolve first-paper "
            "venue/literature gaps or open a fresh no-training baseline-first source intake"
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


def build_markdown(*, gate: dict[str, Any], route_rows: list[dict[str, Any]], remaining_rows: list[dict[str, Any]], decision_rows: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        "# Phase 154 Route Coverage and Remaining-Scheme Audit",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Currently executable model routes verified: `{_csv_value(gate['currently_executable_model_routes_verified'])}`",
        f"- All possible future schemes exhausted: `{_csv_value(gate['all_possible_future_schemes_exhausted'])}`",
        f"- Future/preconditioned route rows: `{gate['future_preconditioned_route_rows']}`",
        f"- First-paper draft allowed now: `{_csv_value(gate['first_paper_draft_allowed_now'])}`",
        f"- First-paper submission ready: `{_csv_value(gate['first_paper_submission_ready'])}`",
        f"- Phase 154 model training allowed: `{_csv_value(gate['phase154_model_training_allowed'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "The audit answers the user's route-coverage question: all currently opened "
            "and executable model/research routes have been verified or closed under the "
            "current data and gate conditions, but the future scheme space is not exhausted. "
            "Future work requires a new source, new registration, a new dense target/split, "
            "or venue/literature input."
        ),
        "",
        "## Route Coverage",
        *_markdown_table(route_rows, ROUTE_FIELDS),
        "",
        "## Remaining Schemes",
        *_markdown_table(remaining_rows, REMAINING_FIELDS),
        "",
        "## Decisions",
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
    phase153_gate = _read_json(resolved["phase153_gate"])
    contribution_rows = _read_csv(resolved["phase153_contribution_table"])
    phrasing_rows = _read_csv(resolved["phase153_phrasing_guard_table"])
    open_gap_rows = _read_csv(resolved["phase153_open_gap_table"])
    route_closure_rows = _read_csv(resolved["phase152_route_closure_table"])
    claim_status_rows = _read_csv(resolved["phase116_claim_status_table"])

    route_rows = build_route_rows(root=root, phase_inputs=resolved)
    remaining_rows = build_remaining_rows()
    decision_rows = build_decision_rows(
        route_rows=route_rows, remaining_rows=remaining_rows, root=root, output_dir=output_dir
    )
    gate = build_gate(
        phase153_gate=phase153_gate,
        route_rows=route_rows,
        remaining_rows=remaining_rows,
        decision_rows=decision_rows,
    )

    route_path = output_dir / "phase154_route_coverage_table.csv"
    remaining_path = output_dir / "phase154_remaining_scheme_table.csv"
    decision_path = output_dir / "phase154_route_coverage_decision_table.csv"
    gate_path = output_dir / "phase154_route_coverage_gate.json"
    markdown_path = output_dir / "phase154_route_coverage_and_remaining_scheme_audit.md"
    manifest_path = output_dir / "phase154_route_coverage_manifest.json"

    _write_csv(route_path, route_rows, ROUTE_FIELDS)
    _write_csv(remaining_path, remaining_rows, REMAINING_FIELDS)
    _write_csv(decision_path, decision_rows, DECISION_FIELDS)
    _write_json(gate_path, gate)
    with markdown_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            build_markdown(
                gate=gate,
                route_rows=route_rows,
                remaining_rows=remaining_rows,
                decision_rows=decision_rows,
            )
        )

    manifest = {
        "phase": 154,
        "description": "route coverage and remaining feasible-scheme audit",
        "inputs": {name: _display_path(path, root) for name, path in resolved.items()},
        "input_counts": {
            "phase153_contribution_rows": len(contribution_rows),
            "phase153_phrasing_guard_rows": len(phrasing_rows),
            "phase153_open_gap_rows": len(open_gap_rows),
            "phase152_route_closure_rows": len(route_closure_rows),
            "phase116_claim_status_rows": len(claim_status_rows),
        },
        "outputs": {
            "route_coverage_table": _display_path(route_path, root),
            "remaining_scheme_table": _display_path(remaining_path, root),
            "decision_table": _display_path(decision_path, root),
            "gate": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "route_coverage_rows": len(route_rows),
            "remaining_scheme_rows": len(remaining_rows),
            "decision_rows": len(decision_rows),
            "future_preconditioned_route_rows": gate["future_preconditioned_route_rows"],
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
    manifest = build_package(root=args.root, output_dir=args.output_dir, phase_inputs=phase_inputs)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
