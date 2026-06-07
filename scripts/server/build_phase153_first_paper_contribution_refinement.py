#!/usr/bin/env python3
"""Build Phase 153 first-paper contribution and claim-language refinement package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("docs/results/phase153_first_paper_contribution_refinement")

PHASE_INPUTS = {
    "phase60_main_table": Path(
        "docs/results/phase60_manuscript_evidence_package/"
        "phase60_main_spot_size_seed_positive_table.csv"
    ),
    "phase60_route_guard_table": Path(
        "docs/results/phase60_manuscript_evidence_package/"
        "phase60_route_guard_boundary_table.csv"
    ),
    "phase88_claim_lock_table": Path(
        "docs/results/phase88_fallback_manuscript_finalization/"
        "phase88_claim_lock_table.csv"
    ),
    "phase116_positive_floor_table": Path(
        "docs/results/phase116_paper_evidence_consolidation/"
        "phase116_positive_floor_table.csv"
    ),
    "phase116_claim_status_table": Path(
        "docs/results/phase116_paper_evidence_consolidation/"
        "phase116_manuscript_claim_status_table.csv"
    ),
    "phase152_gate": Path(
        "docs/results/phase152_paper_evidence_neural_operator_route_closure/"
        "phase152_paper_evidence_refresh_gate.json"
    ),
    "phase152_claim_boundary_table": Path(
        "docs/results/phase152_paper_evidence_neural_operator_route_closure/"
        "phase152_claim_boundary_refresh_table.csv"
    ),
}

CONTRIBUTION_FIELDS = (
    "contribution_id",
    "contribution_title",
    "writeable_claim",
    "evidence_anchor",
    "main_text_use",
    "scope_guard",
    "novelty_boundary",
    "risk_if_overwritten",
)

SECTION_FIELDS = (
    "section_id",
    "section_name",
    "section_purpose",
    "primary_claim",
    "evidence_anchor",
    "must_include",
    "must_not_include",
)

PHRASING_FIELDS = (
    "guard_id",
    "unsafe_or_overbroad_phrase",
    "paper_safe_replacement",
    "reason",
    "evidence_anchor",
)

OPEN_GAP_FIELDS = (
    "gap_id",
    "gap_type",
    "status",
    "why_it_matters",
    "required_resolution",
    "blocks_submission",
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


def _metric_summary(rows: list[dict[str, str]]) -> str:
    pieces: list[str] = []
    for dataset in ("broad12", "broad21"):
        dataset_rows = [row for row in rows if row.get("dataset") == dataset]
        if not dataset_rows:
            continue
        metrics = []
        for row in dataset_rows:
            metrics.append(
                f"{row.get('metric')}: {row.get('broad_process_v1_mean')} "
                f"vs strong {row.get('best_strong_baseline')}"
            )
        pieces.append(f"{dataset} ({'; '.join(metrics)})")
    return " | ".join(pieces)


def build_contribution_rows(
    *,
    root: Path,
    phase60_main_table: Path,
    phase60_route_guard_table: Path,
    phase88_claim_lock_table: Path,
    phase116_positive_floor_table: Path,
    phase152_claim_boundary_table: Path,
    main_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    metric_summary = _metric_summary(main_rows)
    return [
        {
            "contribution_id": "P153-CONTRIB-001",
            "contribution_title": "Route-guarded process-conditioned Macro PINN floor",
            "writeable_claim": (
                "A conservative route-guarded process-conditioned Macro PINN improves "
                "fixed-sampling broad12/broad21 spot_size transfer over strong baselines "
                "across three seeds."
            ),
            "evidence_anchor": _display_path(phase60_main_table, root),
            "main_text_use": "primary experimental contribution",
            "scope_guard": "fixed-sampling broad12/broad21 spot_size only; seeds 7/1/2",
            "novelty_boundary": metric_summary,
            "risk_if_overwritten": "Overclaiming as universal process modeling would contradict route-guard diagnostics.",
        },
        {
            "contribution_id": "P153-CONTRIB-002",
            "contribution_title": "Validation-only route governance and strong-baseline discipline",
            "writeable_claim": (
                "The study contributes an auditable route-governance protocol that uses "
                "validation-only route selection, strong non-neural baselines, seed checks, "
                "and explicit diagnostic closure before model promotion."
            ),
            "evidence_anchor": _display_path(phase88_claim_lock_table, root),
            "main_text_use": "methods and evaluation contribution",
            "scope_guard": "protocol contribution; not a claim that every route improves performance",
            "novelty_boundary": "Use as reproducibility/governance framing tied to recorded artifacts.",
            "risk_if_overwritten": "Removing the guard makes negative branches look like failed tuning rather than controlled evidence.",
        },
        {
            "contribution_id": "P153-CONTRIB-003",
            "contribution_title": "Boundary-aware process-axis evidence map",
            "writeable_claim": (
                "The paper separates the spot_size positive branch from route-guard-only "
                "or fallback behavior on line, laser_power, scan_speed, and full process splits."
            ),
            "evidence_anchor": _display_path(phase60_route_guard_table, root),
            "main_text_use": "results boundary and limitations",
            "scope_guard": "do not claim all process axes beat strong baselines",
            "novelty_boundary": "Boundary map clarifies where process conditioning is useful versus guarded.",
            "risk_if_overwritten": "Axis-sensitive evidence would be flattened into an unsupported generalization claim.",
        },
        {
            "contribution_id": "P153-CONTRIB-004",
            "contribution_title": "Closed-branch diagnostic ledger including neural operators",
            "writeable_claim": (
                "The manuscript can report a diagnostic ledger showing why NIST AMMT, "
                "CAPL/path-contact, microstructure, MPEA, and neural-operator routes were "
                "not promoted to main-text model claims."
            ),
            "evidence_anchor": _display_path(phase152_claim_boundary_table, root),
            "main_text_use": "appendix and limitations contribution",
            "scope_guard": "diagnostic closure only; no success claim for closed branches",
            "novelty_boundary": "Contribution is transparent claim governance, not a new neural-operator architecture.",
            "risk_if_overwritten": "Closed diagnostics could be misrepresented as model innovations.",
        },
        {
            "contribution_id": "P153-CONTRIB-005",
            "contribution_title": "Paper-ready evidence contract",
            "writeable_claim": (
                "The first paper has a machine-readable evidence contract connecting main "
                "claims, claim guards, artifact paths, and remaining submission blockers."
            ),
            "evidence_anchor": _display_path(phase116_positive_floor_table, root),
            "main_text_use": "reproducibility and artifact availability framing",
            "scope_guard": "submission readiness still requires venue/literature alignment",
            "novelty_boundary": "Engineering reproducibility support; not an added experimental result.",
            "risk_if_overwritten": "Paper could drift from evidence-backed claims into unsupported framing.",
        },
    ]


def build_section_rows(*, root: Path, phase_inputs: dict[str, Path]) -> list[dict[str, Any]]:
    main_anchor = _display_path(phase_inputs["phase60_main_table"], root)
    guard_anchor = _display_path(phase_inputs["phase60_route_guard_table"], root)
    closure_anchor = _display_path(phase_inputs["phase152_claim_boundary_table"], root)
    return [
        {
            "section_id": "P153-SEC-001",
            "section_name": "Introduction",
            "section_purpose": "Frame the problem as evidence-controlled process-conditioned thermal modeling.",
            "primary_claim": "The work targets reliable route-guarded process transfer, not universal AM digital twins.",
            "evidence_anchor": main_anchor,
            "must_include": "fixed-sampling broad12/broad21 spot_size scope and strong-baseline requirement",
            "must_not_include": "complete GNN-PINN, universal process modeling, FNO success",
        },
        {
            "section_id": "P153-SEC-002",
            "section_name": "Method",
            "section_purpose": "Describe Macro PINN, process-route selection, and validation-only governance.",
            "primary_claim": "broad_process_v1 is a conservative route guard with explicit fallback behavior.",
            "evidence_anchor": guard_anchor,
            "must_include": "route table, selection policy, train/validation/test separation",
            "must_not_include": "trainable mixture-of-experts success or unverified GNN/microstructure mechanism",
        },
        {
            "section_id": "P153-SEC-003",
            "section_name": "Results",
            "section_purpose": "Lead with the seed-robust spot_size floor and separate boundary axes.",
            "primary_claim": "spot_size is the only current process-conditioned strong-baseline positive main result.",
            "evidence_anchor": main_anchor,
            "must_include": "three metrics for broad12 and broad21; seeds 7/1/2",
            "must_not_include": "density-invariant robustness or all-axis superiority",
        },
        {
            "section_id": "P153-SEC-004",
            "section_name": "Limitations and Appendix",
            "section_purpose": "Record negative diagnostics and explain why newer routes were not promoted.",
            "primary_claim": "Closed branches are useful boundary evidence, not model success.",
            "evidence_anchor": closure_anchor,
            "must_include": "neural-operator closure, CAPL/path-contact closure, MPEA and microstructure diagnostics",
            "must_not_include": "operator training, dense-field operator success, or 80GB necessity",
        },
        {
            "section_id": "P153-SEC-005",
            "section_name": "Conclusion",
            "section_purpose": "Restate the narrow positive result and the route-governance contribution.",
            "primary_claim": "The contribution is a guarded, reproducible first step with explicit boundaries.",
            "evidence_anchor": _display_path(phase_inputs["phase88_claim_lock_table"], root),
            "must_include": "venue/literature gaps remain before submission-ready claims",
            "must_not_include": "future-work diagnostics as completed contributions",
        },
    ]


def build_phrasing_rows(*, root: Path, phase_inputs: dict[str, Path]) -> list[dict[str, Any]]:
    main_anchor = _display_path(phase_inputs["phase60_main_table"], root)
    closure_anchor = _display_path(phase_inputs["phase152_claim_boundary_table"], root)
    return [
        {
            "guard_id": "P153-PHRASE-001",
            "unsafe_or_overbroad_phrase": "complete GNN-PINN framework",
            "paper_safe_replacement": "route-guarded process-conditioned Macro PINN prototype",
            "reason": "The stable evidence is a Macro PINN route floor, not a complete GNN-PINN system.",
            "evidence_anchor": main_anchor,
        },
        {
            "guard_id": "P153-PHRASE-002",
            "unsafe_or_overbroad_phrase": "general process-condition modeling",
            "paper_safe_replacement": "fixed-sampling spot_size transfer under broad_process_v1",
            "reason": "Only spot_size is main-text strong-baseline positive; other axes are boundary evidence.",
            "evidence_anchor": main_anchor,
        },
        {
            "guard_id": "P153-PHRASE-003",
            "unsafe_or_overbroad_phrase": "density-invariant robustness",
            "paper_safe_replacement": "fixed-sampling seed-robust transfer with density stress reported as a limitation",
            "reason": "Alternate-density stress remains a boundary, not a positive robustness result.",
            "evidence_anchor": _display_path(phase_inputs["phase88_claim_lock_table"], root),
        },
        {
            "guard_id": "P153-PHRASE-004",
            "unsafe_or_overbroad_phrase": "neural-operator/FNO success",
            "paper_safe_replacement": "neural-operator route closed as diagnostic after fixed-grid baseline review",
            "reason": "Phase 149-151 found no strong-baseline-visible operator modeling gap.",
            "evidence_anchor": closure_anchor,
        },
        {
            "guard_id": "P153-PHRASE-005",
            "unsafe_or_overbroad_phrase": "source-path/Green/CAPL/path-contact success",
            "paper_safe_replacement": "source-path and path-contact routes are appendix diagnostics under current guards",
            "reason": "Path/contact and source-kernel routes did not clear their baseline/registration gates.",
            "evidence_anchor": closure_anchor,
        },
        {
            "guard_id": "P153-PHRASE-006",
            "unsafe_or_overbroad_phrase": "microstructure GNN success",
            "paper_safe_replacement": "microstructure routes remain diagnostic due unstable alignment/performance",
            "reason": "Existing microstructure evidence is not stable enough for a model contribution claim.",
            "evidence_anchor": _display_path(phase_inputs["phase116_claim_status_table"], root),
        },
        {
            "guard_id": "P153-PHRASE-007",
            "unsafe_or_overbroad_phrase": "A100-SXM4-80GB is required",
            "paper_safe_replacement": "A800/A100-40GB remains sufficient for the current gates; 80GB is unproven",
            "reason": "No seed-positive branch has hit a measured 40GB memory/runtime bottleneck.",
            "evidence_anchor": closure_anchor,
        },
    ]


def build_open_gap_rows(*, phase152_gate: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "gap_id": "P153-GAP-001",
            "gap_type": "target_venue",
            "status": "open",
            "why_it_matters": "Final section ordering, citation density, and submission language depend on venue.",
            "required_resolution": "User supplies target venue/author guide or accepted benchmark papers.",
            "blocks_submission": True,
        },
        {
            "gap_id": "P153-GAP-002",
            "gap_type": "literature_verification",
            "status": "open",
            "why_it_matters": "Novelty statements need verified benchmark papers and citation proximity.",
            "required_resolution": "Run literature verification before final Introduction/Related Work claims.",
            "blocks_submission": True,
        },
        {
            "gap_id": "P153-GAP-003",
            "gap_type": "neural_operator_branch",
            "status": "closed_diagnostic",
            "why_it_matters": "Prevents FNO/operator claims from entering the manuscript.",
            "required_resolution": phase152_gate.get("next_action", "keep route closed"),
            "blocks_submission": False,
        },
        {
            "gap_id": "P153-GAP-004",
            "gap_type": "additional_model_training",
            "status": "closed_for_first_paper",
            "why_it_matters": "New training would blur the frozen first-paper floor unless a fresh gate passes.",
            "required_resolution": "Open only a fresh no-training baseline-first intake before any new training.",
            "blocks_submission": False,
        },
    ]


def build_gate(
    *,
    contribution_rows: list[dict[str, Any]],
    section_rows: list[dict[str, Any]],
    phrasing_rows: list[dict[str, Any]],
    open_gap_rows: list[dict[str, Any]],
    phase152_gate: dict[str, Any],
) -> dict[str, Any]:
    phase152_closed = (
        phase152_gate.get("status")
        == "phase152_paper_evidence_refresh_ready_first_paper_narrow_claims_neural_operator_closed"
    )
    first_paper_ready = _is_true(phase152_gate.get("first_paper_draft_allowed_now"))
    locks_false = (
        _is_false(phase152_gate.get("phase152_model_training_allowed"))
        and _is_false(phase152_gate.get("operator_training_allowed_now"))
        and _is_false(phase152_gate.get("a100_training_allowed_now"))
        and _is_false(phase152_gate.get("a100_80gb_request_now"))
    )
    submission_blockers = sum(1 for row in open_gap_rows if _is_true(row["blocks_submission"]))
    ready = phase152_closed and first_paper_ready and locks_false
    return {
        "status": (
            "phase153_first_paper_contribution_refinement_ready_narrow_claims"
            if ready
            else "phase153_first_paper_contribution_refinement_incomplete"
        ),
        "contribution_rows": len(contribution_rows),
        "section_rows": len(section_rows),
        "phrasing_guard_rows": len(phrasing_rows),
        "open_gap_rows": len(open_gap_rows),
        "submission_blocker_rows": submission_blockers,
        "first_paper_draft_allowed_now": bool(first_paper_ready),
        "first_paper_submission_ready": submission_blockers == 0,
        "main_paper_floor": phase152_gate.get(
            "main_paper_floor",
            "fixed-sampling broad12/broad21 spot_size under broad_process_v1, seeds 7/1/2",
        ),
        "contribution_refinement_ready": bool(ready),
        "new_model_claim_ready": False,
        "phase153_model_mechanism_allowed": False,
        "phase153_model_training_allowed": False,
        "operator_training_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": (
            "use this package to draft/refine the first-paper contribution section; "
            "resolve venue/literature gaps before submission polish"
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
    contribution_rows: list[dict[str, Any]],
    section_rows: list[dict[str, Any]],
    phrasing_rows: list[dict[str, Any]],
    open_gap_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# Phase 153 First-Paper Contribution Refinement",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Contribution refinement ready: `{_csv_value(gate['contribution_refinement_ready'])}`",
        f"- First-paper draft allowed now: `{_csv_value(gate['first_paper_draft_allowed_now'])}`",
        f"- First-paper submission ready: `{_csv_value(gate['first_paper_submission_ready'])}`",
        f"- New model claim ready: `{_csv_value(gate['new_model_claim_ready'])}`",
        f"- Phase 153 model training allowed: `{_csv_value(gate['phase153_model_training_allowed'])}`",
        f"- Operator training allowed now: `{_csv_value(gate['operator_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Writing Boundary",
        (
            "The first paper should be framed as a narrow, evidence-controlled contribution: "
            "route-guarded process-conditioned Macro PINN evidence for fixed-sampling "
            "broad12/broad21 spot_size, plus a reproducible claim-governance protocol. "
            "Closed branches remain diagnostic appendices."
        ),
        "",
        "## Contribution Table",
        *_markdown_table(contribution_rows, CONTRIBUTION_FIELDS),
        "",
        "## Section Map",
        *_markdown_table(section_rows, SECTION_FIELDS),
        "",
        "## Phrasing Guards",
        *_markdown_table(phrasing_rows, PHRASING_FIELDS),
        "",
        "## Open Gaps",
        *_markdown_table(open_gap_rows, OPEN_GAP_FIELDS),
        "",
    ]
    return "\n".join(lines)


def build_package(*, root: Path, output_dir: Path, phase_inputs: dict[str, Path]) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    resolved = {
        name: path if path.is_absolute() else root / path for name, path in phase_inputs.items()
    }

    main_rows = _read_csv(resolved["phase60_main_table"])
    route_guard_rows = _read_csv(resolved["phase60_route_guard_table"])
    claim_lock_rows = _read_csv(resolved["phase88_claim_lock_table"])
    positive_floor_rows = _read_csv(resolved["phase116_positive_floor_table"])
    claim_status_rows = _read_csv(resolved["phase116_claim_status_table"])
    claim_boundary_rows = _read_csv(resolved["phase152_claim_boundary_table"])
    phase152_gate = _read_json(resolved["phase152_gate"])

    contribution_rows = build_contribution_rows(
        root=root,
        phase60_main_table=resolved["phase60_main_table"],
        phase60_route_guard_table=resolved["phase60_route_guard_table"],
        phase88_claim_lock_table=resolved["phase88_claim_lock_table"],
        phase116_positive_floor_table=resolved["phase116_positive_floor_table"],
        phase152_claim_boundary_table=resolved["phase152_claim_boundary_table"],
        main_rows=main_rows,
    )
    section_rows = build_section_rows(root=root, phase_inputs=resolved)
    phrasing_rows = build_phrasing_rows(root=root, phase_inputs=resolved)
    open_gap_rows = build_open_gap_rows(phase152_gate=phase152_gate)
    gate = build_gate(
        contribution_rows=contribution_rows,
        section_rows=section_rows,
        phrasing_rows=phrasing_rows,
        open_gap_rows=open_gap_rows,
        phase152_gate=phase152_gate,
    )

    contribution_path = output_dir / "phase153_contribution_refinement_table.csv"
    section_path = output_dir / "phase153_manuscript_section_map.csv"
    phrasing_path = output_dir / "phase153_claim_phrasing_guard_table.csv"
    gap_path = output_dir / "phase153_open_gap_table.csv"
    gate_path = output_dir / "phase153_first_paper_contribution_refinement_gate.json"
    markdown_path = output_dir / "phase153_first_paper_contribution_refinement.md"
    manifest_path = output_dir / "phase153_first_paper_contribution_refinement_manifest.json"

    _write_csv(contribution_path, contribution_rows, CONTRIBUTION_FIELDS)
    _write_csv(section_path, section_rows, SECTION_FIELDS)
    _write_csv(phrasing_path, phrasing_rows, PHRASING_FIELDS)
    _write_csv(gap_path, open_gap_rows, OPEN_GAP_FIELDS)
    _write_json(gate_path, gate)
    with markdown_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            build_markdown(
                gate=gate,
                contribution_rows=contribution_rows,
                section_rows=section_rows,
                phrasing_rows=phrasing_rows,
                open_gap_rows=open_gap_rows,
            )
        )

    manifest = {
        "phase": 153,
        "description": "first-paper contribution refinement and claim-language guard package",
        "inputs": {name: _display_path(path, root) for name, path in resolved.items()},
        "outputs": {
            "contribution_table": _display_path(contribution_path, root),
            "section_map": _display_path(section_path, root),
            "phrasing_guard_table": _display_path(phrasing_path, root),
            "open_gap_table": _display_path(gap_path, root),
            "gate": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "phase60_main_rows": len(main_rows),
            "phase60_route_guard_rows": len(route_guard_rows),
            "phase88_claim_lock_rows": len(claim_lock_rows),
            "phase116_positive_floor_rows": len(positive_floor_rows),
            "phase116_claim_status_rows": len(claim_status_rows),
            "phase152_claim_boundary_rows": len(claim_boundary_rows),
            "contribution_rows": len(contribution_rows),
            "section_rows": len(section_rows),
            "phrasing_guard_rows": len(phrasing_rows),
            "submission_blocker_rows": gate["submission_blocker_rows"],
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
