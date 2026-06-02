#!/usr/bin/env python3
"""Build the Phase 90 manuscript-v1 claim-integration package.

Phase 90 integrates Phase 89 writing-ready literature evidence into the Phase
61/74 manuscript base while preserving the Phase 55/60/74 experimental claim
boundary. It does not finalize target-venue style, citation density, or layout.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


INTEGRATION_FIELDS = (
    "integration_id",
    "target_section",
    "claim_anchor",
    "evidence_ids",
    "integration_status",
    "allowed_strength",
    "wording_guard",
    "manuscript_locator",
    "unresolved_dependency",
)

AUDIT_FIELDS = (
    "claim_id",
    "claim_type",
    "source_anchor",
    "support_locator",
    "verification_state",
    "integrated_in_v1",
    "allowed_strength",
    "wording_guard",
    "blocker",
)

BLOCKER_FIELDS = (
    "blocker_id",
    "category",
    "status",
    "required_input",
    "blocks_submission",
    "blocks_phase90_core_claims",
    "next_action",
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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
        raise ValueError(f"Expected at least one row in {path}")
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _default_paths(root: Path) -> dict[str, Path]:
    phase61 = root / "docs/results/phase61_manuscript_draft_package"
    phase74 = root / "docs/results/phase74_manuscript_v0_claim_audit"
    phase88 = root / "docs/results/phase88_fallback_manuscript_finalization"
    phase89 = root / "docs/results/phase89_literature_venue_gap_resolution"
    return {
        "phase61_results": phase61 / "phase61_results_draft.md",
        "phase61_methods": phase61 / "phase61_methods_draft.md",
        "phase61_captions": phase61 / "phase61_table_figure_captions.md",
        "phase61_crosswalk": phase61 / "phase61_claim_evidence_crosswalk.csv",
        "phase74_manifest": phase74 / "phase74_manuscript_v0_claim_audit_manifest.json",
        "phase74_boundary": phase74 / "phase74_model_boundary_register.csv",
        "phase88_gate": phase88 / "phase88_fallback_finalization_gate.json",
        "phase89_evidence": phase89 / "phase89_evidence_register.csv",
        "phase89_gap_resolution": phase89 / "phase89_gap_resolution_table.csv",
        "phase89_handoff": phase89 / "phase89_writing_handoff.csv",
        "phase89_manual_queue": phase89 / "phase89_manual_verification_queue.csv",
        "phase89_manifest": phase89 / "phase89_literature_venue_gap_resolution_manifest.json",
    }


def _evidence_citation(row: dict[str, str]) -> str:
    doi = row.get("doi", "")
    doi_text = f", DOI {doi}" if doi else ""
    return f"{row.get('source_title')} ({row.get('authors_or_owner')}, {row.get('year')}{doi_text})"


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}\n"


def _handoff_by_id(handoff_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["handoff_id"]: row for row in handoff_rows}


def _evidence_by_id(evidence_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["evidence_id"]: row for row in evidence_rows}


def _format_evidence_list(evidence_ids: str, evidence_lookup: dict[str, dict[str, str]]) -> str:
    ids = [item for item in evidence_ids.split(";") if item]
    lines = []
    for evidence_id in ids:
        evidence = evidence_lookup[evidence_id]
        lines.append(f"- [{evidence_id}] {_evidence_citation(evidence)}. {evidence.get('supports_claim')}")
    return "\n".join(lines)


def build_integrated_manuscript(
    results_text: str,
    methods_text: str,
    captions_text: str,
    handoff_rows: list[dict[str, str]],
    evidence_rows: list[dict[str, str]],
    boundary_rows: list[dict[str, str]],
    phase74_manifest: dict[str, Any],
) -> str:
    handoffs = _handoff_by_id(handoff_rows)
    evidence_lookup = _evidence_by_id(evidence_rows)
    dataset = handoffs["P89-HANDOFF-DATASET"]
    pinn = handoffs["P89-HANDOFF-PINN"]
    conditioning = handoffs["P89-HANDOFF-CONDITIONING"]
    venue = handoffs["P89-HANDOFF-VENUE"]
    claim_boundary = phase74_manifest.get("claim_boundary") or {}
    main_claim = claim_boundary.get(
        "main_claim",
        "fixed-sampling broad12/broad21 spot_size under broad_process_v1, seeds 7/1/2",
    )
    excluded = claim_boundary.get("excluded_claims") or []
    excluded_lines = "\n".join(f"- {item}" for item in excluded)
    blocked_boundaries = [
        row
        for row in boundary_rows
        if row.get("status") not in {"ready", "supported_for_v0", ""}
    ]
    boundary_lines = "\n".join(
        f"- `{row.get('boundary_id')}`: {row.get('candidate_or_scope')} -> {row.get('status')}"
        for row in blocked_boundaries
    )
    return "\n".join(
        [
            "# Phase 90 Manuscript v1 Claim-Integrated Draft",
            "",
            _section(
                "Scope And Venue Gate",
                (
                    f"Main claim: `{main_claim}`.\n\n"
                    "This manuscript v1 integrates writing-ready literature evidence from Phase 89 with the Phase 61/74 experimental claim package. It remains provisional for venue-specific section order, citation density, and caption style because the target venue has not been provided.\n\n"
                    "Excluded from the main claim:\n\n"
                    f"{excluded_lines}"
                ),
            ),
            _section(
                "Introduction And Dataset Context",
                (
                    f"{dataset['allowed_claim']} [{dataset['evidence_ids']}]\n\n"
                    "Writing support:\n\n"
                    f"{_format_evidence_list(dataset['evidence_ids'], evidence_lookup)}\n\n"
                    f"Guard: {dataset['wording_guard']}"
                ),
            ),
            _section(
                "Related Work And Method Context",
                (
                    f"{pinn['allowed_claim']} [{pinn['evidence_ids']}]\n\n"
                    f"{conditioning['allowed_claim']} [{conditioning['evidence_ids']}]\n\n"
                    "Writing support:\n\n"
                    f"{_format_evidence_list(pinn['evidence_ids'], evidence_lookup)}\n"
                    f"{_format_evidence_list(conditioning['evidence_ids'], evidence_lookup)}\n\n"
                    f"Guards: {pinn['wording_guard']} {conditioning['wording_guard']}"
                ),
            ),
            _section(
                "Methods",
                (
                    methods_text
                    + "\n\n"
                    "Phase 90 integration note: the conditioning mechanism may be described with FiLM-style feature-wise modulation support, but performance claims must remain tied to Phase 55/60/74 artifacts rather than to the conditioning literature alone. [P89-HANDOFF-CONDITIONING; C61-METHOD-001]"
                ),
            ),
            _section("Results", results_text),
            _section("Tables And Figures", captions_text),
            _section(
                "Limitations And Appendix Boundaries",
                (
                    "The following boundaries remain limitations, appendix evidence, or future-work gates rather than main claims:\n\n"
                    f"{boundary_lines}\n\n"
                    "Phase 75 Bayesian inverse closure, Phase 79/80 bounded spot-size parameterization, and Phase 81 registered-target intake remain diagnostic or future-work evidence unless a future branch passes the local/no-training and A100 gates."
                ),
            ),
            _section(
                "Remaining Venue Dependency",
                (
                    f"{venue['allowed_claim']}\n\n"
                    f"Unresolved dependency: {venue['unresolved_dependency']}.\n\n"
                    f"Guard: {venue['wording_guard']}"
                ),
            ),
        ]
    )


def build_integration_rows(handoff_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    locator_by_handoff = {
        "P89-HANDOFF-DATASET": "Phase 90 manuscript v1: Introduction And Dataset Context",
        "P89-HANDOFF-PINN": "Phase 90 manuscript v1: Related Work And Method Context",
        "P89-HANDOFF-CONDITIONING": "Phase 90 manuscript v1: Related Work And Method Context; Methods",
        "P89-HANDOFF-VENUE": "Phase 90 manuscript v1: Remaining Venue Dependency",
    }
    for row in handoff_rows:
        unresolved = row.get("unresolved_dependency", "")
        rows.append(
            {
                "integration_id": row["handoff_id"],
                "target_section": row["target_section"],
                "claim_anchor": row["claim_anchor"],
                "evidence_ids": row.get("evidence_ids", ""),
                "integration_status": "blocked_user_input_required" if unresolved else "integrated_writing_ready",
                "allowed_strength": row.get("allowed_strength"),
                "wording_guard": row.get("wording_guard"),
                "manuscript_locator": locator_by_handoff.get(row["handoff_id"], "Phase 90 manuscript v1"),
                "unresolved_dependency": unresolved,
            }
        )
    return rows


def build_audit_rows(
    crosswalk_rows: list[dict[str, str]],
    evidence_rows: list[dict[str, str]],
    handoff_rows: list[dict[str, str]],
    gap_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in crosswalk_rows:
        writing_ready = row.get("verification_state") == "writing_ready"
        rows.append(
            {
                "claim_id": row.get("claim_anchor_id"),
                "claim_type": row.get("support_type"),
                "source_anchor": row.get("evidence_register_key"),
                "support_locator": row.get("support_locator"),
                "verification_state": row.get("verification_state"),
                "integrated_in_v1": writing_ready,
                "allowed_strength": row.get("allowed_claim_strength"),
                "wording_guard": row.get("open_risk"),
                "blocker": "" if writing_ready else "Phase 61 claim is not writing-ready",
            }
        )
    for evidence in evidence_rows:
        writing_ready = evidence.get("writing_ready") == "true"
        rows.append(
            {
                "claim_id": evidence.get("evidence_id"),
                "claim_type": "literature_or_dataset_evidence",
                "source_anchor": evidence.get("gap_id"),
                "support_locator": evidence.get("stable_url"),
                "verification_state": evidence.get("trust_state"),
                "integrated_in_v1": writing_ready,
                "allowed_strength": evidence.get("allowed_claim_strength"),
                "wording_guard": evidence.get("limitations"),
                "blocker": "" if writing_ready else "Phase 89 evidence is not writing-ready",
            }
        )
    for handoff in handoff_rows:
        unresolved = handoff.get("unresolved_dependency", "")
        rows.append(
            {
                "claim_id": handoff.get("handoff_id"),
                "claim_type": "writing_handoff",
                "source_anchor": handoff.get("claim_anchor"),
                "support_locator": handoff.get("source_locator"),
                "verification_state": "blocked_user_input_required" if unresolved else "writing_ready",
                "integrated_in_v1": not bool(unresolved),
                "allowed_strength": handoff.get("allowed_strength"),
                "wording_guard": handoff.get("wording_guard"),
                "blocker": unresolved,
            }
        )
    for gap in gap_rows:
        if gap.get("blocks_submission_after_phase89") == "true":
            rows.append(
                {
                    "claim_id": gap.get("gap_id"),
                    "claim_type": "submission_blocker",
                    "source_anchor": gap.get("location"),
                    "support_locator": "docs/results/phase89_literature_venue_gap_resolution/phase89_gap_resolution_table.csv",
                    "verification_state": gap.get("resolution_status"),
                    "integrated_in_v1": False,
                    "allowed_strength": "none_until_resolved",
                    "wording_guard": gap.get("next_action"),
                    "blocker": gap.get("unresolved_reason"),
                }
            )
    return rows


def build_blocker_rows(manual_rows: list[dict[str, str]], gap_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(manual_rows, start=1):
        rows.append(
            {
                "blocker_id": f"P90-BLOCKER-{index:03d}",
                "category": row.get("category"),
                "status": "unresolved_user_input_required",
                "required_input": row.get("needed_input"),
                "blocks_submission": row.get("blocks_submission") == "true",
                "blocks_phase90_core_claims": False,
                "next_action": row.get("suggested_user_action"),
            }
        )
    if not rows:
        unresolved_gaps = [row for row in gap_rows if row.get("blocks_submission_after_phase89") == "true"]
        for index, row in enumerate(unresolved_gaps, start=1):
            rows.append(
                {
                    "blocker_id": f"P90-BLOCKER-{index:03d}",
                    "category": "gap_resolution",
                    "status": row.get("resolution_status"),
                    "required_input": row.get("unresolved_reason"),
                    "blocks_submission": True,
                    "blocks_phase90_core_claims": False,
                    "next_action": row.get("next_action"),
                }
            )
    return rows


def build_gate(
    integration_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    blocker_rows: list[dict[str, Any]],
    phase88_gate: dict[str, Any],
    phase89_manifest: dict[str, Any],
) -> dict[str, Any]:
    unresolved_core = [
        row
        for row in integration_rows
        if row["integration_status"] != "integrated_writing_ready"
        and row["integration_id"] != "P89-HANDOFF-VENUE"
    ]
    submission_blockers = [row for row in blocker_rows if row["blocks_submission"]]
    integrated_core = not unresolved_core
    gate = phase89_manifest.get("gate") or {}
    if integrated_core and submission_blockers:
        status = "manuscript_v1_core_claims_integrated_venue_unresolved"
        next_action = "enter Phase 91 table/figure and appendix freeze; request target venue before final formatting"
    elif integrated_core:
        status = "manuscript_v1_claims_integrated_submission_ready_for_formatting"
        next_action = "enter Phase 91 table/figure freeze and Phase 93 formatting"
    else:
        status = "manuscript_v1_claim_integration_blocked"
        next_action = "repair unresolved non-venue claim integration rows"
    return {
        "status": status,
        "experimental_claim_complete": bool(phase88_gate.get("experimental_claim_complete")),
        "core_literature_ready": bool(gate.get("core_literature_ready")),
        "core_claims_integrated": integrated_core,
        "venue_alignment_ready": False if submission_blockers else bool(gate.get("venue_alignment_ready")),
        "submission_ready": False if submission_blockers else integrated_core,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "integrated_handoff_rows": sum(
            1 for row in integration_rows if row["integration_status"] == "integrated_writing_ready"
        ),
        "blocked_handoff_rows": sum(
            1 for row in integration_rows if row["integration_status"] != "integrated_writing_ready"
        ),
        "audit_rows": len(audit_rows),
        "submission_blockers": len(submission_blockers),
        "next_action": next_action,
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
    gate: dict[str, Any],
    integration_rows: list[dict[str, Any]],
    blocker_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Phase 90 Manuscript v1 Claim Integration",
            "",
            "## Gate Decision",
            "",
            f"Status: `{gate['status']}`.",
            f"Core claims integrated: `{str(gate['core_claims_integrated']).lower()}`.",
            f"Submission ready: `{str(gate['submission_ready']).lower()}`.",
            f"A100 training allowed now: `{str(gate['a100_training_allowed_now']).lower()}`.",
            "",
            "## Integration Rows",
            "",
            _markdown_table(
                integration_rows,
                [
                    ("integration_id", "Integration"),
                    ("target_section", "Section"),
                    ("integration_status", "Status"),
                    ("allowed_strength", "Strength"),
                    ("unresolved_dependency", "Dependency"),
                ],
            ),
            "",
            "## Remaining Blockers",
            "",
            _markdown_table(
                blocker_rows,
                [
                    ("blocker_id", "Blocker"),
                    ("category", "Category"),
                    ("status", "Status"),
                    ("blocks_submission", "Blocks submission"),
                    ("next_action", "Next action"),
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

    results_text = _read_text(resolved["phase61_results"])
    methods_text = _read_text(resolved["phase61_methods"])
    captions_text = _read_text(resolved["phase61_captions"])
    crosswalk_rows = _read_csv(resolved["phase61_crosswalk"])
    phase74_manifest = _read_json(resolved["phase74_manifest"])
    boundary_rows = _read_csv(resolved["phase74_boundary"])
    phase88_gate = _read_json(resolved["phase88_gate"])
    evidence_rows = _read_csv(resolved["phase89_evidence"])
    gap_rows = _read_csv(resolved["phase89_gap_resolution"])
    handoff_rows = _read_csv(resolved["phase89_handoff"])
    manual_rows = _read_csv(resolved["phase89_manual_queue"])
    phase89_manifest = _read_json(resolved["phase89_manifest"])

    integration_rows = build_integration_rows(handoff_rows)
    audit_rows = build_audit_rows(crosswalk_rows, evidence_rows, handoff_rows, gap_rows)
    blocker_rows = build_blocker_rows(manual_rows, gap_rows)
    gate = build_gate(integration_rows, audit_rows, blocker_rows, phase88_gate, phase89_manifest)
    manuscript = build_integrated_manuscript(
        results_text,
        methods_text,
        captions_text,
        handoff_rows,
        evidence_rows,
        boundary_rows,
        phase74_manifest,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manuscript_path = output_dir / "phase90_manuscript_v1_claim_integrated.md"
    integration_path = output_dir / "phase90_literature_integration_table.csv"
    audit_path = output_dir / "phase90_claim_evidence_audit.csv"
    blocker_path = output_dir / "phase90_venue_blocker_queue.csv"
    gate_path = output_dir / "phase90_manuscript_v1_claim_integration_gate.json"
    markdown_path = output_dir / "phase90_manuscript_v1_claim_integration.md"
    manifest_path = output_dir / "phase90_manuscript_v1_claim_integration_manifest.json"

    manuscript_path.write_text(manuscript, encoding="utf-8")
    _write_csv(integration_path, integration_rows, INTEGRATION_FIELDS)
    _write_csv(audit_path, audit_rows, AUDIT_FIELDS)
    _write_csv(blocker_path, blocker_rows, BLOCKER_FIELDS)
    _write_json(gate_path, gate)
    markdown_path.write_text(build_markdown(gate, integration_rows, blocker_rows), encoding="utf-8")

    manifest = {
        "phase": 90,
        "objective": "manuscript_v1_claim_integration",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "manuscript_v1": _display_path(manuscript_path, root),
            "literature_integration_table": _display_path(integration_path, root),
            "claim_evidence_audit": _display_path(audit_path, root),
            "venue_blocker_queue": _display_path(blocker_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "phase61_claim_rows": len(crosswalk_rows),
            "phase89_evidence_rows": len(evidence_rows),
            "phase89_handoff_rows": len(handoff_rows),
            "integration_rows": len(integration_rows),
            "audit_rows": len(audit_rows),
            "blocker_rows": len(blocker_rows),
        },
        "gate": gate,
        "phase74_claim_boundary": phase74_manifest.get("claim_boundary"),
        "phase89_gate": phase89_manifest.get("gate"),
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase90_manuscript_v1_claim_integration"),
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
