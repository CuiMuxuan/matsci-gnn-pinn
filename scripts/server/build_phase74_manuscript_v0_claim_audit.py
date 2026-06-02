#!/usr/bin/env python3
"""Build the Phase 74 manuscript-v0 evidence-locked claim-audit package.

Phase 74 assembles the Phase 60/61 manuscript-facing evidence plus Phase 68-71
candidate gates into a manuscript-v0 package. It does not add training evidence
or literature claims. Its job is to make the current paper claim auditable and
to prevent paused/blocked model branches from leaking into the main claim.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CLAIM_AUDIT_FIELDS = (
    "claim_id",
    "claim_location",
    "claim_summary",
    "source_anchor",
    "support_type",
    "evidence_locator",
    "audit_status",
    "allowed_in_v0",
    "claim_strength",
    "required_wording_guard",
)

INVENTORY_FIELDS = (
    "artifact_id",
    "manuscript_role",
    "artifact_path",
    "source_phase",
    "status",
    "caption_or_use",
)

BOUNDARY_FIELDS = (
    "boundary_id",
    "candidate_or_scope",
    "status",
    "main_text_treatment",
    "appendix_treatment",
    "evidence_locator",
    "reason",
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
    if isinstance(value, float):
        return f"{value:.6f}"
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
    phase60 = root / "docs/results/phase60_manuscript_evidence_package"
    phase61 = root / "docs/results/phase61_manuscript_draft_package"
    phase68 = root / "docs/results/phase68_validation_signal_scorecard"
    phase69 = root / "docs/results/phase69_spot_size_signal_probe"
    phase70 = root / "docs/results/phase70_route_policy_audit"
    phase71 = root / "docs/results/phase71_data_registration_audit"
    phase57 = root / "docs/results/phase57_claim_governance"
    return {
        "phase57_ledger": phase57 / "phase57_claim_ledger.csv",
        "phase60_main": phase60 / "phase60_main_spot_size_seed_positive_table.csv",
        "phase60_route": phase60 / "phase60_route_guard_boundary_table.csv",
        "phase60_stress": phase60 / "phase60_stress_boundary_table.csv",
        "phase60_appendix": phase60 / "phase60_appendix_negative_diagnostic_table.csv",
        "phase60_next_gate": phase60 / "phase60_next_branch_gate_table.csv",
        "phase60_manifest": phase60 / "phase60_manuscript_evidence_package_manifest.json",
        "phase61_results": phase61 / "phase61_results_draft.md",
        "phase61_methods": phase61 / "phase61_methods_draft.md",
        "phase61_captions": phase61 / "phase61_table_figure_captions.md",
        "phase61_crosswalk": phase61 / "phase61_claim_evidence_crosswalk.csv",
        "phase61_gaps": phase61 / "phase61_literature_gap_register.csv",
        "phase61_manifest": phase61 / "phase61_manuscript_draft_package_manifest.json",
        "phase68_scorecard": phase68 / "phase68_candidate_signal_scorecard.csv",
        "phase68_manifest": phase68 / "phase68_validation_signal_scorecard_manifest.json",
        "phase69_gate": phase69 / "phase69_candidate_a_gate.json",
        "phase70_gate": phase70 / "phase70_candidate_b_gate.json",
        "phase71_gate": phase71 / "phase71_candidate_c_gate.json",
        "phase71_manifest": phase71 / "phase71_data_registration_audit_manifest.json",
    }


def _bool_text(value: bool) -> str:
    return "yes" if value else "no"


def build_claim_audit_rows(
    crosswalk_rows: list[dict[str, str]],
    literature_gaps: list[dict[str, str]],
    phase68_manifest: dict[str, Any],
    phase69_gate: dict[str, Any],
    phase70_gate: dict[str, Any],
    phase71_gate: dict[str, Any],
) -> list[dict[str, Any]]:
    gap_count = len(literature_gaps)
    model_opened = bool((phase68_manifest.get("current_decision") or {}).get("trainable_model_opened"))
    rows: list[dict[str, Any]] = []
    for row in crosswalk_rows:
        verification = row.get("verification_state", "")
        strength = row.get("allowed_claim_strength", "")
        open_risk = row.get("open_risk", "")
        if verification == "writing_ready":
            status = "supported_for_v0"
            allowed = True
        else:
            status = "needs_verification"
            allowed = False
        rows.append(
            {
                "claim_id": row.get("claim_anchor_id"),
                "claim_location": row.get("manuscript_location"),
                "claim_summary": row.get("claim_summary"),
                "source_anchor": row.get("claim_anchor_id"),
                "support_type": row.get("support_type"),
                "evidence_locator": row.get("support_locator"),
                "audit_status": status,
                "allowed_in_v0": _bool_text(allowed),
                "claim_strength": strength,
                "required_wording_guard": open_risk,
            }
        )
    rows.extend(
        [
            {
                "claim_id": "C74-LIT-LOCK",
                "claim_location": "Introduction/Related Work",
                "claim_summary": "External literature and target-venue claims remain placeholders until verified.",
                "source_anchor": "phase61_literature_gap_register",
                "support_type": "literature_gap",
                "evidence_locator": "docs/results/phase61_manuscript_draft_package/phase61_literature_gap_register.csv",
                "audit_status": "locked_out_of_v0_claims" if gap_count else "no_literature_gap_rows",
                "allowed_in_v0": "no" if gap_count else "yes",
                "claim_strength": "none_for_unverified_literature",
                "required_wording_guard": "Do not write final Introduction or Related Work claims until literature gaps are resolved.",
            },
            {
                "claim_id": "C74-MODEL-GATE",
                "claim_location": "Discussion/Future Work",
                "claim_summary": "No new model branch is currently open for A100 training.",
                "source_anchor": "phase68_69_70_71_gates",
                "support_type": "gate",
                "evidence_locator": "docs/results/phase68_validation_signal_scorecard/; docs/results/phase69_spot_size_signal_probe/; docs/results/phase70_route_policy_audit/; docs/results/phase71_data_registration_audit/",
                "audit_status": "supported_for_v0",
                "allowed_in_v0": "yes",
                "claim_strength": "boundary",
                "required_wording_guard": (
                    "Candidate A status="
                    f"{phase69_gate.get('status')}; Candidate B status={phase70_gate.get('status')}; "
                    f"Candidate C status={phase71_gate.get('status')}; trainable_model_opened={model_opened}."
                ),
            },
        ]
    )
    return rows


def build_inventory_rows(paths: dict[str, Path], root: Path) -> list[dict[str, Any]]:
    artifact_specs = [
        ("T1", "main_table", "phase60_main", "Phase 60", "ready", "Main spot_size seed-positive table."),
        ("T2", "route_guard_table", "phase60_route", "Phase 60", "ready", "Route-guard and no-process fallback boundary table."),
        ("T3", "stress_boundary_table", "phase60_stress", "Phase 60", "ready", "Stress and density-boundary table."),
        ("T4", "appendix_negative_table", "phase60_appendix", "Phase 60", "ready", "Appendix negative diagnostics and blocked branches."),
        ("T5", "next_branch_gate_table", "phase60_next_gate", "Phase 60", "ready", "Next-branch gate table."),
        ("M1", "results_v0_source", "phase61_results", "Phase 61", "ready", "Results draft source for v0 assembly."),
        ("M2", "methods_v0_source", "phase61_methods", "Phase 61", "ready", "Methods draft source for v0 assembly."),
        ("M3", "captions", "phase61_captions", "Phase 61", "ready", "Caption package for manuscript tables/figures."),
        ("A1", "claim_crosswalk", "phase61_crosswalk", "Phase 61", "ready", "Claim-to-evidence crosswalk."),
        ("A2", "candidate_signal_scorecard", "phase68_scorecard", "Phase 68", "ready", "Model candidate gate scorecard."),
        ("A3", "candidate_a_gate", "phase69_gate", "Phase 69", "ready", "Candidate A non-training gate."),
        ("A4", "candidate_b_gate", "phase70_gate", "Phase 70", "ready", "Candidate B non-training gate."),
        ("A5", "candidate_c_gate", "phase71_gate", "Phase 71", "ready", "Candidate C data-registration gate."),
    ]
    rows = []
    for artifact_id, role, key, source_phase, status, use in artifact_specs:
        path = paths[key]
        rows.append(
            {
                "artifact_id": artifact_id,
                "manuscript_role": role,
                "artifact_path": _display_path(path, root),
                "source_phase": source_phase,
                "status": status if path.exists() else "missing",
                "caption_or_use": use,
            }
        )
    return rows


def build_boundary_rows(
    phase60_manifest: dict[str, Any],
    phase68_manifest: dict[str, Any],
    phase69_gate: dict[str, Any],
    phase70_gate: dict[str, Any],
    phase71_gate: dict[str, Any],
) -> list[dict[str, Any]]:
    claim_boundary = phase60_manifest.get("claim_boundary") or {}
    model_gate = phase60_manifest.get("model_expansion_gate") or {}
    excluded = claim_boundary.get("excluded_claims") or []
    rows: list[dict[str, Any]] = []
    for index, claim in enumerate(excluded, start=1):
        rows.append(
            {
                "boundary_id": f"C74-EXCL-{index:03d}",
                "candidate_or_scope": claim,
                "status": "excluded_from_main_claim",
                "main_text_treatment": "state as limitation or boundary only",
                "appendix_treatment": "map to appendix or stress-boundary evidence",
                "evidence_locator": "docs/results/phase60_manuscript_evidence_package/phase60_manuscript_evidence_package_manifest.json",
                "reason": "Phase 60 claim boundary excludes this wording from the main claim.",
            }
        )
    rows.extend(
        [
            {
                "boundary_id": "C74-GATE-A",
                "candidate_or_scope": phase69_gate.get("candidate", "Candidate A"),
                "status": phase69_gate.get("status"),
                "main_text_treatment": "future work only",
                "appendix_treatment": "non-training gate row",
                "evidence_locator": "docs/results/phase69_spot_size_signal_probe/phase69_candidate_a_gate.json",
                "reason": phase69_gate.get("reason"),
            },
            {
                "boundary_id": "C74-GATE-B",
                "candidate_or_scope": phase70_gate.get("candidate", "Candidate B"),
                "status": phase70_gate.get("status"),
                "main_text_treatment": "future work only",
                "appendix_treatment": "route-policy blocked gate",
                "evidence_locator": "docs/results/phase70_route_policy_audit/phase70_candidate_b_gate.json",
                "reason": phase70_gate.get("reason"),
            },
            {
                "boundary_id": "C74-GATE-C",
                "candidate_or_scope": phase71_gate.get("candidate", "Candidate C"),
                "status": phase71_gate.get("status"),
                "main_text_treatment": "data limitation and future registered-target work",
                "appendix_treatment": "data-registration blocked gate",
                "evidence_locator": "docs/results/phase71_data_registration_audit/phase71_candidate_c_gate.json",
                "reason": phase71_gate.get("reason"),
            },
            {
                "boundary_id": "C74-GATE-DENSITY",
                "candidate_or_scope": "density-failure-driven model expansion",
                "status": model_gate.get("decision"),
                "main_text_treatment": "route boundary, not model signal",
                "appendix_treatment": "Phase 59 residual upper-bound gate",
                "evidence_locator": "docs/results/phase60_manuscript_evidence_package/phase60_manuscript_evidence_package_manifest.json",
                "reason": model_gate.get("reason"),
            },
            {
                "boundary_id": "C74-GATE-TRAINABLE",
                "candidate_or_scope": "all current trainable model branches",
                "status": "no_trainable_model_opened"
                if not (phase68_manifest.get("current_decision") or {}).get("trainable_model_opened")
                else "trainable_model_opened",
                "main_text_treatment": "do not imply a newer architecture has passed",
                "appendix_treatment": "Phase 68 scorecard summary",
                "evidence_locator": "docs/results/phase68_validation_signal_scorecard/phase68_validation_signal_scorecard_manifest.json",
                "reason": (phase68_manifest.get("current_decision") or {}).get("reason"),
            },
        ]
    )
    return rows


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}\n"


def build_manuscript_v0(
    results_text: str,
    methods_text: str,
    boundary_rows: list[dict[str, Any]],
    claim_boundary: dict[str, Any],
) -> str:
    blocked = [
        row
        for row in boundary_rows
        if str(row.get("status", "")).startswith(("paused", "blocked", "block", "no_trainable"))
        or "excluded" in str(row.get("status", ""))
    ]
    boundary_lines = "\n".join(
        f"- `{row['boundary_id']}`: {row['candidate_or_scope']} -> {row['status']}"
        for row in blocked
    )
    return "\n".join(
        [
            "# Phase 74 Manuscript v0 Evidence-Locked Draft",
            "",
            _section(
                "Scope Lock",
                (
                    f"Main claim: `{claim_boundary.get('main_claim')}`.\n\n"
                    "This v0 draft is evidence-locked to internal result claims. It does not finalize Introduction, Related Work, or target-venue style claims while the Phase 61 literature gaps remain open."
                ),
            ),
            _section("Results Draft", results_text),
            _section("Methods Draft", methods_text),
            _section(
                "Model-Expansion Boundaries",
                (
                    "The following boundaries must remain limitations, appendix evidence, or future work until a later gate reopens them:\n\n"
                    f"{boundary_lines}"
                ),
            ),
            _section(
                "Next Writing Action",
                (
                    "Use this v0 as the internal manuscript base. The next writing step is to resolve literature and target-venue gaps, then polish sections without changing the evidence boundary."
                ),
            ),
        ]
    )


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(key, "")).replace("\n", " ") for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def build_package_markdown(
    manifest: dict[str, Any],
    claim_rows: list[dict[str, Any]],
    inventory_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
) -> str:
    gate = manifest["writing_stage_gate"]
    return "\n".join(
        [
            "# Phase 74 Manuscript v0 Claim-Audit Package",
            "",
            "## Writing Gate",
            "",
            f"Mode: `{gate['mode']}`.",
            f"Status: `{gate['status']}`.",
            f"Main claim locked: `{str(gate['main_claim_locked']).lower()}`.",
            f"Literature gaps open: `{gate['literature_gap_rows']}`.",
            f"Trainable model opened now: `{str(gate['trainable_model_opened_now']).lower()}`.",
            "",
            "## Claim Audit",
            "",
            _markdown_table(
                claim_rows,
                [
                    ("claim_id", "Claim"),
                    ("claim_location", "Location"),
                    ("audit_status", "Audit status"),
                    ("allowed_in_v0", "Allowed"),
                    ("claim_strength", "Strength"),
                ],
            ),
            "",
            "## Table/Figure Inventory",
            "",
            _markdown_table(
                inventory_rows,
                [
                    ("artifact_id", "Artifact"),
                    ("manuscript_role", "Role"),
                    ("status", "Status"),
                    ("artifact_path", "Path"),
                ],
            ),
            "",
            "## Boundary Register",
            "",
            _markdown_table(
                boundary_rows,
                [
                    ("boundary_id", "Boundary"),
                    ("candidate_or_scope", "Scope"),
                    ("status", "Status"),
                    ("main_text_treatment", "Main text treatment"),
                ],
            ),
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

    phase60_manifest = _read_json(resolved["phase60_manifest"])
    phase61_manifest = _read_json(resolved["phase61_manifest"])
    phase68_manifest = _read_json(resolved["phase68_manifest"])
    phase69_gate = _read_json(resolved["phase69_gate"])
    phase70_gate = _read_json(resolved["phase70_gate"])
    phase71_gate = _read_json(resolved["phase71_gate"])
    phase71_manifest = _read_json(resolved["phase71_manifest"])
    crosswalk_rows = _read_csv(resolved["phase61_crosswalk"])
    literature_gaps = _read_csv(resolved["phase61_gaps"])
    results_text = _read_text(resolved["phase61_results"])
    methods_text = _read_text(resolved["phase61_methods"])

    claim_rows = build_claim_audit_rows(
        crosswalk_rows,
        literature_gaps,
        phase68_manifest,
        phase69_gate,
        phase70_gate,
        phase71_gate,
    )
    inventory_rows = build_inventory_rows(resolved, root)
    boundary_rows = build_boundary_rows(
        phase60_manifest,
        phase68_manifest,
        phase69_gate,
        phase70_gate,
        phase71_gate,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manuscript_path = output_dir / "phase74_manuscript_v0_evidence_locked.md"
    claim_audit_path = output_dir / "phase74_claim_audit_table.csv"
    inventory_path = output_dir / "phase74_table_figure_inventory.csv"
    boundary_path = output_dir / "phase74_model_boundary_register.csv"
    package_path = output_dir / "phase74_manuscript_v0_claim_audit_package.md"
    manifest_path = output_dir / "phase74_manuscript_v0_claim_audit_manifest.json"

    claim_boundary = phase60_manifest.get("claim_boundary") or {}
    manuscript_path.write_text(
        build_manuscript_v0(results_text, methods_text, boundary_rows, claim_boundary),
        encoding="utf-8",
    )
    _write_csv(claim_audit_path, claim_rows, CLAIM_AUDIT_FIELDS)
    _write_csv(inventory_path, inventory_rows, INVENTORY_FIELDS)
    _write_csv(boundary_path, boundary_rows, BOUNDARY_FIELDS)

    unsupported_claim_rows = [
        row for row in claim_rows if row["allowed_in_v0"] == "no"
    ]
    missing_inventory = [row for row in inventory_rows if row["status"] == "missing"]
    trainable_opened = bool((phase68_manifest.get("current_decision") or {}).get("trainable_model_opened"))
    gate = {
        "mode": "evidence_locked_manuscript_v0",
        "status": "ready_for_internal_manuscript_review",
        "main_claim_locked": True,
        "trainable_model_opened_now": trainable_opened,
        "candidate_a_status": phase69_gate.get("status"),
        "candidate_b_status": phase70_gate.get("status"),
        "candidate_c_status": phase71_gate.get("status"),
        "literature_gap_rows": len(literature_gaps),
        "unsupported_v0_claim_rows": len(unsupported_claim_rows),
        "missing_inventory_rows": len(missing_inventory),
        "next_action": "resolve literature/venue gaps or start Phase 75 local identifiability gate; do not start A100 model training from current gates",
    }
    manifest = {
        "phase": 74,
        "objective": "manuscript_v0_evidence_locked_claim_audit",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "manuscript_v0": _display_path(manuscript_path, root),
            "claim_audit_table": _display_path(claim_audit_path, root),
            "table_figure_inventory": _display_path(inventory_path, root),
            "model_boundary_register": _display_path(boundary_path, root),
            "package_markdown": _display_path(package_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "claim_audit_rows": len(claim_rows),
            "supported_claim_rows": sum(1 for row in claim_rows if row["allowed_in_v0"] == "yes"),
            "unsupported_v0_claim_rows": len(unsupported_claim_rows),
            "inventory_rows": len(inventory_rows),
            "boundary_rows": len(boundary_rows),
            "literature_gap_rows": len(literature_gaps),
        },
        "writing_stage_gate": gate,
        "claim_boundary": claim_boundary,
        "model_expansion_gate": phase60_manifest.get("model_expansion_gate"),
        "phase61_writing_gate": phase61_manifest.get("writing_stage_gate"),
        "phase71_candidate_c_gate": phase71_manifest.get("candidate_c_gate"),
    }
    package_path.write_text(
        build_package_markdown(manifest, claim_rows, inventory_rows, boundary_rows),
        encoding="utf-8",
    )
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase74_manuscript_v0_claim_audit"),
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
