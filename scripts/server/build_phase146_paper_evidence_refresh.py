#!/usr/bin/env python3
"""Build Phase 146 paper evidence refresh after Phase 144-145 MPEA diagnostics.

This package consumes only existing small artifacts. It refreshes the first
paper claim boundary after the latest MPEA external-data diagnostic, without
reading raw data, running baselines, or opening model training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("docs/results/phase146_paper_evidence_refresh")

EXTERNAL_FIELDS = (
    "branch_id",
    "source_phases",
    "dataset",
    "target",
    "terminal_status",
    "blocking_audits",
    "final_use",
    "paper_boundary",
    "model_training_allowed",
    "a100_training_allowed_now",
    "a100_80gb_request_now",
    "evidence_source",
)
CLAIM_FIELDS = (
    "claim_id",
    "claim_area",
    "claim_status",
    "allowed_use",
    "wording_guard",
    "evidence_anchor",
)
DECISION_FIELDS = (
    "decision_id",
    "route",
    "decision",
    "rationale",
    "blocks_submission",
    "blocks_model_training",
    "next_action",
    "evidence_anchor",
)

PHASE_INPUTS = {
    "phase143_gate": Path(
        "docs/results/phase143_paper_evidence_refresh/phase143_paper_evidence_refresh_gate.json"
    ),
    "phase145_mpea_terminal": Path(
        "docs/results/phase145_mpea_mechanical_focused_review/"
        "phase145_mpea_mechanical_focused_review_gate.json"
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


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


def _display_path(path: Path, root: Path | None = None) -> str:
    try:
        if root is not None:
            return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        pass
    return path.as_posix()


def _is_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value is False
    if isinstance(value, str):
        return value.strip().lower() in {"false", "0", "no", ""}
    return not bool(value)


def _blocking_audits(gate: dict[str, Any]) -> str:
    audits = gate.get("blocking_audits", [])
    if isinstance(audits, list):
        return ";".join(str(item) for item in audits)
    return str(audits)


def build_external_diagnostic_rows(
    *, phase145_gate: dict[str, Any], phase145_path: Path, root: Path
) -> list[dict[str, Any]]:
    return [
        {
            "branch_id": "mpea_mechanical",
            "source_phases": "144-145",
            "dataset": "Citrine Informatics MPEA dataset",
            "target": phase145_gate.get("selected_target", "hardness_hv"),
            "terminal_status": phase145_gate.get("status"),
            "blocking_audits": _blocking_audits(phase145_gate),
            "final_use": "appendix_negative_external_diagnostic",
            "paper_boundary": (
                "MPEA hardness is blocked by split sensitivity, process/shortcut controls, "
                "and target-distribution imbalance"
            ),
            "model_training_allowed": phase145_gate.get("phase145_model_training_allowed", False),
            "a100_training_allowed_now": phase145_gate.get("a100_training_allowed_now", False),
            "a100_80gb_request_now": phase145_gate.get("a100_80gb_request_now", False),
            "evidence_source": _display_path(phase145_path, root),
        }
    ]


def build_claim_boundary_rows(
    *, phase143_gate: dict[str, Any], external_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    external_names = ", ".join(row["branch_id"] for row in external_rows)
    return [
        {
            "claim_id": "P146-CLAIM-001",
            "claim_area": "first_paper_main_claim",
            "claim_status": "allowed_narrow_floor",
            "allowed_use": "main_text",
            "wording_guard": (
                "claim only route-guarded fixed-sampling broad12/broad21 spot_size under "
                "broad_process_v1; Phase 144-145 MPEA diagnostics do not expand the main claim"
            ),
            "evidence_anchor": "docs/results/phase116_paper_evidence_consolidation/phase116_positive_floor_table.csv",
        },
        {
            "claim_id": "P146-CLAIM-002",
            "claim_area": "latest_mpea_branch",
            "claim_status": "diagnostic_only",
            "allowed_use": "appendix_or_limitations",
            "wording_guard": f"Phase 144-145 branches are closed diagnostics: {external_names}",
            "evidence_anchor": "docs/results/phase146_paper_evidence_refresh/phase146_external_diagnostic_refresh_table.csv",
        },
        {
            "claim_id": "P146-CLAIM-003",
            "claim_area": "not_allowed_claims",
            "claim_status": "blocked",
            "allowed_use": "explicit_exclusions",
            "wording_guard": (
                "do not claim complete GNN-PINN, general process-condition modeling, "
                "density-invariant robustness, source-path/Green feature success, "
                "microstructure GNN success, Matbench glass/is-metal model success, "
                "or MPEA hardness model success"
            ),
            "evidence_anchor": "docs/results/phase145_mpea_mechanical_focused_review/phase145_mpea_mechanical_focused_review_gate.json",
        },
        {
            "claim_id": "P146-CLAIM-004",
            "claim_area": "submission_readiness",
            "claim_status": "blocked_missing_venue_benchmark",
            "allowed_use": "planning",
            "wording_guard": "submission readiness still requires target venue and benchmark-paper comparison",
            "evidence_anchor": "docs/results/phase116_paper_evidence_consolidation/phase116_remaining_blocker_table.csv",
        },
        {
            "claim_id": "P146-CLAIM-005",
            "claim_area": "phase143_refresh_status",
            "claim_status": "preserved" if phase143_gate.get("first_paper_draft_allowed_now") else "incomplete",
            "allowed_use": "quality_gate",
            "wording_guard": "Phase 146 preserves the Phase 143/116 floor evidence and does not strengthen claims",
            "evidence_anchor": "docs/results/phase143_paper_evidence_refresh/phase143_paper_evidence_refresh_gate.json",
        },
    ]


def build_decision_rows(
    *, phase143_gate: dict[str, Any], external_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    locks_ok = all(
        _is_false(row["model_training_allowed"])
        and _is_false(row["a100_training_allowed_now"])
        and _is_false(row["a100_80gb_request_now"])
        for row in external_rows
    )
    floor_ready = bool(phase143_gate.get("first_paper_draft_allowed_now")) and locks_ok
    return [
        {
            "decision_id": "P146-DECISION-001",
            "route": "first_paper_claim_boundary",
            "decision": "preserve_narrow_claims" if floor_ready else "blocked",
            "rationale": "Phase 144-145 MPEA diagnostics do not add a new model claim",
            "blocks_submission": True,
            "blocks_model_training": False,
            "next_action": "continue first-paper polishing around the narrow floor or run another no-training baseline-first intake",
            "evidence_anchor": "docs/results/phase146_paper_evidence_refresh/phase146_claim_boundary_refresh_table.csv",
        },
        {
            "decision_id": "P146-DECISION-002",
            "route": "phase144_145_mpea_training",
            "decision": "blocked",
            "rationale": "MPEA branch failed focused review and keeps model training/A100 locks false",
            "blocks_submission": False,
            "blocks_model_training": True,
            "next_action": "do not train on Phase 144-145 MPEA diagnostics",
            "evidence_anchor": "docs/results/phase145_mpea_mechanical_focused_review/phase145_mpea_mechanical_focused_review_gate.json",
        },
        {
            "decision_id": "P146-DECISION-003",
            "route": "a100_sxm4_80gb_request",
            "decision": "blocked",
            "rationale": "no seed-positive branch has produced a measured 40GB memory/runtime blockage",
            "blocks_submission": False,
            "blocks_model_training": False,
            "next_action": "continue using A800 40GB for no-training reviews and small reproductions",
            "evidence_anchor": "docs/results/phase146_paper_evidence_refresh/phase146_external_diagnostic_refresh_table.csv",
        },
    ]


def build_gate(
    *, phase143_gate: dict[str, Any], external_rows: list[dict[str, Any]], decision_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    phase143_ready = phase143_gate.get("first_paper_draft_allowed_now") is True
    external_complete = len(external_rows) == 1
    locks_ok = all(
        _is_false(row["model_training_allowed"])
        and _is_false(row["a100_training_allowed_now"])
        and _is_false(row["a100_80gb_request_now"])
        for row in external_rows
    )
    ready = phase143_ready and external_complete and locks_ok
    return {
        "status": (
            "phase146_paper_evidence_refresh_ready_first_paper_narrow_claims"
            if ready
            else "phase146_paper_evidence_refresh_incomplete"
        ),
        "phase143_first_paper_draft_allowed": phase143_ready,
        "latest_external_branches": len(external_rows),
        "latest_external_diagnostics_complete": external_complete,
        "latest_external_training_locks_verified": locks_ok,
        "first_paper_draft_allowed_now": ready,
        "first_paper_submission_ready": False,
        "main_paper_floor": phase143_gate.get("main_paper_floor"),
        "new_external_model_claim_ready": False,
        "phase146_model_mechanism_allowed": False,
        "phase146_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "decision_rows": len(decision_rows),
        "blocked_model_training_routes": sum(
            1 for row in decision_rows if str(row.get("blocks_model_training")).lower() == "true"
        ),
        "next_action": (
            "continue first-paper refinement around the route-guarded spot_size floor, "
            "or open a fresh no-training baseline-first source intake; do not train from closed diagnostics"
        ),
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_csv_value(row.get(key)).replace("\n", " ") for key, _ in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def build_markdown(
    *, gate: dict[str, Any], external_rows: list[dict[str, Any]], claim_rows: list[dict[str, Any]], decision_rows: list[dict[str, Any]]
) -> str:
    return "\n".join(
        [
            "# Phase 146 Paper Evidence Refresh",
            "",
            f"- Status: `{gate['status']}`",
            f"- First paper draft allowed now: `{gate['first_paper_draft_allowed_now']}`",
            f"- New external model claim ready: `{gate['new_external_model_claim_ready']}`",
            f"- Model training allowed: `{gate['phase146_model_training_allowed']}`",
            f"- A100 80GB request now: `{gate['a100_80gb_request_now']}`",
            "",
            "## Latest External Diagnostics",
            "",
            _markdown_table(
                external_rows,
                [
                    ("branch_id", "Branch"),
                    ("terminal_status", "Status"),
                    ("final_use", "Final use"),
                    ("paper_boundary", "Boundary"),
                ],
            ),
            "",
            "## Claim Boundary",
            "",
            _markdown_table(
                claim_rows,
                [
                    ("claim_id", "Claim"),
                    ("claim_status", "Status"),
                    ("allowed_use", "Allowed use"),
                    ("wording_guard", "Guard"),
                ],
            ),
            "",
            "## Decisions",
            "",
            _markdown_table(
                decision_rows,
                [
                    ("decision_id", "Decision"),
                    ("route", "Route"),
                    ("decision", "Outcome"),
                    ("next_action", "Next action"),
                ],
            ),
        ]
    ) + "\n"


def build_package(*, root: Path, output_dir: Path, phase_inputs: dict[str, Path]) -> dict[str, Any]:
    resolved = {name: path if path.is_absolute() else root / path for name, path in phase_inputs.items()}
    phase143_gate = _read_json(resolved["phase143_gate"])
    phase145_gate = _read_json(resolved["phase145_mpea_terminal"])
    external_rows = build_external_diagnostic_rows(
        phase145_gate=phase145_gate,
        phase145_path=resolved["phase145_mpea_terminal"],
        root=root,
    )
    claim_rows = build_claim_boundary_rows(phase143_gate=phase143_gate, external_rows=external_rows)
    decision_rows = build_decision_rows(phase143_gate=phase143_gate, external_rows=external_rows)
    gate = build_gate(phase143_gate=phase143_gate, external_rows=external_rows, decision_rows=decision_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    external_path = output_dir / "phase146_external_diagnostic_refresh_table.csv"
    claim_path = output_dir / "phase146_claim_boundary_refresh_table.csv"
    decision_path = output_dir / "phase146_next_decision_table.csv"
    gate_path = output_dir / "phase146_paper_evidence_refresh_gate.json"
    markdown_path = output_dir / "phase146_paper_evidence_refresh.md"
    manifest_path = output_dir / "phase146_paper_evidence_refresh_manifest.json"

    _write_csv(external_path, external_rows, EXTERNAL_FIELDS)
    _write_csv(claim_path, claim_rows, CLAIM_FIELDS)
    _write_csv(decision_path, decision_rows, DECISION_FIELDS)
    _write_json(gate_path, gate)
    markdown_path.write_text(
        build_markdown(
            gate=gate,
            external_rows=external_rows,
            claim_rows=claim_rows,
            decision_rows=decision_rows,
        ),
        encoding="utf-8",
    )
    manifest = {
        "phase": 146,
        "objective": "paper_evidence_refresh_after_phase144_145_mpea_diagnostics",
        "inputs": {name: _display_path(path, root) for name, path in sorted(resolved.items())},
        "outputs": {
            "external_diagnostic_refresh_table": _display_path(external_path, root),
            "claim_boundary_refresh_table": _display_path(claim_path, root),
            "next_decision_table": _display_path(decision_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "latest_external_diagnostic_rows": len(external_rows),
            "claim_boundary_rows": len(claim_rows),
            "next_decision_rows": len(decision_rows),
            "training_allowed_external_rows": sum(
                1 for row in external_rows if not _is_false(row["model_training_allowed"])
            ),
            "a100_training_allowed_external_rows": sum(
                1 for row in external_rows if not _is_false(row["a100_training_allowed_now"])
            ),
            "a100_80gb_allowed_external_rows": sum(
                1 for row in external_rows if not _is_false(row["a100_80gb_request_now"])
            ),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    for name, default_path in PHASE_INPUTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default_path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    phase_inputs = {name: getattr(args, name) for name in PHASE_INPUTS}
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(root=root, output_dir=output_dir, phase_inputs=phase_inputs)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
