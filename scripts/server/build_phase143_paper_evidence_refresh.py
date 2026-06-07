#!/usr/bin/env python3
"""Build Phase 143 paper evidence refresh after Phase 138-142 diagnostics.

This package consumes only existing small artifacts. It refreshes the first
paper claim boundary after the latest Matbench external data diagnostics,
without reading raw data, running baselines, or opening model training.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("docs/results/phase143_paper_evidence_refresh")


def _load_phase137_module():
    script = Path(__file__).with_name("build_phase137_paper_evidence_refresh.py")
    spec = importlib.util.spec_from_file_location("phase137_helpers", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Phase 137 helper module from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase137 = _load_phase137_module()

EXTERNAL_FIELDS = phase137.EXTERNAL_FIELDS
CLAIM_FIELDS = phase137.CLAIM_FIELDS
DECISION_FIELDS = phase137.DECISION_FIELDS

PHASE_INPUTS = {
    "phase137_gate": Path("docs/results/phase137_paper_evidence_refresh/phase137_paper_evidence_refresh_gate.json"),
    "phase116_gate": Path(
        "docs/results/phase116_paper_evidence_consolidation/"
        "phase116_paper_evidence_consolidation_gate.json"
    ),
    "phase116_claim_status": Path(
        "docs/results/phase116_paper_evidence_consolidation/"
        "phase116_manuscript_claim_status_table.csv"
    ),
    "phase116_blockers": Path(
        "docs/results/phase116_paper_evidence_consolidation/phase116_remaining_blocker_table.csv"
    ),
    "phase139_glass_terminal": Path(
        "docs/results/phase139_matbench_glass_focused_review/"
        "phase139_matbench_glass_focused_review_gate.json"
    ),
    "phase140_mp_is_metal_blocker": Path(
        "docs/results/phase140_matbench_mp_is_metal_baseline_gate/"
        "phase140_matbench_mp_is_metal_gate.json"
    ),
    "phase142_expt_is_metal_terminal": Path(
        "docs/results/phase142_matbench_expt_is_metal_focused_review/"
        "phase142_matbench_expt_is_metal_focused_review_gate.json"
    ),
}

TERMINAL_BRANCHES = (
    {
        "branch_id": "matbench_glass",
        "input_key": "phase139_glass_terminal",
        "source_phases": "138-139",
        "dataset": "matbench_glass",
        "target": "gfa",
        "training_lock_key": "phase139_model_training_allowed",
        "final_use": "appendix_negative_external_diagnostic",
        "paper_boundary": "nearest-neighbor identity and class-balance audits block glass model claims",
    },
    {
        "branch_id": "matbench_mp_is_metal_large_source",
        "input_key": "phase140_mp_is_metal_blocker",
        "source_phases": "140",
        "dataset": "matbench_mp_is_metal",
        "target": "is_metal",
        "training_lock_key": "phase140_model_training_allowed",
        "optional_missing_status": "phase140_matbench_mp_is_metal_real_gate_missing_source_acquisition_blocked",
        "final_use": "blocked_source_acquisition_diagnostic",
        "paper_boundary": "large-source HTTPS acquisition blocked before any real gate artifact; not a compute or model result",
    },
    {
        "branch_id": "matbench_expt_is_metal",
        "input_key": "phase142_expt_is_metal_terminal",
        "source_phases": "141-142",
        "dataset": "matbench_expt_is_metal",
        "target": "is_metal",
        "training_lock_key": "phase142_model_training_allowed",
        "final_use": "appendix_negative_external_diagnostic",
        "paper_boundary": "nearest-neighbor identity and class-balance audits block experimental is-metal model claims",
    },
)


def _read_json(path: Path) -> dict[str, Any]:
    return phase137._read_json(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    return phase137._read_csv(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    phase137._write_json(path, payload)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    phase137._write_csv(path, rows, fields)


def _display_path(path: Path, root: Path | None = None) -> str:
    return phase137._display_path(path, root)


def _is_false(value: Any) -> bool:
    return phase137._is_false(value)


def _is_true(value: Any) -> bool:
    return phase137._is_true(value)


def _blocking_audits(gate: dict[str, Any]) -> str:
    return phase137._blocking_audits(gate)


def build_external_diagnostic_rows(
    *, gates: dict[str, dict[str, Any]], input_paths: dict[str, Path], root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in TERMINAL_BRANCHES:
        gate = gates.get(spec["input_key"])
        if gate is None:
            gate = {
                "status": spec.get("optional_missing_status", "missing_required_gate"),
                "selected_target": spec["target"],
                "blocking_audits": ["missing_real_gate_artifact"],
                spec["training_lock_key"]: False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            }
        rows.append(
            {
                "branch_id": spec["branch_id"],
                "source_phases": spec["source_phases"],
                "dataset": spec["dataset"],
                "target": gate.get("selected_target") or spec["target"],
                "terminal_status": gate.get("status"),
                "blocking_audits": _blocking_audits(gate),
                "final_use": spec["final_use"],
                "paper_boundary": spec["paper_boundary"],
                "model_training_allowed": gate.get(spec["training_lock_key"], False),
                "a100_training_allowed_now": gate.get("a100_training_allowed_now", False),
                "a100_80gb_request_now": gate.get("a100_80gb_request_now", False),
                "evidence_source": (
                    _display_path(input_paths[spec["input_key"]], root)
                    if spec["input_key"] in input_paths
                    else "missing_optional_input"
                ),
            }
        )
    return rows


def build_claim_boundary_rows(
    *,
    phase116_gate: dict[str, Any],
    phase137_gate: dict[str, Any],
    external_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    disallowed = ", ".join(row["branch_id"] for row in external_rows)
    return [
        {
            "claim_id": "P143-CLAIM-001",
            "claim_area": "first_paper_main_claim",
            "claim_status": "allowed_narrow_floor",
            "allowed_use": "main_text",
            "wording_guard": (
                "claim only route-guarded fixed-sampling broad12/broad21 spot_size under "
                "broad_process_v1; Phase 138-142 external diagnostics do not expand the main claim"
            ),
            "evidence_anchor": "docs/results/phase116_paper_evidence_consolidation/phase116_positive_floor_table.csv",
        },
        {
            "claim_id": "P143-CLAIM-002",
            "claim_area": "new_external_branches",
            "claim_status": "diagnostic_only",
            "allowed_use": "appendix_or_limitations",
            "wording_guard": f"Phase 138-142 external branches are closed or blocked diagnostics: {disallowed}",
            "evidence_anchor": "docs/results/phase143_paper_evidence_refresh/phase143_external_diagnostic_refresh_table.csv",
        },
        {
            "claim_id": "P143-CLAIM-003",
            "claim_area": "not_allowed_claims",
            "claim_status": "blocked",
            "allowed_use": "explicit_exclusions",
            "wording_guard": (
                "do not claim complete GNN-PINN, general process-condition modeling, "
                "density-invariant robustness, source-path/Green feature success, "
                "microstructure GNN success, or Matbench glass/is-metal model success"
            ),
            "evidence_anchor": "docs/results/phase142_matbench_expt_is_metal_focused_review/phase142_matbench_expt_is_metal_focused_review_gate.json",
        },
        {
            "claim_id": "P143-CLAIM-004",
            "claim_area": "submission_readiness",
            "claim_status": "blocked_missing_venue_benchmark",
            "allowed_use": "planning",
            "wording_guard": "submission readiness still requires target venue and benchmark-paper comparison",
            "evidence_anchor": "docs/results/phase116_paper_evidence_consolidation/phase116_remaining_blocker_table.csv",
        },
        {
            "claim_id": "P143-CLAIM-005",
            "claim_area": "phase137_refresh_status",
            "claim_status": "preserved" if phase137_gate.get("first_paper_draft_allowed_now") else "incomplete",
            "allowed_use": "quality_gate",
            "wording_guard": "Phase 143 cannot strengthen claims beyond the Phase 137/116 floor evidence",
            "evidence_anchor": "docs/results/phase137_paper_evidence_refresh/phase137_paper_evidence_refresh_gate.json",
        },
        {
            "claim_id": "P143-CLAIM-006",
            "claim_area": "phase116_floor_status",
            "claim_status": "preserved" if phase116_gate.get("paper_evidence_consolidated") else "incomplete",
            "allowed_use": "quality_gate",
            "wording_guard": "Phase 143 preserves the existing floor instead of opening new model training",
            "evidence_anchor": "docs/results/phase116_paper_evidence_consolidation/phase116_paper_evidence_consolidation_gate.json",
        },
    ]


def build_decision_rows(
    *,
    phase116_gate: dict[str, Any],
    phase137_gate: dict[str, Any],
    blockers: list[dict[str, str]],
    external_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    submission_blockers = [row for row in blockers if _is_true(row.get("blocks_submission"))]
    all_external_locked = all(
        _is_false(row["model_training_allowed"])
        and _is_false(row["a100_training_allowed_now"])
        and _is_false(row["a100_80gb_request_now"])
        for row in external_rows
    )
    floor_ready = bool(phase116_gate.get("paper_evidence_consolidated")) and bool(
        phase137_gate.get("first_paper_draft_allowed_now")
    )
    return [
        {
            "decision_id": "P143-DECISION-001",
            "route": "first_paper_claim_boundary",
            "decision": "preserve_narrow_claims" if floor_ready else "blocked",
            "rationale": "Phase 138-142 diagnostics do not add a new model claim; current floor remains Phase 55/60/74/91/116 spot_size",
            "blocks_submission": bool(submission_blockers),
            "blocks_model_training": False,
            "next_action": "continue first-paper polishing around the narrow floor or run another no-training baseline-first intake",
            "evidence_anchor": "docs/results/phase143_paper_evidence_refresh/phase143_claim_boundary_refresh_table.csv",
        },
        {
            "decision_id": "P143-DECISION-002",
            "route": "phase138_142_external_training",
            "decision": "blocked",
            "rationale": "latest external branches keep model training and A100 locks false"
            if all_external_locked
            else "one or more latest external gates opened a lock unexpectedly",
            "blocks_submission": False,
            "blocks_model_training": True,
            "next_action": "do not train on Phase 138-142 external diagnostics",
            "evidence_anchor": "docs/results/phase143_paper_evidence_refresh/phase143_external_diagnostic_refresh_table.csv",
        },
        {
            "decision_id": "P143-DECISION-003",
            "route": "a100_sxm4_80gb_request",
            "decision": "blocked",
            "rationale": "no seed-positive branch has produced a measured 40GB memory/runtime blockage",
            "blocks_submission": False,
            "blocks_model_training": False,
            "next_action": "continue using A800 40GB for no-training reviews and small reproductions",
            "evidence_anchor": "docs/results/phase143_paper_evidence_refresh/phase143_external_diagnostic_refresh_table.csv",
        },
    ]


def build_gate(
    *,
    phase116_gate: dict[str, Any],
    phase137_gate: dict[str, Any],
    external_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase116_ready = phase116_gate.get("paper_evidence_consolidated") is True
    phase137_ready = phase137_gate.get("first_paper_draft_allowed_now") is True
    external_complete = len(external_rows) == len(TERMINAL_BRANCHES)
    locks_ok = all(
        _is_false(row["model_training_allowed"])
        and _is_false(row["a100_training_allowed_now"])
        and _is_false(row["a100_80gb_request_now"])
        for row in external_rows
    )
    submission_ready = False
    if phase116_ready and phase137_ready and external_complete and locks_ok:
        status = "phase143_paper_evidence_refresh_ready_first_paper_narrow_claims"
        next_action = (
            "continue first-paper refinement around the route-guarded spot_size floor, "
            "or open a fresh no-training baseline-first source intake; do not train from closed diagnostics"
        )
    else:
        status = "phase143_paper_evidence_refresh_incomplete"
        next_action = "repair missing Phase 116/137 or latest external terminal gate evidence before continuing"
    return {
        "status": status,
        "phase116_paper_evidence_consolidated": phase116_ready,
        "phase137_first_paper_draft_allowed": phase137_ready,
        "latest_external_branches": len(external_rows),
        "latest_external_branches_expected": len(TERMINAL_BRANCHES),
        "latest_external_diagnostics_complete": external_complete,
        "latest_external_training_locks_verified": locks_ok,
        "first_paper_draft_allowed_now": phase116_ready and phase137_ready and external_complete and locks_ok,
        "first_paper_submission_ready": submission_ready,
        "main_paper_floor": phase116_gate.get("main_paper_floor"),
        "new_external_model_claim_ready": False,
        "phase143_model_mechanism_allowed": False,
        "phase143_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "decision_rows": len(decision_rows),
        "blocked_model_training_routes": sum(
            1 for row in decision_rows if _is_true(row.get("blocks_model_training"))
        ),
        "next_action": next_action,
    }


def build_markdown(
    *,
    gate: dict[str, Any],
    external_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Phase 143 Paper Evidence Refresh",
            "",
            f"- Status: `{gate['status']}`",
            f"- First paper draft allowed now: `{gate['first_paper_draft_allowed_now']}`",
            f"- New external model claim ready: `{gate['new_external_model_claim_ready']}`",
            f"- Model training allowed: `{gate['phase143_model_training_allowed']}`",
            f"- A100 80GB request now: `{gate['a100_80gb_request_now']}`",
            "",
            "## Latest External Diagnostics",
            "",
            phase137._markdown_table(
                external_rows,
                [
                    ("Branch", "branch_id"),
                    ("Status", "terminal_status"),
                    ("Final use", "final_use"),
                    ("Boundary", "paper_boundary"),
                ],
            ),
            "",
            "## Claim Boundary",
            "",
            phase137._markdown_table(
                claim_rows,
                [
                    ("Claim", "claim_id"),
                    ("Status", "claim_status"),
                    ("Allowed use", "allowed_use"),
                    ("Guard", "wording_guard"),
                ],
            ),
            "",
            "## Decisions",
            "",
            phase137._markdown_table(
                decision_rows,
                [
                    ("Decision", "decision_id"),
                    ("Route", "route"),
                    ("Outcome", "decision"),
                    ("Next action", "next_action"),
                ],
            ),
        ]
    ) + "\n"


def build_package(*, root: Path, output_dir: Path, phase_inputs: dict[str, Path]) -> dict[str, Any]:
    resolved = {name: path if path.is_absolute() else root / path for name, path in phase_inputs.items()}
    phase116_gate = _read_json(resolved["phase116_gate"])
    phase137_gate = _read_json(resolved["phase137_gate"])
    phase116_blockers = _read_csv(resolved["phase116_blockers"])
    gates: dict[str, dict[str, Any]] = {}
    for spec in TERMINAL_BRANCHES:
        path = resolved.get(spec["input_key"])
        if path is not None and path.exists():
            gates[spec["input_key"]] = _read_json(path)
        elif "optional_missing_status" not in spec:
            raise FileNotFoundError(path or spec["input_key"])
    external_rows = build_external_diagnostic_rows(gates=gates, input_paths=resolved, root=root)
    claim_rows = build_claim_boundary_rows(
        phase116_gate=phase116_gate,
        phase137_gate=phase137_gate,
        external_rows=external_rows,
    )
    decision_rows = build_decision_rows(
        phase116_gate=phase116_gate,
        phase137_gate=phase137_gate,
        blockers=phase116_blockers,
        external_rows=external_rows,
    )
    gate = build_gate(
        phase116_gate=phase116_gate,
        phase137_gate=phase137_gate,
        external_rows=external_rows,
        decision_rows=decision_rows,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    external_path = output_dir / "phase143_external_diagnostic_refresh_table.csv"
    claim_path = output_dir / "phase143_claim_boundary_refresh_table.csv"
    decision_path = output_dir / "phase143_next_decision_table.csv"
    gate_path = output_dir / "phase143_paper_evidence_refresh_gate.json"
    markdown_path = output_dir / "phase143_paper_evidence_refresh.md"
    manifest_path = output_dir / "phase143_paper_evidence_refresh_manifest.json"

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
        "phase": 143,
        "objective": "paper_evidence_refresh_after_phase138_142_external_diagnostics",
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
