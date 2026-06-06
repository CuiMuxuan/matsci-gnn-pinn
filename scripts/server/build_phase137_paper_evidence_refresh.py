#!/usr/bin/env python3
"""Build Phase 137 paper evidence refresh after external diagnostics.

This package consumes only existing small artifacts. It refreshes the first
paper claim boundary after Phase 117-136 external data diagnostics, without
reading raw data, running baselines, or opening model training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PHASE_INPUTS = {
    "phase116_gate": Path(
        "docs/results/phase116_paper_evidence_consolidation/"
        "phase116_paper_evidence_consolidation_gate.json"
    ),
    "phase116_positive_floor": Path(
        "docs/results/phase116_paper_evidence_consolidation/phase116_positive_floor_table.csv"
    ),
    "phase116_claim_status": Path(
        "docs/results/phase116_paper_evidence_consolidation/"
        "phase116_manuscript_claim_status_table.csv"
    ),
    "phase116_blockers": Path(
        "docs/results/phase116_paper_evidence_consolidation/phase116_remaining_blocker_table.csv"
    ),
    "battery_failure_terminal": Path(
        "docs/results/phase119_battery_failure_candidate_sweep/"
        "phase119_battery_failure_candidate_sweep_gate.json"
    ),
    "matbench_steels_terminal": Path(
        "docs/results/phase122_matbench_steels_low_capacity_mechanism_gate/"
        "phase122_matbench_steels_low_capacity_mechanism_gate.json"
    ),
    "matbench_expt_gap_terminal": Path(
        "docs/results/phase125_matbench_expt_gap_low_capacity_mechanism_gate/"
        "phase125_matbench_expt_gap_low_capacity_mechanism_gate.json"
    ),
    "matbench_phonons_terminal": Path(
        "docs/results/phase127_matbench_phonons_focused_review/"
        "phase127_matbench_phonons_focused_review_gate.json"
    ),
    "matbench_dielectric_terminal": Path(
        "docs/results/phase128_matbench_dielectric_baseline_gate/"
        "phase128_matbench_dielectric_gate.json"
    ),
    "matbench_log_gvrh_terminal": Path(
        "docs/results/phase134_matbench_log_gvrh_focused_review/"
        "phase134_matbench_log_gvrh_focused_review_gate.json"
    ),
    "matbench_log_kvrh_terminal": Path(
        "docs/results/phase131_matbench_log_kvrh_focused_review/"
        "phase131_matbench_log_kvrh_focused_review_gate.json"
    ),
    "matbench_jdft2d_terminal": Path(
        "docs/results/phase133_matbench_jdft2d_focused_review/"
        "phase133_matbench_jdft2d_focused_review_gate.json"
    ),
    "matbench_perovskites_terminal": Path(
        "docs/results/phase136_matbench_perovskites_focused_review/"
        "phase136_matbench_perovskites_focused_review_gate.json"
    ),
}

TERMINAL_BRANCHES = (
    {
        "branch_id": "battery_failure_databank",
        "input_key": "battery_failure_terminal",
        "source_phases": "117-119",
        "dataset": "Battery Failure Databank",
        "target": "all Phase 117 candidate targets",
        "training_lock_key": "phase119_model_training_allowed",
        "final_use": "appendix_negative_external_diagnostic",
        "paper_boundary": "do not use battery failure targets for model claims",
    },
    {
        "branch_id": "matbench_steels",
        "input_key": "matbench_steels_terminal",
        "source_phases": "120-122",
        "dataset": "matbench_steels",
        "target": "yield_strength_mpa",
        "training_lock_key": "phase122_model_training_allowed",
        "final_use": "appendix_negative_external_diagnostic",
        "paper_boundary": "interpretable low-capacity alloy mechanism did not beat the focused-review guard",
    },
    {
        "branch_id": "matbench_expt_gap",
        "input_key": "matbench_expt_gap_terminal",
        "source_phases": "123-125",
        "dataset": "matbench_expt_gap",
        "target": "gap_expt_ev",
        "training_lock_key": "phase125_model_training_allowed",
        "final_use": "appendix_negative_external_diagnostic",
        "paper_boundary": "low-capacity band-gap mechanism did not beat the focused-review guard",
    },
    {
        "branch_id": "matbench_phonons",
        "input_key": "matbench_phonons_terminal",
        "source_phases": "126-127",
        "dataset": "matbench_phonons",
        "target": "last_phdos_peak",
        "training_lock_key": "phase127_model_training_allowed",
        "final_use": "appendix_negative_external_diagnostic",
        "paper_boundary": "shortcut and target-distribution audits block model claims",
    },
    {
        "branch_id": "matbench_dielectric",
        "input_key": "matbench_dielectric_terminal",
        "source_phases": "128",
        "dataset": "matbench_dielectric",
        "target": "refractive_index_n",
        "training_lock_key": "phase128_model_training_allowed",
        "final_use": "appendix_negative_external_diagnostic",
        "paper_boundary": "baseline-first gate closed by no stable guarded gap",
    },
    {
        "branch_id": "matbench_log_gvrh",
        "input_key": "matbench_log_gvrh_terminal",
        "source_phases": "129/134",
        "dataset": "matbench_log_gvrh",
        "target": "log10_g_vrh",
        "training_lock_key": "phase134_model_training_allowed",
        "final_use": "appendix_negative_external_diagnostic",
        "paper_boundary": "target-distribution imbalance blocks mechanism or training",
    },
    {
        "branch_id": "matbench_log_kvrh",
        "input_key": "matbench_log_kvrh_terminal",
        "source_phases": "130-131",
        "dataset": "matbench_log_kvrh",
        "target": "log10_k_vrh",
        "training_lock_key": "phase131_model_training_allowed",
        "final_use": "appendix_negative_external_diagnostic",
        "paper_boundary": "target-distribution imbalance blocks mechanism or training",
    },
    {
        "branch_id": "matbench_jdft2d",
        "input_key": "matbench_jdft2d_terminal",
        "source_phases": "132-133",
        "dataset": "matbench_jdft2d",
        "target": "exfoliation_en",
        "training_lock_key": "phase133_model_training_allowed",
        "final_use": "appendix_negative_external_diagnostic",
        "paper_boundary": "split sensitivity, shortcut dominance, and target imbalance block model claims",
    },
    {
        "branch_id": "matbench_perovskites",
        "input_key": "matbench_perovskites_terminal",
        "source_phases": "135-136",
        "dataset": "matbench_perovskites",
        "target": "e_form",
        "training_lock_key": "phase136_model_training_allowed",
        "final_use": "appendix_negative_external_diagnostic",
        "paper_boundary": "target-distribution imbalance blocks mechanism or training",
    },
)

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
        return f"{value:.9f}"
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


def _is_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value is False
    return str(value).strip().lower() == "false"


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _blocking_audits(gate: dict[str, Any]) -> str:
    audits = gate.get("blocking_audits")
    if isinstance(audits, list):
        return ";".join(str(item) for item in audits)
    if audits:
        return str(audits)
    if gate.get("blocking_audit_rows"):
        return f"{gate.get('blocking_audit_rows')} blocking audit rows"
    status = str(gate.get("status") or "")
    if "closed" in status or "blocked" in status:
        return gate.get("reason") or gate.get("next_action") or "closed_by_terminal_gate"
    return ""


def build_external_diagnostic_rows(
    *, gates: dict[str, dict[str, Any]], input_paths: dict[str, Path], root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in TERMINAL_BRANCHES:
        gate = gates[spec["input_key"]]
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
                "evidence_source": _display_path(input_paths[spec["input_key"]], root),
            }
        )
    return rows


def build_claim_boundary_rows(
    *,
    phase116_gate: dict[str, Any],
    phase116_claims: list[dict[str, str]],
    external_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    disallowed = ", ".join(row["branch_id"] for row in external_rows)
    return [
        {
            "claim_id": "P137-CLAIM-001",
            "claim_area": "first_paper_main_claim",
            "claim_status": "allowed_narrow_floor",
            "allowed_use": "main_text",
            "wording_guard": (
                "claim only route-guarded fixed-sampling broad12/broad21 spot_size "
                "under broad_process_v1; do not generalize to full GNN-PINN or universal process conditioning"
            ),
            "evidence_anchor": "docs/results/phase116_paper_evidence_consolidation/phase116_positive_floor_table.csv",
        },
        {
            "claim_id": "P137-CLAIM-002",
            "claim_area": "external_data_branches",
            "claim_status": "diagnostic_only",
            "allowed_use": "appendix_or_limitations",
            "wording_guard": f"external branches are closed diagnostics: {disallowed}",
            "evidence_anchor": "docs/results/phase137_paper_evidence_refresh/phase137_external_diagnostic_refresh_table.csv",
        },
        {
            "claim_id": "P137-CLAIM-003",
            "claim_area": "not_allowed_claims",
            "claim_status": "blocked",
            "allowed_use": "explicit_exclusions",
            "wording_guard": (
                "do not claim complete GNN-PINN, general process-condition modeling, "
                "density-invariant robustness, successful source-path/Green features, "
                "or successful microstructure GNN"
            ),
            "evidence_anchor": "docs/results/phase74_manuscript_v0_claim_audit/phase74_model_boundary_register.csv",
        },
        {
            "claim_id": "P137-CLAIM-004",
            "claim_area": "submission_readiness",
            "claim_status": "blocked_missing_venue_benchmark",
            "allowed_use": "planning",
            "wording_guard": "submission readiness still requires target venue and benchmark-paper comparison",
            "evidence_anchor": "docs/results/phase116_paper_evidence_consolidation/phase116_remaining_blocker_table.csv",
        },
        {
            "claim_id": "P137-CLAIM-005",
            "claim_area": "phase116_floor_status",
            "claim_status": "preserved" if phase116_gate.get("paper_evidence_consolidated") else "incomplete",
            "allowed_use": "quality_gate",
            "wording_guard": "Phase 137 cannot strengthen claims beyond Phase 116 floor evidence",
            "evidence_anchor": "docs/results/phase116_paper_evidence_consolidation/phase116_paper_evidence_consolidation_gate.json",
        },
    ]


def build_decision_rows(
    *,
    phase116_gate: dict[str, Any],
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
    return [
        {
            "decision_id": "P137-DECISION-001",
            "route": "first_paper_draft",
            "decision": "allowed_with_narrow_claims" if phase116_gate.get("paper_evidence_consolidated") else "blocked",
            "rationale": "current positive floor remains Phase 55/60/74/91/116 route-guarded spot_size evidence",
            "blocks_submission": bool(submission_blockers),
            "blocks_model_training": False,
            "next_action": "draft or polish first paper around the narrow route-guarded floor; resolve venue/benchmark blockers before submission",
            "evidence_anchor": "docs/results/phase137_paper_evidence_refresh/phase137_claim_boundary_refresh_table.csv",
        },
        {
            "decision_id": "P137-DECISION-002",
            "route": "external_diagnostic_training",
            "decision": "blocked",
            "rationale": "Phase 117-136 terminal external gates all keep model training and A100 locks false"
            if all_external_locked
            else "one or more external terminal gates opened a lock unexpectedly",
            "blocks_submission": False,
            "blocks_model_training": True,
            "next_action": "do not train on closed external diagnostics",
            "evidence_anchor": "docs/results/phase137_paper_evidence_refresh/phase137_external_diagnostic_refresh_table.csv",
        },
        {
            "decision_id": "P137-DECISION-003",
            "route": "new_baseline_first_source",
            "decision": "allowed_no_training_intake_only",
            "rationale": "fresh source intake remains useful only if it starts with provenance, splits, strong baselines, and shortcut guards",
            "blocks_submission": False,
            "blocks_model_training": True,
            "next_action": "open a new baseline-first source only after the active evidence package is closed",
            "evidence_anchor": "task_plan.md",
        },
        {
            "decision_id": "P137-DECISION-004",
            "route": "a100_sxm4_80gb_request",
            "decision": "blocked",
            "rationale": "no seed-positive branch has produced a measured 40GB memory/runtime blockage",
            "blocks_submission": False,
            "blocks_model_training": False,
            "next_action": "continue using A800 40GB for no-training reviews and small reproductions",
            "evidence_anchor": "docs/results/phase137_paper_evidence_refresh/phase137_external_diagnostic_refresh_table.csv",
        },
    ]


def build_gate(
    *,
    phase116_gate: dict[str, Any],
    external_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase116_ready = phase116_gate.get("paper_evidence_consolidated") is True
    external_complete = len(external_rows) == len(TERMINAL_BRANCHES)
    locks_ok = all(
        _is_false(row["model_training_allowed"])
        and _is_false(row["a100_training_allowed_now"])
        and _is_false(row["a100_80gb_request_now"])
        for row in external_rows
    )
    submission_ready = False
    if phase116_ready and external_complete and locks_ok:
        status = "phase137_paper_evidence_refresh_ready_first_paper_narrow_claims"
        next_action = (
            "draft or refine the first paper around the route-guarded spot_size floor, "
            "or open a fresh no-training baseline-first source intake; do not train from closed diagnostics"
        )
    else:
        status = "phase137_paper_evidence_refresh_incomplete"
        next_action = "repair missing Phase 116 or external terminal gate evidence before continuing"
    return {
        "status": status,
        "phase116_paper_evidence_consolidated": phase116_ready,
        "terminal_external_branches": len(external_rows),
        "terminal_external_branches_expected": len(TERMINAL_BRANCHES),
        "external_diagnostics_complete": external_complete,
        "external_training_locks_verified": locks_ok,
        "first_paper_draft_allowed_now": phase116_ready and external_complete and locks_ok,
        "first_paper_submission_ready": submission_ready,
        "main_paper_floor": phase116_gate.get("main_paper_floor"),
        "new_external_model_claim_ready": False,
        "phase137_model_mechanism_allowed": False,
        "phase137_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "decision_rows": len(decision_rows),
        "blocked_model_training_routes": sum(
            1 for row in decision_rows if _is_true(row.get("blocks_model_training"))
        ),
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
    *,
    gate: dict[str, Any],
    external_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Phase 137 Paper Evidence Refresh",
            "",
            "## Gate Decision",
            "",
            f"Status: `{gate['status']}`.",
            f"First-paper draft allowed now: `{str(gate['first_paper_draft_allowed_now']).lower()}`.",
            f"Submission ready: `{str(gate['first_paper_submission_ready']).lower()}`.",
            f"Model training allowed: `{str(gate['phase137_model_training_allowed']).lower()}`.",
            f"A100 training allowed now: `{str(gate['a100_training_allowed_now']).lower()}`.",
            f"A100-SXM4-80GB request now: `{str(gate['a100_80gb_request_now']).lower()}`.",
            "",
            "Phase 137 refreshes claim boundaries from existing small artifacts only. It does not read raw data, run baselines, or train a model.",
            "",
            "## External Diagnostics",
            "",
            _markdown_table(
                external_rows,
                [
                    ("branch_id", "Branch"),
                    ("terminal_status", "Terminal status"),
                    ("blocking_audits", "Blockers"),
                    ("paper_boundary", "Boundary"),
                ],
            ),
            "",
            "## Claim Boundaries",
            "",
            _markdown_table(
                claim_rows,
                [
                    ("claim_id", "Claim"),
                    ("claim_status", "Status"),
                    ("allowed_use", "Use"),
                    ("wording_guard", "Wording guard"),
                ],
            ),
            "",
            "## Next Decisions",
            "",
            _markdown_table(
                decision_rows,
                [
                    ("decision_id", "Decision"),
                    ("route", "Route"),
                    ("decision", "Result"),
                    ("next_action", "Next action"),
                ],
            ),
            "",
        ]
    )


def build_package(*, root: Path, output_dir: Path, phase_inputs: dict[str, Path]) -> dict[str, Any]:
    resolved = {name: path if path.is_absolute() else root / path for name, path in phase_inputs.items()}
    phase116_gate = _read_json(resolved["phase116_gate"])
    phase116_claims = _read_csv(resolved["phase116_claim_status"])
    phase116_blockers = _read_csv(resolved["phase116_blockers"])
    gates = {
        spec["input_key"]: _read_json(resolved[spec["input_key"]])
        for spec in TERMINAL_BRANCHES
    }
    external_rows = build_external_diagnostic_rows(gates=gates, input_paths=resolved, root=root)
    claim_rows = build_claim_boundary_rows(
        phase116_gate=phase116_gate,
        phase116_claims=phase116_claims,
        external_rows=external_rows,
    )
    decision_rows = build_decision_rows(
        phase116_gate=phase116_gate,
        blockers=phase116_blockers,
        external_rows=external_rows,
    )
    gate = build_gate(
        phase116_gate=phase116_gate,
        external_rows=external_rows,
        decision_rows=decision_rows,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    external_path = output_dir / "phase137_external_diagnostic_refresh_table.csv"
    claim_path = output_dir / "phase137_claim_boundary_refresh_table.csv"
    decision_path = output_dir / "phase137_next_decision_table.csv"
    gate_path = output_dir / "phase137_paper_evidence_refresh_gate.json"
    markdown_path = output_dir / "phase137_paper_evidence_refresh.md"
    manifest_path = output_dir / "phase137_paper_evidence_refresh_manifest.json"

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
        "phase": 137,
        "objective": "paper_evidence_refresh_after_phase117_136_external_diagnostics",
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
            "external_diagnostic_rows": len(external_rows),
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase137_paper_evidence_refresh"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(root=root, output_dir=output_dir, phase_inputs=PHASE_INPUTS)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
