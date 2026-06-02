#!/usr/bin/env python3
"""Build the Phase 88 fallback manuscript finalization package.

Phase 88 is reached only after Phase 81-87 cannot open a new model branch. It
locks the current experimental contribution around the Phase 55/60/74
route-guarded Macro PINN floor and records failed candidates as appendix or
future-work evidence. It does not claim submission readiness while literature
or target-venue gaps remain open.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CLAIM_LOCK_FIELDS = (
    "lock_id",
    "claim_scope",
    "status",
    "evidence_locator",
    "manuscript_treatment",
    "allowed_strength",
    "blocker_or_guard",
    "next_action",
)

APPENDIX_FIELDS = (
    "appendix_id",
    "phase",
    "branch",
    "status",
    "artifact",
    "manuscript_use",
    "reason",
)

REMAINING_WORK_FIELDS = (
    "work_id",
    "category",
    "status",
    "required_input_or_gate",
    "blocks_submission",
    "blocks_experimental_claim",
    "next_action",
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
    phase74 = root / "docs/results/phase74_manuscript_v0_claim_audit"
    phase75 = root / "docs/results/phase75_bayesian_inverse_closure_gate"
    phase79 = root / "docs/results/phase79_bounded_spot_size_parameterization_gate"
    phase80 = root / "docs/results/phase80_bounded_spot_size_local_surrogate_gate"
    phase81 = root / "docs/results/phase81_registered_target_intake_gate"
    return {
        "phase60_manifest": phase60 / "phase60_manuscript_evidence_package_manifest.json",
        "phase60_main": phase60 / "phase60_main_spot_size_seed_positive_table.csv",
        "phase60_route": phase60 / "phase60_route_guard_boundary_table.csv",
        "phase60_stress": phase60 / "phase60_stress_boundary_table.csv",
        "phase60_appendix": phase60 / "phase60_appendix_negative_diagnostic_table.csv",
        "phase61_manifest": phase61 / "phase61_manuscript_draft_package_manifest.json",
        "phase61_literature_gaps": phase61 / "phase61_literature_gap_register.csv",
        "phase74_manifest": phase74 / "phase74_manuscript_v0_claim_audit_manifest.json",
        "phase74_claim_audit": phase74 / "phase74_claim_audit_table.csv",
        "phase74_boundary": phase74 / "phase74_model_boundary_register.csv",
        "phase75_manifest": phase75 / "phase75_bayesian_inverse_closure_gate_manifest.json",
        "phase79_manifest": phase79 / "phase79_bounded_spot_size_parameterization_gate_manifest.json",
        "phase80_manifest": phase80 / "phase80_bounded_spot_size_local_surrogate_gate_manifest.json",
        "phase81_manifest": phase81 / "phase81_registered_target_intake_gate_manifest.json",
        "phase81_table": phase81 / "phase81_registered_target_intake_table.csv",
    }


def _status_counts(rows: list[dict[str, Any]], key: str = "status") -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get(key) or "")
        counts[status] = counts.get(status, 0) + 1
    return counts


def build_claim_lock_rows(
    phase60_manifest: dict[str, Any],
    phase74_manifest: dict[str, Any],
    phase75_manifest: dict[str, Any],
    phase80_manifest: dict[str, Any],
    phase81_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    phase60_boundary = phase60_manifest.get("claim_boundary") or {}
    phase74_gate = phase74_manifest.get("writing_stage_gate") or {}
    phase75_gate = phase75_manifest.get("gate_status") or {}
    phase80_gate = phase80_manifest.get("gate") or {}
    phase81_gate = phase81_manifest.get("gate") or {}
    return [
        {
            "lock_id": "P88-MAIN-LOCK",
            "claim_scope": phase60_boundary.get(
                "main_claim",
                "fixed-sampling broad12/broad21 spot_size under broad_process_v1",
            ),
            "status": "locked_experimental_main_claim",
            "evidence_locator": "docs/results/phase60_manuscript_evidence_package/; docs/results/phase74_manuscript_v0_claim_audit/",
            "manuscript_treatment": "main results and methods contribution",
            "allowed_strength": "paper_facing_with_fixed_sampling_scope",
            "blocker_or_guard": "do not claim density-invariant robustness or universal process-axis superiority",
            "next_action": "resolve literature and target-venue gaps before submission polish",
        },
        {
            "lock_id": "P88-ROUTE-GUARD",
            "claim_scope": "route-guard boundary axes and no-process fallback cases",
            "status": "locked_boundary_claim",
            "evidence_locator": "docs/results/phase60_manuscript_evidence_package/phase60_route_guard_boundary_table.csv",
            "manuscript_treatment": "boundary table and limitations",
            "allowed_strength": "boundary",
            "blocker_or_guard": "laser_power, scan_speed, and full process remain route-guard-only where strong baselines dominate",
            "next_action": "keep wording separate from the main spot_size process-conditioning claim",
        },
        {
            "lock_id": "P88-DENSITY",
            "claim_scope": "alternate-density broad21 spot_size stress",
            "status": "locked_limitation",
            "evidence_locator": "docs/results/phase60_manuscript_evidence_package/phase60_stress_boundary_table.csv",
            "manuscript_treatment": "limitations and appendix stress boundary",
            "allowed_strength": "negative_boundary",
            "blocker_or_guard": "Phase 59 selected mean fallback rather than a learnable residual signal",
            "next_action": "do not motivate new A100 training from density failure without a new validation-visible signal",
        },
        {
            "lock_id": "P88-BAYES",
            "claim_scope": "bayesian_inverse_closure_v1",
            "status": str(phase75_gate.get("status") or "blocked"),
            "evidence_locator": "docs/results/phase75_bayesian_inverse_closure_gate/",
            "manuscript_treatment": "appendix diagnostic",
            "allowed_strength": "synthetic_positive_local_ambench_negative",
            "blocker_or_guard": phase75_gate.get("reason"),
            "next_action": "do not run Phase 76 A100 validation from this candidate",
        },
        {
            "lock_id": "P88-SPOT-SURROGATE",
            "claim_scope": "bounded_spot_size_parameterization_v1",
            "status": str(phase80_gate.get("status") or "blocked"),
            "evidence_locator": "docs/results/phase79_bounded_spot_size_parameterization_gate/; docs/results/phase80_bounded_spot_size_local_surrogate_gate/",
            "manuscript_treatment": "appendix diagnostic",
            "allowed_strength": "blocked_local_surrogate",
            "blocker_or_guard": phase80_gate.get("reason"),
            "next_action": "do not run broad12/broad21 A100 training unless a materially stronger local signal appears",
        },
        {
            "lock_id": "P88-REGISTERED-TARGET",
            "claim_scope": "registered target expansion and heat-kernel/Green's-function features",
            "status": str(phase81_gate.get("status") or "blocked"),
            "evidence_locator": "docs/results/phase81_registered_target_intake_gate/",
            "manuscript_treatment": "future work and data limitation",
            "allowed_strength": "blocked_data_intake",
            "blocker_or_guard": phase81_gate.get("reason"),
            "next_action": phase81_gate.get("next_action"),
        },
        {
            "lock_id": "P88-WRITING",
            "claim_scope": "manuscript v0 writing state",
            "status": str(phase74_gate.get("status") or "ready_for_internal_review"),
            "evidence_locator": "docs/results/phase61_manuscript_draft_package/; docs/results/phase74_manuscript_v0_claim_audit/",
            "manuscript_treatment": "internal manuscript base",
            "allowed_strength": "experimental_claim_complete_not_submission_ready",
            "blocker_or_guard": "literature and target-venue gaps remain outside the internal experimental claim package",
            "next_action": "resolve literature gaps, target venue requirements, then polish and format",
        },
    ]


def build_appendix_rows(
    phase60_appendix_rows: list[dict[str, str]],
    phase75_manifest: dict[str, Any],
    phase79_manifest: dict[str, Any],
    phase80_manifest: dict[str, Any],
    phase81_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(phase60_appendix_rows, start=1):
        rows.append(
            {
                "appendix_id": f"P88-APPX-BASE-{index:03d}",
                "phase": row.get("phase"),
                "branch": row.get("branch"),
                "status": row.get("result") or row.get("status"),
                "artifact": row.get("evidence"),
                "manuscript_use": row.get("paper_use"),
                "reason": "Carried forward from Phase 60 appendix negative diagnostic table.",
            }
        )
    phase75_gate = phase75_manifest.get("gate_status") or {}
    phase79_gate = phase79_manifest.get("gate") or {}
    phase80_gate = phase80_manifest.get("gate") or {}
    phase81_gate = phase81_manifest.get("gate") or {}
    rows.extend(
        [
            {
                "appendix_id": "P88-APPX-075",
                "phase": 75,
                "branch": "bayesian_inverse_closure_v1",
                "status": phase75_gate.get("status"),
                "artifact": "docs/results/phase75_bayesian_inverse_closure_gate/",
                "manuscript_use": "appendix diagnostic: synthetic-positive but local AM-Bench-negative",
                "reason": phase75_gate.get("reason"),
            },
            {
                "appendix_id": "P88-APPX-079",
                "phase": 79,
                "branch": "bounded_spot_size_parameterization_v1 safety gate",
                "status": phase79_gate.get("status"),
                "artifact": "docs/results/phase79_bounded_spot_size_parameterization_gate/",
                "manuscript_use": "appendix diagnostic: direct A100 blocked by density debt",
                "reason": phase79_gate.get("reason"),
            },
            {
                "appendix_id": "P88-APPX-080",
                "phase": 80,
                "branch": "bounded_spot_size_parameterization_v1 local surrogate",
                "status": phase80_gate.get("status"),
                "artifact": "docs/results/phase80_bounded_spot_size_local_surrogate_gate/",
                "manuscript_use": "appendix diagnostic: local surrogate below gain threshold",
                "reason": phase80_gate.get("reason"),
            },
            {
                "appendix_id": "P88-APPX-081",
                "phase": 81,
                "branch": "registered target intake",
                "status": phase81_gate.get("status"),
                "artifact": "docs/results/phase81_registered_target_intake_gate/",
                "manuscript_use": "future-work data limitation",
                "reason": phase81_gate.get("reason"),
            },
        ]
    )
    return rows


def build_remaining_work_rows(
    literature_gaps: list[dict[str, str]],
    phase81_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    phase81_gate = phase81_manifest.get("gate") or {}
    rows: list[dict[str, Any]] = [
        {
            "work_id": "P88-WORK-LIT",
            "category": "literature_verification",
            "status": "open" if literature_gaps else "complete",
            "required_input_or_gate": f"{len(literature_gaps)} Phase 61 literature or target-style gaps",
            "blocks_submission": bool(literature_gaps),
            "blocks_experimental_claim": False,
            "next_action": "verify AM-Bench, PINN/process-conditioned modeling, and target-venue sources before final Introduction/Related Work",
        },
        {
            "work_id": "P88-WORK-VENUE",
            "category": "target_venue_alignment",
            "status": "open",
            "required_input_or_gate": "target venue, author guide, or accepted benchmark papers",
            "blocks_submission": True,
            "blocks_experimental_claim": False,
            "next_action": "select venue and align manuscript structure, citation density, and caption style",
        },
        {
            "work_id": "P88-WORK-REGISTERED-DATA",
            "category": "future_registered_target",
            "status": phase81_gate.get("status"),
            "required_input_or_gate": "pad camera-to-galvo registration, aligned single-track scan path, or external public registered-target data card",
            "blocks_submission": False,
            "blocks_experimental_claim": False,
            "next_action": phase81_gate.get("next_action"),
        },
        {
            "work_id": "P88-WORK-A100",
            "category": "compute",
            "status": "blocked_no_training_gate",
            "required_input_or_gate": "Phase 84/85 passing candidate plus measured 40GB blockage for 80GB request",
            "blocks_submission": False,
            "blocks_experimental_claim": False,
            "next_action": "do not request A100-SXM4-80GB now",
        },
    ]
    return rows


def build_gate(
    claim_rows: list[dict[str, Any]],
    remaining_rows: list[dict[str, Any]],
    phase75_manifest: dict[str, Any],
    phase80_manifest: dict[str, Any],
    phase81_manifest: dict[str, Any],
) -> dict[str, Any]:
    phase75_gate = phase75_manifest.get("gate_status") or {}
    phase80_gate = phase80_manifest.get("gate") or {}
    phase81_gate = phase81_manifest.get("gate") or {}
    no_a100_training = (
        not bool(phase75_gate.get("phase76_seed7_allowed"))
        and not bool(phase80_gate.get("a100_seed7_allowed"))
        and not bool(phase81_gate.get("a100_training_allowed_now"))
    )
    open_submission_blockers = [
        row for row in remaining_rows if row["blocks_submission"] and row["status"] != "complete"
    ]
    experimental_claim_complete = no_a100_training and any(
        row["lock_id"] == "P88-MAIN-LOCK" and row["status"] == "locked_experimental_main_claim"
        for row in claim_rows
    )
    if experimental_claim_complete:
        status = "fallback_experimental_claim_complete"
        next_action = "resolve literature/venue gaps, then polish and format the fallback manuscript"
    else:
        status = "fallback_claim_blocked"
        next_action = "repair missing claim-lock inputs before manuscript finalization"
    return {
        "status": status,
        "experimental_claim_complete": experimental_claim_complete,
        "submission_ready": False if open_submission_blockers else experimental_claim_complete,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "open_submission_blockers": len(open_submission_blockers),
        "claim_lock_rows": len(claim_rows),
        "phase75_status": phase75_gate.get("status"),
        "phase80_status": phase80_gate.get("status"),
        "phase81_status": phase81_gate.get("status"),
        "main_claim": "fixed-sampling broad12/broad21 spot_size under broad_process_v1, seeds 7/1/2",
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
    claim_rows: list[dict[str, Any]],
    appendix_rows: list[dict[str, Any]],
    remaining_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Phase 88 Fallback Manuscript Finalization",
            "",
            "## Gate Decision",
            "",
            f"Status: `{gate['status']}`.",
            f"Experimental claim complete: `{str(gate['experimental_claim_complete']).lower()}`.",
            f"Submission ready: `{str(gate['submission_ready']).lower()}`.",
            f"A100 training allowed now: `{str(gate['a100_training_allowed_now']).lower()}`.",
            f"A100-SXM4-80GB request now: `{str(gate['a100_80gb_request_now']).lower()}`.",
            "",
            "The fallback manuscript should be finalized around the existing fixed-sampling `spot_size` floor. It is not submission-ready until literature and target-venue gaps are resolved.",
            "",
            "## Claim Locks",
            "",
            _markdown_table(
                claim_rows,
                [
                    ("lock_id", "Lock"),
                    ("claim_scope", "Scope"),
                    ("status", "Status"),
                    ("manuscript_treatment", "Treatment"),
                    ("blocker_or_guard", "Guard"),
                ],
            ),
            "",
            "## Added Appendix Diagnostics",
            "",
            _markdown_table(
                appendix_rows[-4:],
                [
                    ("appendix_id", "Appendix row"),
                    ("phase", "Phase"),
                    ("branch", "Branch"),
                    ("status", "Status"),
                    ("manuscript_use", "Use"),
                ],
            ),
            "",
            "## Remaining Work",
            "",
            _markdown_table(
                remaining_rows,
                [
                    ("work_id", "Work"),
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
    phase60_manifest = _read_json(resolved["phase60_manifest"])
    phase60_appendix_rows = _read_csv(resolved["phase60_appendix"])
    phase61_lit_gaps = _read_csv(resolved["phase61_literature_gaps"])
    phase74_manifest = _read_json(resolved["phase74_manifest"])
    phase75_manifest = _read_json(resolved["phase75_manifest"])
    phase79_manifest = _read_json(resolved["phase79_manifest"])
    phase80_manifest = _read_json(resolved["phase80_manifest"])
    phase81_manifest = _read_json(resolved["phase81_manifest"])

    claim_rows = build_claim_lock_rows(
        phase60_manifest,
        phase74_manifest,
        phase75_manifest,
        phase80_manifest,
        phase81_manifest,
    )
    appendix_rows = build_appendix_rows(
        phase60_appendix_rows,
        phase75_manifest,
        phase79_manifest,
        phase80_manifest,
        phase81_manifest,
    )
    remaining_rows = build_remaining_work_rows(phase61_lit_gaps, phase81_manifest)
    gate = build_gate(claim_rows, remaining_rows, phase75_manifest, phase80_manifest, phase81_manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    claim_lock_path = output_dir / "phase88_claim_lock_table.csv"
    appendix_path = output_dir / "phase88_appendix_diagnostic_table.csv"
    remaining_path = output_dir / "phase88_remaining_work_table.csv"
    gate_path = output_dir / "phase88_fallback_finalization_gate.json"
    markdown_path = output_dir / "phase88_fallback_manuscript_finalization.md"
    manifest_path = output_dir / "phase88_fallback_manuscript_finalization_manifest.json"

    _write_csv(claim_lock_path, claim_rows, CLAIM_LOCK_FIELDS)
    _write_csv(appendix_path, appendix_rows, APPENDIX_FIELDS)
    _write_csv(remaining_path, remaining_rows, REMAINING_WORK_FIELDS)
    _write_json(gate_path, gate)
    markdown_path.write_text(build_markdown(gate, claim_rows, appendix_rows, remaining_rows), encoding="utf-8")

    manifest = {
        "phase": 88,
        "objective": "fallback_manuscript_finalization",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "claim_lock_table": _display_path(claim_lock_path, root),
            "appendix_diagnostic_table": _display_path(appendix_path, root),
            "remaining_work_table": _display_path(remaining_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "claim_lock_rows": len(claim_rows),
            "appendix_rows": len(appendix_rows),
            "base_phase60_appendix_rows": len(phase60_appendix_rows),
            "remaining_work_rows": len(remaining_rows),
            "remaining_work_status_counts": _status_counts(remaining_rows),
            "literature_gap_rows": len(phase61_lit_gaps),
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
        default=Path("docs/results/phase88_fallback_manuscript_finalization"),
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
