#!/usr/bin/env python3
"""Build the Phase 116 paper evidence consolidation package.

Phase 116 does not add experiments, read raw data, or open training. It
consolidates the current paper-facing positive floor with the latest closed
NIST AMMT diagnostics so the manuscript boundary is explicit after Phase 115.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PHASE_INPUTS = {
    "phase60_manifest": Path(
        "docs/results/phase60_manuscript_evidence_package/"
        "phase60_manuscript_evidence_package_manifest.json"
    ),
    "phase74_claim_audit": Path(
        "docs/results/phase74_manuscript_v0_claim_audit/phase74_claim_audit_table.csv"
    ),
    "phase74_boundary_register": Path(
        "docs/results/phase74_manuscript_v0_claim_audit/phase74_model_boundary_register.csv"
    ),
    "phase74_manifest": Path(
        "docs/results/phase74_manuscript_v0_claim_audit/"
        "phase74_manuscript_v0_claim_audit_manifest.json"
    ),
    "phase91_gate": Path(
        "docs/results/phase91_table_figure_appendix_freeze/"
        "phase91_table_figure_appendix_freeze_gate.json"
    ),
    "phase91_main_table": Path(
        "docs/results/phase91_table_figure_appendix_freeze/phase91_main_table_freeze.csv"
    ),
    "phase91_appendix": Path(
        "docs/results/phase91_table_figure_appendix_freeze/"
        "phase91_appendix_diagnostic_freeze.csv"
    ),
    "phase92_gate": Path(
        "docs/results/phase92_benchmark_review_intake/"
        "phase92_benchmark_review_intake_gate.json"
    ),
    "phase92_manual_queue": Path(
        "docs/results/phase92_benchmark_review_intake/phase92_manual_benchmark_queue.csv"
    ),
    "phase115_gate": Path(
        "docs/results/phase115_nist_ammt_diagnostic_closure_package/"
        "phase115_nist_ammt_diagnostic_closure_gate.json"
    ),
    "phase115_claim_use": Path(
        "docs/results/phase115_nist_ammt_diagnostic_closure_package/"
        "phase115_nist_ammt_diagnostic_claim_use_table.csv"
    ),
    "phase115_boundary": Path(
        "docs/results/phase115_nist_ammt_diagnostic_closure_package/"
        "phase115_nist_ammt_diagnostic_boundary_table.csv"
    ),
}

POSITIVE_FIELDS = (
    "floor_id",
    "dataset",
    "split",
    "route",
    "metric",
    "broad_process_v1_mean",
    "broad_process_v1_std",
    "best_strong_baseline",
    "delta_vs_best_strong",
    "n_seeds",
    "claim_anchor",
    "manuscript_use",
    "evidence_source",
)
NEGATIVE_FIELDS = (
    "diagnostic_id",
    "source_phase",
    "branch",
    "status",
    "blocked_item",
    "reason",
    "claim_use",
    "model_training_allowed",
    "a100_training_allowed_now",
    "a100_80gb_request_now",
    "evidence_source",
)
CLAIM_FIELDS = (
    "claim_id",
    "claim_area",
    "status",
    "allowed_use",
    "evidence_anchor",
    "wording_guard",
    "source_phase",
)
BLOCKER_FIELDS = (
    "blocker_id",
    "category",
    "priority",
    "needed_input",
    "blocks_submission",
    "blocks_model_training",
    "next_action",
    "evidence_source",
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


def _boolish(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return value.strip().lower() in {"true", "1", "yes", "y"}


def build_positive_floor_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    positive: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        positive.append(
            {
                "floor_id": f"P116-FLOOR-{index:03d}",
                "dataset": row.get("dataset"),
                "split": row.get("split"),
                "route": row.get("route"),
                "metric": row.get("metric"),
                "broad_process_v1_mean": row.get("broad_process_v1_mean"),
                "broad_process_v1_std": row.get("broad_process_v1_std"),
                "best_strong_baseline": row.get("best_strong_baseline"),
                "delta_vs_best_strong": row.get("delta_vs_best_strong"),
                "n_seeds": row.get("n_seeds"),
                "claim_anchor": row.get("claim_anchor"),
                "manuscript_use": "current_main_text_floor",
                "evidence_source": (
                    "docs/results/phase91_table_figure_appendix_freeze/"
                    "phase91_main_table_freeze.csv"
                ),
            }
        )
    return positive


def build_negative_addendum_rows(
    *,
    appendix_rows: list[dict[str, str]],
    nist_boundary_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(appendix_rows, start=1):
        rows.append(
            {
                "diagnostic_id": f"P116-APPX-{index:03d}",
                "source_phase": row.get("phase"),
                "branch": row.get("branch"),
                "status": row.get("status"),
                "blocked_item": row.get("branch"),
                "reason": row.get("reason"),
                "claim_use": row.get("manuscript_use"),
                "model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
                "evidence_source": row.get("artifact"),
            }
        )
    for index, row in enumerate(nist_boundary_rows, start=1):
        rows.append(
            {
                "diagnostic_id": f"P116-NIST-{index:03d}",
                "source_phase": "115",
                "branch": row.get("branch"),
                "status": "closed_no_training",
                "blocked_item": row.get("blocked_item"),
                "reason": row.get("reason"),
                "claim_use": "appendix_nist_ammt_diagnostic",
                "model_training_allowed": row.get("model_training_allowed", "false"),
                "a100_training_allowed_now": row.get("a100_training_allowed_now", "false"),
                "a100_80gb_request_now": row.get("a100_80gb_request_now", "false"),
                "evidence_source": (
                    "docs/results/phase115_nist_ammt_diagnostic_closure_package/"
                    "phase115_nist_ammt_diagnostic_boundary_table.csv"
                ),
            }
        )
    return rows


def build_claim_status_rows(
    *,
    claim_audit_rows: list[dict[str, str]],
    nist_claim_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in claim_audit_rows:
        allowed = str(row.get("allowed_in_v0", "")).strip().lower() == "yes"
        rows.append(
            {
                "claim_id": row.get("claim_id"),
                "claim_area": row.get("claim_location"),
                "status": row.get("audit_status"),
                "allowed_use": "main_or_appendix_v0" if allowed else "not_allowed_in_v0",
                "evidence_anchor": row.get("evidence_locator"),
                "wording_guard": row.get("required_wording_guard"),
                "source_phase": "74",
            }
        )
    for row in nist_claim_rows:
        rows.append(
            {
                "claim_id": row.get("claim_id"),
                "claim_area": row.get("branch"),
                "status": row.get("evidence_status"),
                "allowed_use": row.get("claim_use"),
                "evidence_anchor": (
                    "docs/results/phase115_nist_ammt_diagnostic_closure_package/"
                    "phase115_nist_ammt_diagnostic_claim_use_table.csv"
                ),
                "wording_guard": "NIST AMMT evidence is appendix/diagnostic only unless a fresh gate passes.",
                "source_phase": "115",
            }
        )
    return rows


def build_blocker_rows(
    *,
    phase92_manual_rows: list[dict[str, str]],
    phase74_boundary_rows: list[dict[str, str]],
    phase115_gate: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in phase92_manual_rows:
        rows.append(
            {
                "blocker_id": row.get("queue_id"),
                "category": row.get("needed_input"),
                "priority": row.get("priority"),
                "needed_input": row.get("minimum_acceptance"),
                "blocks_submission": row.get("blocks_submission"),
                "blocks_model_training": row.get("blocks_model_training"),
                "next_action": row.get("suggested_user_action"),
                "evidence_source": (
                    "docs/results/phase92_benchmark_review_intake/"
                    "phase92_manual_benchmark_queue.csv"
                ),
            }
        )
    for index, row in enumerate(phase74_boundary_rows, start=1):
        rows.append(
            {
                "blocker_id": f"P116-C74-{index:03d}",
                "category": row.get("candidate_or_scope"),
                "priority": "P1",
                "needed_input": row.get("main_text_treatment"),
                "blocks_submission": False,
                "blocks_model_training": "training" in str(row.get("status", "")).lower()
                or "blocked" in str(row.get("status", "")).lower(),
                "next_action": row.get("appendix_treatment"),
                "evidence_source": row.get("evidence_locator"),
            }
        )
    rows.append(
        {
            "blocker_id": "P116-NIST-AMMT-TRAINING-LOCK",
            "category": "NIST AMMT diagnostic branches",
            "priority": "P0",
            "needed_input": "fresh baseline-first registered target/data-source gate",
            "blocks_submission": False,
            "blocks_model_training": True,
            "next_action": phase115_gate.get("next_action"),
            "evidence_source": (
                "docs/results/phase115_nist_ammt_diagnostic_closure_package/"
                "phase115_nist_ammt_diagnostic_closure_gate.json"
            ),
        }
    )
    rows.append(
        {
            "blocker_id": "P116-A100-80GB-LOCK",
            "category": "compute escalation",
            "priority": "P0",
            "needed_input": "measured 40GB blockage on a seed-positive branch",
            "blocks_submission": False,
            "blocks_model_training": False,
            "next_action": "do not request A100-SXM4-80GB from current evidence package",
            "evidence_source": (
                "docs/results/phase115_nist_ammt_diagnostic_closure_package/"
                "phase115_nist_ammt_diagnostic_boundary_table.csv"
            ),
        }
    )
    return rows


def _positive_floor_ready(rows: list[dict[str, Any]]) -> bool:
    datasets = {row.get("dataset") for row in rows}
    metrics = {row.get("metric") for row in rows}
    return (
        len(rows) == 6
        and datasets == {"broad12", "broad21"}
        and metrics == {"Test RMSE", "Hot q90 RMSE", "Gradient q90 RMSE"}
        and {row.get("split") for row in rows} == {"spot_size"}
    )


def _training_locks_ok(
    *,
    phase91_gate: dict[str, Any],
    phase92_gate: dict[str, Any],
    phase115_gate: dict[str, Any],
    negative_rows: list[dict[str, Any]],
) -> bool:
    gate_locks = all(
        [
            phase91_gate.get("a100_training_allowed_now") is False,
            phase91_gate.get("a100_80gb_request_now") is False,
            phase92_gate.get("a100_training_allowed_now") is False,
            phase92_gate.get("a100_80gb_request_now") is False,
            phase115_gate.get("all_training_locks_verified") is True,
            phase115_gate.get("phase115_model_training_allowed") is False,
            phase115_gate.get("a100_training_allowed_now") is False,
            phase115_gate.get("a100_80gb_request_now") is False,
        ]
    )
    row_locks = all(
        _is_false(row.get("model_training_allowed"))
        and _is_false(row.get("a100_training_allowed_now"))
        and _is_false(row.get("a100_80gb_request_now"))
        for row in negative_rows
    )
    return gate_locks and row_locks


def build_gate(
    *,
    positive_rows: list[dict[str, Any]],
    negative_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    blocker_rows: list[dict[str, Any]],
    phase60_manifest: dict[str, Any],
    phase74_manifest: dict[str, Any],
    phase91_gate: dict[str, Any],
    phase92_gate: dict[str, Any],
    phase115_gate: dict[str, Any],
) -> dict[str, Any]:
    positive_ready = _positive_floor_ready(positive_rows)
    phase91_ready = phase91_gate.get("table_figure_appendix_frozen") is True
    phase74_locked = ((phase74_manifest.get("writing_stage_gate") or {}).get("main_claim_locked")) is True
    phase115_closed = (
        phase115_gate.get("status")
        == "phase115_nist_ammt_diagnostic_closure_package_ready_all_new_branches_closed"
        and phase115_gate.get("main_paper_new_nist_ammt_claim_ready") is False
    )
    locks_ok = _training_locks_ok(
        phase91_gate=phase91_gate,
        phase92_gate=phase92_gate,
        phase115_gate=phase115_gate,
        negative_rows=negative_rows,
    )
    venue_unresolved = phase92_gate.get("benchmark_review_ready") is False
    core_ready = positive_ready and phase91_ready and phase74_locked and phase115_closed and locks_ok
    if core_ready and venue_unresolved:
        status = "phase116_paper_evidence_consolidation_ready_venue_unresolved"
        next_action = (
            "use this package for manuscript/appendix drafting, or start a fresh "
            "baseline-first data-source intake; do not train from closed NIST AMMT branches"
        )
    elif core_ready:
        status = "phase116_paper_evidence_consolidation_ready_for_benchmark_review"
        next_action = "run benchmark-paper comparison before venue-specific formatting"
    else:
        status = "phase116_paper_evidence_consolidation_incomplete"
        next_action = "repair missing floor, closure, or training-lock inputs"
    return {
        "status": status,
        "paper_evidence_consolidated": core_ready,
        "positive_floor_ready": positive_ready,
        "phase74_main_claim_locked": phase74_locked,
        "phase91_table_figure_appendix_frozen": phase91_ready,
        "phase92_status": phase92_gate.get("status"),
        "phase92_benchmark_review_ready": phase92_gate.get("benchmark_review_ready"),
        "phase115_nist_ammt_closed": phase115_closed,
        "all_training_locks_verified": locks_ok,
        "main_paper_floor": (phase60_manifest.get("claim_boundary") or {}).get(
            "main_claim",
            "Phase 55/60/74 broad_process_v1 fixed-sampling spot_size",
        ),
        "main_paper_new_nist_ammt_claim_ready": False,
        "submission_ready": False,
        "venue_alignment_ready": False,
        "phase116_model_mechanism_allowed": False,
        "phase116_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "positive_floor_rows": len(positive_rows),
        "negative_diagnostic_rows": len(negative_rows),
        "claim_status_rows": len(claim_rows),
        "remaining_blocker_rows": len(blocker_rows),
        "submission_blocker_rows": sum(1 for row in blocker_rows if _boolish(row.get("blocks_submission"))),
        "model_training_blocker_rows": sum(
            1 for row in blocker_rows if _boolish(row.get("blocks_model_training"))
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
    positive_rows: list[dict[str, Any]],
    negative_rows: list[dict[str, Any]],
    blocker_rows: list[dict[str, Any]],
) -> str:
    nist_rows = [row for row in negative_rows if str(row.get("diagnostic_id", "")).startswith("P116-NIST")]
    top_blockers = [row for row in blocker_rows if _boolish(row.get("blocks_submission"))][:4]
    return "\n".join(
        [
            "# Phase 116 Paper Evidence Consolidation",
            "",
            "## Gate Decision",
            "",
            f"Status: `{gate['status']}`.",
            f"Evidence consolidated: `{str(gate['paper_evidence_consolidated']).lower()}`.",
            f"Submission ready: `{str(gate['submission_ready']).lower()}`.",
            f"Model training allowed: `{str(gate['phase116_model_training_allowed']).lower()}`.",
            f"A100 training allowed now: `{str(gate['a100_training_allowed_now']).lower()}`.",
            f"A100-SXM4-80GB request now: `{str(gate['a100_80gb_request_now']).lower()}`.",
            "",
            "Phase 116 consolidates existing small artifacts only. It does not read raw data, run baselines, or train a model.",
            "",
            "## Current Positive Floor",
            "",
            f"Main paper floor: `{gate['main_paper_floor']}`.",
            "",
            _markdown_table(
                positive_rows,
                [
                    ("dataset", "Dataset"),
                    ("metric", "Metric"),
                    ("broad_process_v1_mean", "broad_process_v1"),
                    ("best_strong_baseline", "Best strong baseline"),
                    ("delta_vs_best_strong", "Delta"),
                    ("n_seeds", "Seeds"),
                ],
            ),
            "",
            "## NIST AMMT Addendum",
            "",
            _markdown_table(
                nist_rows,
                [
                    ("branch", "Branch"),
                    ("blocked_item", "Blocked item"),
                    ("reason", "Reason"),
                    ("claim_use", "Use"),
                ],
            ),
            "",
            "## Submission Blockers",
            "",
            _markdown_table(
                top_blockers,
                [
                    ("blocker_id", "Blocker"),
                    ("priority", "Priority"),
                    ("category", "Category"),
                    ("needed_input", "Needed input"),
                ],
            ),
            "",
            "## Next Action",
            "",
            str(gate["next_action"]),
            "",
        ]
    )


def build_package(*, root: Path, output_dir: Path, phase_inputs: dict[str, Path]) -> dict[str, Any]:
    resolved = {
        name: path if path.is_absolute() else root / path for name, path in phase_inputs.items()
    }
    phase60_manifest = _read_json(resolved["phase60_manifest"])
    phase74_claim_audit = _read_csv(resolved["phase74_claim_audit"])
    phase74_boundary = _read_csv(resolved["phase74_boundary_register"])
    phase74_manifest = _read_json(resolved["phase74_manifest"])
    phase91_gate = _read_json(resolved["phase91_gate"])
    phase91_main = _read_csv(resolved["phase91_main_table"])
    phase91_appendix = _read_csv(resolved["phase91_appendix"])
    phase92_gate = _read_json(resolved["phase92_gate"])
    phase92_manual = _read_csv(resolved["phase92_manual_queue"])
    phase115_gate = _read_json(resolved["phase115_gate"])
    phase115_claims = _read_csv(resolved["phase115_claim_use"])
    phase115_boundary = _read_csv(resolved["phase115_boundary"])

    positive_rows = build_positive_floor_rows(phase91_main)
    negative_rows = build_negative_addendum_rows(
        appendix_rows=phase91_appendix,
        nist_boundary_rows=phase115_boundary,
    )
    claim_rows = build_claim_status_rows(
        claim_audit_rows=phase74_claim_audit,
        nist_claim_rows=phase115_claims,
    )
    blocker_rows = build_blocker_rows(
        phase92_manual_rows=phase92_manual,
        phase74_boundary_rows=phase74_boundary,
        phase115_gate=phase115_gate,
    )
    gate = build_gate(
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        claim_rows=claim_rows,
        blocker_rows=blocker_rows,
        phase60_manifest=phase60_manifest,
        phase74_manifest=phase74_manifest,
        phase91_gate=phase91_gate,
        phase92_gate=phase92_gate,
        phase115_gate=phase115_gate,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    positive_path = output_dir / "phase116_positive_floor_table.csv"
    negative_path = output_dir / "phase116_negative_diagnostic_addendum.csv"
    claim_path = output_dir / "phase116_manuscript_claim_status_table.csv"
    blocker_path = output_dir / "phase116_remaining_blocker_table.csv"
    gate_path = output_dir / "phase116_paper_evidence_consolidation_gate.json"
    markdown_path = output_dir / "phase116_paper_evidence_consolidation.md"
    manifest_path = output_dir / "phase116_paper_evidence_consolidation_manifest.json"

    _write_csv(positive_path, positive_rows, POSITIVE_FIELDS)
    _write_csv(negative_path, negative_rows, NEGATIVE_FIELDS)
    _write_csv(claim_path, claim_rows, CLAIM_FIELDS)
    _write_csv(blocker_path, blocker_rows, BLOCKER_FIELDS)
    _write_json(gate_path, gate)
    markdown_path.write_text(
        build_markdown(
            gate=gate,
            positive_rows=positive_rows,
            negative_rows=negative_rows,
            blocker_rows=blocker_rows,
        ),
        encoding="utf-8",
    )
    manifest = {
        "phase": 116,
        "objective": "paper_evidence_consolidation_after_nist_ammt_closure",
        "inputs": {name: _display_path(path, root) for name, path in sorted(resolved.items())},
        "outputs": {
            "positive_floor_table": _display_path(positive_path, root),
            "negative_diagnostic_addendum": _display_path(negative_path, root),
            "manuscript_claim_status_table": _display_path(claim_path, root),
            "remaining_blocker_table": _display_path(blocker_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "positive_floor_rows": len(positive_rows),
            "negative_diagnostic_rows": len(negative_rows),
            "claim_status_rows": len(claim_rows),
            "remaining_blocker_rows": len(blocker_rows),
            "phase91_appendix_rows": len(phase91_appendix),
            "phase115_boundary_rows": len(phase115_boundary),
            "training_allowed_negative_rows": sum(
                1 for row in negative_rows if not _is_false(row.get("model_training_allowed"))
            ),
            "a100_80gb_allowed_negative_rows": sum(
                1 for row in negative_rows if not _is_false(row.get("a100_80gb_request_now"))
            ),
        },
        "gate": gate,
        "source_statuses": {
            "phase74_writing_stage_gate": (phase74_manifest.get("writing_stage_gate") or {}).get(
                "status"
            ),
            "phase91": phase91_gate.get("status"),
            "phase92": phase92_gate.get("status"),
            "phase115": phase115_gate.get("status"),
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
        default=Path("docs/results/phase116_paper_evidence_consolidation"),
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
