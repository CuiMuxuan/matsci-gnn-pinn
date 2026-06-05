#!/usr/bin/env python3
"""Build Phase 115 NIST AMMT diagnostic closure package.

This no-training synthesis closes the NIST AMMT branches that were explored
after the Phase 111 registered-target closure: Phase 112/113 Melt Pool Camera
targets and Phase 114 AM G-code strategy source features. It reads only small
gate/review artifacts and keeps all model-training and A100-80GB flags locked.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PHASE_INPUTS = {
    "phase111_gate": Path(
        "docs/results/phase111_nist_ammt_registered_target_closure_package/"
        "phase111_nist_ammt_registered_target_closure_gate.json"
    ),
    "phase112_gate": Path(
        "docs/results/phase112_nist_ammt_melt_pool_target_gate/"
        "phase112_nist_ammt_melt_pool_target_gate.json"
    ),
    "phase113_gate": Path(
        "docs/results/phase113_nist_ammt_melt_pool_focused_review/"
        "phase113_nist_ammt_melt_pool_focused_review_gate.json"
    ),
    "phase113_review_table": Path(
        "docs/results/phase113_nist_ammt_melt_pool_focused_review/"
        "phase113_nist_ammt_melt_pool_focused_review_table.csv"
    ),
    "phase114_gate": Path(
        "docs/results/phase114_nist_ammt_gcode_strategy_source_gate/"
        "phase114_nist_ammt_gcode_strategy_source_gate.json"
    ),
    "phase114_review_table": Path(
        "docs/results/phase114_nist_ammt_gcode_strategy_source_gate/"
        "phase114_nist_ammt_gcode_strategy_target_review_table.csv"
    ),
}

EVIDENCE_FIELDS = (
    "phase",
    "branch",
    "gate_status",
    "row_count",
    "selected_item",
    "validation_metric",
    "test_metric",
    "closure_reason",
    "claim_use",
)
CLAIM_FIELDS = (
    "claim_id",
    "branch",
    "claim_text",
    "claim_use",
    "evidence_status",
)
BOUNDARY_FIELDS = (
    "boundary_id",
    "branch",
    "blocked_item",
    "reason",
    "model_mechanism_allowed",
    "model_training_allowed",
    "a100_training_allowed_now",
    "a100_80gb_request_now",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
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
        return f"{value:.6f}"
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


def _false(payload: dict[str, Any], key: str) -> bool:
    return payload.get(key) is False


def _review_targets(rows: list[dict[str, str]], status_key: str, blocked_status: str) -> list[str]:
    return [row.get("target", "") for row in rows if row.get(status_key) == blocked_status]


def _evidence_rows(
    *,
    phase111: dict[str, Any],
    phase112: dict[str, Any],
    phase113: dict[str, Any],
    phase114: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "phase": "111",
            "branch": "registered_layer_camera_sequence",
            "gate_status": phase111.get("status"),
            "row_count": "",
            "selected_item": "Phase 106-110 closure package",
            "validation_metric": "",
            "test_metric": "",
            "closure_reason": "sequence branch already closed as diagnostic",
            "claim_use": "appendix_diagnostic_package",
        },
        {
            "phase": "112",
            "branch": "melt_pool_camera_target",
            "gate_status": phase112.get("status"),
            "row_count": phase112.get("row_count"),
            "selected_item": phase112.get("selected_target"),
            "validation_metric": phase112.get("selected_validation_rmse"),
            "test_metric": phase112.get("selected_test_rmse"),
            "closure_reason": "opened focused review only; training stayed locked",
            "claim_use": "diagnostic_intermediate",
        },
        {
            "phase": "113",
            "branch": "melt_pool_camera_target",
            "gate_status": phase113.get("status"),
            "row_count": "",
            "selected_item": phase113.get("phase112_selected_target"),
            "validation_metric": phase113.get("phase112_selected_validation_rmse"),
            "test_metric": phase113.get("phase112_selected_test_rmse"),
            "closure_reason": "all Phase 112 candidates reversed versus mean guard on test",
            "claim_use": "appendix_negative_result",
        },
        {
            "phase": "114",
            "branch": "gcode_strategy_source",
            "gate_status": phase114.get("status"),
            "row_count": phase114.get("row_count"),
            "selected_item": phase114.get("selected_target"),
            "validation_metric": phase114.get("selected_validation_rmse"),
            "test_metric": phase114.get("selected_test_rmse"),
            "closure_reason": "no guarded baseline gap beyond XYPT guard and shortcut checks",
            "claim_use": "appendix_negative_result",
        },
    ]


def _claim_rows() -> list[dict[str, Any]]:
    return [
        {
            "claim_id": "nist_ammt_registered_intake_diagnostic_package",
            "branch": "nist_ammt_all_closed_branches",
            "claim_text": "NIST AMMT registered Layer Camera, Melt Pool Camera, and G-code strategy probes are reproducible diagnostics.",
            "claim_use": "appendix_methods_and_negative_results",
            "evidence_status": "allowed_appendix_only",
        },
        {
            "claim_id": "melt_pool_target_branch_closed",
            "branch": "melt_pool_camera_target",
            "claim_text": "Melt Pool Camera target summaries did not survive focused validation/test review.",
            "claim_use": "appendix_negative_result",
            "evidence_status": "closed_negative",
        },
        {
            "claim_id": "gcode_strategy_source_branch_closed",
            "branch": "gcode_strategy_source",
            "claim_text": "G-code strategy features did not create a guarded modeling gap beyond the XYPT source guard.",
            "claim_use": "appendix_negative_result",
            "evidence_status": "closed_negative",
        },
        {
            "claim_id": "main_paper_floor_unchanged",
            "branch": "broad_process_v1_spot_size",
            "claim_text": "The current paper-facing model floor remains Phase 55/60/74 broad_process_v1 fixed-sampling spot_size.",
            "claim_use": "main_text_floor",
            "evidence_status": "unchanged_positive_floor",
        },
    ]


def _boundary_rows(
    *,
    phase112: dict[str, Any],
    phase113: dict[str, Any],
    phase114: dict[str, Any],
    phase113_review_rows: list[dict[str, str]],
    phase114_review_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    reversal_targets = _review_targets(
        phase113_review_rows,
        "focused_review_status",
        "blocked_validation_test_reversal",
    )
    gcode_blocked = [
        f"{row.get('target')}:{row.get('status')}"
        for row in phase114_review_rows
        if not str(row.get("phase114_candidate", "")).lower() == "true"
    ]
    return [
        {
            "boundary_id": "phase115_no_training_on_phase112_melt_pool_targets",
            "branch": "melt_pool_camera_target",
            "blocked_item": ", ".join(reversal_targets) or phase112.get("selected_target"),
            "reason": "Phase 113 focused review found validation/test reversal versus mean guard",
            "model_mechanism_allowed": False,
            "model_training_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
        {
            "boundary_id": "phase115_no_training_on_phase114_gcode_features",
            "branch": "gcode_strategy_source",
            "blocked_item": "; ".join(gcode_blocked) or "G-code strategy features",
            "reason": "Phase 114 found no candidate target after XYPT guard and shortcut checks",
            "model_mechanism_allowed": False,
            "model_training_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
        {
            "boundary_id": "phase115_no_new_nist_ammt_main_claim",
            "branch": "nist_ammt_diagnostics",
            "blocked_item": "new NIST AMMT main-text model claim",
            "reason": "Phase 111, Phase 113, and Phase 114 all close as diagnostic or negative branches",
            "model_mechanism_allowed": False,
            "model_training_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
        {
            "boundary_id": "phase115_no_a100_80gb_request",
            "branch": "compute",
            "blocked_item": "A100-SXM4-80GB escalation",
            "reason": "no seed-positive model branch or 40GB memory/runtime blockage exists",
            "model_mechanism_allowed": False,
            "model_training_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
    ]


def _build_gate(
    *,
    phase111: dict[str, Any],
    phase112: dict[str, Any],
    phase113: dict[str, Any],
    phase114: dict[str, Any],
) -> dict[str, Any]:
    phase111_closed = (
        phase111.get("status")
        == "phase111_registered_target_closure_package_ready_sequence_branch_closed"
    )
    phase113_closed = (
        phase113.get("status")
        == "phase113_melt_pool_focused_review_closed_validation_test_reversal"
    )
    phase114_closed = (
        phase114.get("status")
        == "phase114_gcode_strategy_source_gate_closed_no_guarded_baseline_gap"
    )
    locks_ok = all(
        [
            _false(phase111, "phase111_model_training_allowed"),
            _false(phase111, "a100_training_allowed_now"),
            _false(phase111, "a100_80gb_request_now"),
            _false(phase112, "phase112_model_training_allowed"),
            _false(phase112, "a100_training_allowed_now"),
            _false(phase112, "a100_80gb_request_now"),
            _false(phase113, "phase113_model_training_allowed"),
            _false(phase113, "a100_training_allowed_now"),
            _false(phase113, "a100_80gb_request_now"),
            _false(phase114, "phase114_model_training_allowed"),
            _false(phase114, "a100_training_allowed_now"),
            _false(phase114, "a100_80gb_request_now"),
        ]
    )
    ready = phase111_closed and phase113_closed and phase114_closed and locks_ok
    return {
        "status": "phase115_nist_ammt_diagnostic_closure_package_ready_all_new_branches_closed"
        if ready
        else "phase115_nist_ammt_diagnostic_closure_package_incomplete",
        "phase111_registered_target_branch_closed": phase111_closed,
        "phase112_status": phase112.get("status"),
        "phase113_melt_pool_branch_closed": phase113_closed,
        "phase114_gcode_strategy_branch_closed": phase114_closed,
        "all_training_locks_verified": locks_ok,
        "main_paper_new_nist_ammt_claim_ready": False,
        "main_paper_floor": "Phase 55/60/74 broad_process_v1 fixed-sampling spot_size",
        "phase115_model_mechanism_allowed": False,
        "phase115_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": "do not train on Phase 112-114 NIST AMMT branches; return to manuscript consolidation or a fresh baseline-first data-source intake",
    }


def _write_markdown(
    path: Path,
    gate: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Phase 115 NIST AMMT Diagnostic Closure Package",
        "",
        f"- Status: `{gate['status']}`",
        f"- Main paper floor: `{gate['main_paper_floor']}`",
        "- Model mechanism allowed: `false`",
        "- Model training allowed: `false`",
        "- A100 training allowed now: `false`",
        "",
        "## Evidence",
        "",
        "| Phase | Branch | Status | Row count | Selected item | Closure reason |",
        "|---|---|---|---:|---|---|",
    ]
    for row in evidence_rows:
        lines.append(
            "| {phase} | {branch} | {gate_status} | {row_count} | {selected_item} | {closure_reason} |".format(
                **{key: _csv_value(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Claim Use",
            "",
            "| Claim | Use | Evidence status |",
            "|---|---|---|",
        ]
    )
    for row in claim_rows:
        lines.append(f"| {row['claim_id']} | {row['claim_use']} | {row['evidence_status']} |")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "| Boundary | Branch | Blocked item | Reason |",
            "|---|---|---|---|",
        ]
    )
    for row in boundary_rows:
        lines.append(
            f"| {row['boundary_id']} | {row['branch']} | {row['blocked_item']} | {row['reason']} |"
        )
    lines.extend(["", f"Next action: {gate['next_action']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_package(*, root: Path, output_dir: Path, phase_inputs: dict[str, Path]) -> dict[str, Any]:
    resolved = {
        name: path if path.is_absolute() else root / path
        for name, path in phase_inputs.items()
    }
    phase111 = _read_json(resolved["phase111_gate"])
    phase112 = _read_json(resolved["phase112_gate"])
    phase113 = _read_json(resolved["phase113_gate"])
    phase114 = _read_json(resolved["phase114_gate"])
    phase113_review = _read_csv(resolved["phase113_review_table"])
    phase114_review = _read_csv(resolved["phase114_review_table"])

    evidence = _evidence_rows(
        phase111=phase111,
        phase112=phase112,
        phase113=phase113,
        phase114=phase114,
    )
    claims = _claim_rows()
    boundaries = _boundary_rows(
        phase112=phase112,
        phase113=phase113,
        phase114=phase114,
        phase113_review_rows=phase113_review,
        phase114_review_rows=phase114_review,
    )
    gate = _build_gate(
        phase111=phase111,
        phase112=phase112,
        phase113=phase113,
        phase114=phase114,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "phase115_nist_ammt_diagnostic_evidence_table.csv"
    claim_path = output_dir / "phase115_nist_ammt_diagnostic_claim_use_table.csv"
    boundary_path = output_dir / "phase115_nist_ammt_diagnostic_boundary_table.csv"
    gate_path = output_dir / "phase115_nist_ammt_diagnostic_closure_gate.json"
    markdown_path = output_dir / "phase115_nist_ammt_diagnostic_closure_package.md"
    manifest_path = output_dir / "phase115_nist_ammt_diagnostic_closure_manifest.json"
    _write_csv(evidence_path, evidence, EVIDENCE_FIELDS)
    _write_csv(claim_path, claims, CLAIM_FIELDS)
    _write_csv(boundary_path, boundaries, BOUNDARY_FIELDS)
    _write_json(gate_path, gate)
    _write_markdown(markdown_path, gate, evidence, claims, boundaries)
    manifest = {
        "phase": 115,
        "objective": "nist_ammt_diagnostic_closure_package_no_training",
        "inputs": {name: _display_path(path, root) for name, path in resolved.items()},
        "outputs": {
            "evidence_table": _display_path(evidence_path, root),
            "claim_use_table": _display_path(claim_path, root),
            "boundary_table": _display_path(boundary_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown_package": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "evidence_rows": len(evidence),
            "claim_rows": len(claims),
            "boundary_rows": len(boundaries),
            "phase113_review_rows": len(phase113_review),
            "phase114_review_rows": len(phase114_review),
            "training_allowed_boundary_rows": sum(
                1 for row in boundaries if bool(row.get("model_training_allowed"))
            ),
            "a100_80gb_allowed_boundary_rows": sum(
                1 for row in boundaries if bool(row.get("a100_80gb_request_now"))
            ),
        },
        "gate": gate,
        "source_statuses": {
            "phase111": phase111.get("status"),
            "phase112": phase112.get("status"),
            "phase113": phase113.get("status"),
            "phase114": phase114.get("status"),
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
        default=Path("docs/results/phase115_nist_ammt_diagnostic_closure_package"),
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
