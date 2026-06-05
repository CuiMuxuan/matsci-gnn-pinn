#!/usr/bin/env python3
"""Build Phase 111 NIST AMMT registered-target closure package.

This is a no-training synthesis artifact. It converts Phase 106-110 gates into
claim-use, boundary, and next-action tables for manuscript/appendix handling.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PHASE_INPUTS = {
    "phase106": Path(
        "docs/results/phase106_nist_ammt_spatial_target_representation_gate/"
        "phase106_nist_ammt_spatial_target_gate.json"
    ),
    "phase107": Path(
        "docs/results/phase107_nist_ammt_source_region_feature_gate/"
        "phase107_nist_ammt_source_region_feature_gate.json"
    ),
    "phase108": Path(
        "docs/results/phase108_nist_ammt_sequence_target_gate/"
        "phase108_nist_ammt_sequence_target_gate.json"
    ),
    "phase109": Path(
        "docs/results/phase109_nist_ammt_sequence_target_focused_review_gate/"
        "phase109_nist_ammt_sequence_target_focused_review_gate.json"
    ),
    "phase110": Path(
        "docs/results/phase110_nist_ammt_layer_mean_target_review_gate/"
        "phase110_nist_ammt_layer_mean_target_review_gate.json"
    ),
}

CLAIM_FIELDS = (
    "claim_id",
    "phase",
    "claim_text",
    "claim_use",
    "primary_metric",
    "evidence_status",
)
BOUNDARY_FIELDS = (
    "boundary_id",
    "phase",
    "blocked_item",
    "reason",
    "training_allowed",
    "a100_80gb_request_allowed",
)
EVIDENCE_FIELDS = (
    "phase",
    "status",
    "target_or_profile",
    "validation_metric",
    "test_metric",
    "interpretation",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


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


def _evidence_rows(gates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    phase106 = gates["phase106"]
    phase107 = gates["phase107"]
    phase108 = gates["phase108"]
    phase109 = gates["phase109"]
    phase110 = gates["phase110"]
    return [
        {
            "phase": "106",
            "status": phase106["status"],
            "target_or_profile": phase106.get("selected_target"),
            "validation_metric": phase106.get("selected_validation_rmse"),
            "test_metric": phase106.get("selected_test_rmse"),
            "interpretation": "spatial target gap found; opens review only",
        },
        {
            "phase": "107",
            "status": phase107["status"],
            "target_or_profile": phase107.get("selected_feature_profile"),
            "validation_metric": phase107.get("selected_validation_rmse"),
            "test_metric": phase107.get("selected_test_rmse"),
            "interpretation": "source-region features did not clear Phase 106 guard",
        },
        {
            "phase": "108",
            "status": phase108["status"],
            "target_or_profile": phase108.get("selected_target"),
            "validation_metric": phase108.get("selected_validation_rmse"),
            "test_metric": phase108.get("selected_test_rmse"),
            "interpretation": "sequence target candidate required focused review",
        },
        {
            "phase": "109",
            "status": phase109["status"],
            "target_or_profile": phase109.get("reviewed_target"),
            "validation_metric": phase109.get("full_phase108_validation_rmse"),
            "test_metric": phase109.get("full_phase108_test_rmse"),
            "interpretation": "selected sequence target closed as camera/layer-time shortcut",
        },
        {
            "phase": "110",
            "status": phase110["status"],
            "target_or_profile": phase110.get("reviewed_target"),
            "validation_metric": phase110.get("full_phase108_validation_rmse"),
            "test_metric": phase110.get("full_phase108_test_rmse"),
            "interpretation": "alternate layer-mean target closed as layer/time shortcut",
        },
    ]


def _claim_rows() -> list[dict[str, Any]]:
    return [
        {
            "claim_id": "nist_ammt_registered_target_intake_reproducible",
            "phase": "103-106",
            "claim_text": "NIST AMMT registered source/target intake and spatial target summaries are reproducible as no-training artifacts.",
            "claim_use": "appendix_methods_or_data_diagnostic",
            "primary_metric": "artifact completeness and baseline tables",
            "evidence_status": "allowed_appendix_only",
        },
        {
            "claim_id": "spatial_target_gap_diagnostic",
            "phase": "106",
            "claim_text": "Registered Layer Camera spatial summaries expose a baseline-visible target, but this does not open model training.",
            "claim_use": "appendix_future_work_diagnostic",
            "primary_metric": "target_center_periphery_contrast HGB validation RMSE 1.174314",
            "evidence_status": "diagnostic_only",
        },
        {
            "claim_id": "sequence_target_branch_negative",
            "phase": "108-110",
            "claim_text": "Sequence/camera-pair target candidates collapse under focused shortcut review.",
            "claim_use": "appendix_negative_result",
            "primary_metric": "Phase 109/110 shortcut gates",
            "evidence_status": "closed_negative",
        },
        {
            "claim_id": "main_paper_floor_remains_phase55_60_74",
            "phase": "55/60/74",
            "claim_text": "Main manuscript floor remains broad_process_v1 fixed-sampling spot_size three-seed evidence.",
            "claim_use": "main_text_floor",
            "primary_metric": "existing Phase 55/60/74 route-guarded metrics",
            "evidence_status": "unchanged_positive_floor",
        },
    ]


def _boundary_rows(gates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "boundary_id": "no_training_on_phase106_spatial_gap_alone",
            "phase": "106",
            "blocked_item": "A100 model training on target_center_periphery_contrast",
            "reason": "spatial target gap required mechanism review and Phase 107 source-region gate failed",
            "training_allowed": False,
            "a100_80gb_request_allowed": False,
        },
        {
            "boundary_id": "no_training_on_source_region_features",
            "phase": "107",
            "blocked_item": "sampled source-region feature model branch",
            "reason": gates["phase107"].get("next_action"),
            "training_allowed": False,
            "a100_80gb_request_allowed": False,
        },
        {
            "boundary_id": "no_training_on_camera_pair_delta",
            "phase": "109",
            "blocked_item": "target_cp_camera_pair_delta",
            "reason": "camera/layer-time shortcut matched full validation result",
            "training_allowed": False,
            "a100_80gb_request_allowed": False,
        },
        {
            "boundary_id": "no_training_on_layer_mean_sequence_target",
            "phase": "110",
            "blocked_item": "target_cp_layer_mean",
            "reason": "layer_time_only and layer_time_camera beat full validation; source-only had no independent gain",
            "training_allowed": False,
            "a100_80gb_request_allowed": False,
        },
        {
            "boundary_id": "nist_ammt_sequence_branch_closed",
            "phase": "108-110",
            "blocked_item": "NIST AMMT sequence target branch",
            "reason": "selected and alternate sequence targets collapsed under shortcut review",
            "training_allowed": False,
            "a100_80gb_request_allowed": False,
        },
    ]


def _build_gate(gates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    phase110_closed = (
        gates["phase110"].get("status") == "phase110_layer_mean_target_review_closed_layer_time_shortcut"
    )
    phase109_closed = (
        gates["phase109"].get("status") == "phase109_sequence_target_focused_review_closed_camera_shortcut"
    )
    status = (
        "phase111_registered_target_closure_package_ready_sequence_branch_closed"
        if phase109_closed and phase110_closed
        else "phase111_registered_target_closure_package_incomplete"
    )
    return {
        "status": status,
        "phase106_status": gates["phase106"].get("status"),
        "phase107_status": gates["phase107"].get("status"),
        "phase108_status": gates["phase108"].get("status"),
        "phase109_status": gates["phase109"].get("status"),
        "phase110_status": gates["phase110"].get("status"),
        "nist_ammt_sequence_branch_closed": phase109_closed and phase110_closed,
        "appendix_diagnostic_package_ready": phase109_closed and phase110_closed,
        "main_paper_new_nist_ammt_claim_ready": False,
        "main_paper_floor": "Phase 55/60/74 broad_process_v1 fixed-sampling spot_size",
        "phase111_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": "return to manuscript/appendix packaging or open a new registered target/data source; do not train on Phase 106-110 targets",
    }


def _write_markdown(
    path: Path,
    gate: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Phase 111 NIST AMMT Registered-Target Closure Package",
        "",
        f"- Status: `{gate['status']}`",
        f"- NIST AMMT sequence branch closed: `{gate['nist_ammt_sequence_branch_closed']}`",
        f"- Main paper floor: `{gate['main_paper_floor']}`",
        "- Model training allowed: `false`",
        "- A100 training allowed now: `false`",
        "",
        "## Evidence",
        "",
        "| Phase | Status | Target/Profile | Val metric | Test metric | Interpretation |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in evidence_rows:
        lines.append(
            "| {phase} | {status} | {target_or_profile} | {validation_metric} | {test_metric} | {interpretation} |".format(
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
        lines.append(
            f"| {row['claim_id']} | {row['claim_use']} | {row['evidence_status']} |"
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "| Boundary | Blocked item | Reason |",
            "|---|---|---|",
        ]
    )
    for row in boundary_rows:
        lines.append(f"| {row['boundary_id']} | {row['blocked_item']} | {row['reason']} |")
    lines.extend(["", f"Next action: {gate['next_action']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_package(*, root: Path, output_dir: Path, phase_inputs: dict[str, Path]) -> dict[str, Any]:
    gates = {
        phase: _read_json(path if path.is_absolute() else root / path)
        for phase, path in phase_inputs.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = _evidence_rows(gates)
    claims = _claim_rows()
    boundaries = _boundary_rows(gates)
    gate = _build_gate(gates)

    evidence_path = output_dir / "phase111_nist_ammt_registered_target_evidence_table.csv"
    claims_path = output_dir / "phase111_nist_ammt_registered_target_claim_use_table.csv"
    boundaries_path = output_dir / "phase111_nist_ammt_registered_target_boundary_table.csv"
    gate_path = output_dir / "phase111_nist_ammt_registered_target_closure_gate.json"
    markdown_path = output_dir / "phase111_nist_ammt_registered_target_closure_package.md"
    manifest_path = output_dir / "phase111_nist_ammt_registered_target_closure_manifest.json"
    _write_csv(evidence_path, evidence, EVIDENCE_FIELDS)
    _write_csv(claims_path, claims, CLAIM_FIELDS)
    _write_csv(boundaries_path, boundaries, BOUNDARY_FIELDS)
    _write_json(gate_path, gate)
    _write_markdown(markdown_path, gate, evidence, claims, boundaries)
    manifest = {
        "phase": 111,
        "objective": "nist_ammt_registered_target_closure_package_no_training",
        "inputs": {
            phase: _display_path(path if path.is_absolute() else root / path, root)
            for phase, path in phase_inputs.items()
        },
        "outputs": {
            "evidence_table": _display_path(evidence_path, root),
            "claim_use_table": _display_path(claims_path, root),
            "boundary_table": _display_path(boundaries_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown_package": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "evidence_rows": len(evidence),
            "claim_rows": len(claims),
            "boundary_rows": len(boundaries),
            "training_allowed_boundary_rows": sum(
                1 for row in boundaries if bool(row.get("training_allowed"))
            ),
            "a100_80gb_allowed_boundary_rows": sum(
                1 for row in boundaries if bool(row.get("a100_80gb_request_allowed"))
            ),
        },
        "gate": gate,
        "source_gates": {
            phase: gates[phase].get("status") for phase in sorted(gates)
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
        default=Path("docs/results/phase111_nist_ammt_registered_target_closure_package"),
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
