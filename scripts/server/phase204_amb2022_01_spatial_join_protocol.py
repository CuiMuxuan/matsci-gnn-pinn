#!/usr/bin/env python3
"""Freeze a no-evaluation protocol for joining B8 physical sample planes to nominal build coordinates."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_PHASE203 = Path(
    os.environ.get(
        "AMB2022_01_PHASE203_PHYSICAL_TARGET_INTAKE",
        "/root/matsci-gnn-pinn-ops/phase203_amb2022_01_physical_target_intake.json",
    )
)
PROTOCOL_FIELDS = (
    "mapping_requirement_id",
    "source_coordinate_system",
    "destination_coordinate_system",
    "required_evidence",
    "permitted_operation_after_evidence",
    "current_state",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _phase203_ready(phase203: dict[str, Any]) -> bool:
    gate = phase203.get("gate", {})
    return (
        gate.get("status") == "phase203_physical_target_intake_ready_phase204_spatial_join_protocol_design"
        and bool(gate.get("phase204_spatial_join_protocol_design_allowed"))
        and gate.get("physical_target_download_allowed") is False
        and gate.get("physical_target_evaluation_allowed") is False
        and gate.get("model_training_allowed") is False
    )


def build_protocol_rows() -> list[dict[str, Any]]:
    return [
        {
            "mapping_requirement_id": "build_part_identity",
            "source_coordinate_system": "B8-P3 physical specimen identifier",
            "destination_coordinate_system": "AMB2022-01 nominal B8/P3 build geometry",
            "required_evidence": "Official build/part drawing or record that establishes the P3 specimen location in B8.",
            "permitted_operation_after_evidence": "Associate the specimen with one nominal part only.",
            "current_state": "unresolved",
        },
        {
            "mapping_requirement_id": "leg_identity",
            "source_coordinate_system": "Physical sample label L9",
            "destination_coordinate_system": "Nominal AMB2022-01 leg L9 region",
            "required_evidence": "Official geometry naming convention that distinguishes leg number from layer number.",
            "permitted_operation_after_evidence": "Associate the section with one nominal leg volume only.",
            "current_state": "unresolved",
        },
        {
            "mapping_requirement_id": "section_plane_elevation",
            "source_coordinate_system": "B8-P3-L9 XY section plane",
            "destination_coordinate_system": "Nominal build z/layer index",
            "required_evidence": "Sectioning elevation or registered fiducial that maps the physical XY plane to z and layer spacing.",
            "permitted_operation_after_evidence": "Choose a bounded layer neighborhood, not a single inferred layer by assumption.",
            "current_state": "unresolved",
        },
        {
            "mapping_requirement_id": "image_pixel_calibration",
            "source_coordinate_system": "EBSD/TIFF pixel grid",
            "destination_coordinate_system": "Physical specimen-plane millimeters",
            "required_evidence": "Image metadata or companion data defining pixel scale, origin, and axis orientation.",
            "permitted_operation_after_evidence": "Convert image coordinates to sample-plane millimeters.",
            "current_state": "unresolved",
        },
        {
            "mapping_requirement_id": "rigid_or_affine_registration",
            "source_coordinate_system": "Physical specimen-plane millimeters",
            "destination_coordinate_system": "Nominal B8 XYPT millimeters",
            "required_evidence": "At least three non-collinear shared fiducials or an official registered transformation and residual audit.",
            "permitted_operation_after_evidence": "Create an explicit transform with residuals and uncertainty bounds.",
            "current_state": "unresolved",
        },
        {
            "mapping_requirement_id": "causal_history_window",
            "source_coordinate_system": "Registered nominal layer/position",
            "destination_coordinate_system": "Causal B8 XYPT feature history",
            "required_evidence": "Feature construction that uses no commands after the registered observation position/layer.",
            "permitted_operation_after_evidence": "Construct a causally bounded physical-consistency comparison only.",
            "current_state": "unresolved",
        },
    ]


def build_gate(phase203: dict[str, Any], protocol_rows: list[dict[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    if not _phase203_ready(phase203):
        blockers.append("phase203_physical_target_intake_gate_not_ready")
    required_ids = {
        "build_part_identity",
        "leg_identity",
        "section_plane_elevation",
        "image_pixel_calibration",
        "rigid_or_affine_registration",
        "causal_history_window",
    }
    row_ids = {str(row.get("mapping_requirement_id", "")) for row in protocol_rows}
    if row_ids != required_ids or len(protocol_rows) != len(required_ids):
        blockers.append("spatial_join_requirement_contract_broken")
    if any(row.get("current_state") != "unresolved" for row in protocol_rows):
        blockers.append("spatial_transform_claimed_before_evidence")
    blockers = sorted(set(blockers))
    ready = not blockers
    return {
        "status": (
            "phase204_spatial_join_protocol_ready_phase205_b8_preview_metadata_intake"
            if ready
            else "phase204_spatial_join_protocol_incomplete_or_overclaimed"
        ),
        "phase205_b8_preview_metadata_intake_allowed": ready,
        "physical_target_download_allowed": False,
        "physical_target_evaluation_allowed": False,
        "coordinate_transform_estimation_allowed": False,
        "calibrated_temperature_descriptor_allowed": False,
        "model_training_allowed": False,
        "hyperparameter_search_allowed": False,
        "simulation_as_external_validation_allowed": False,
        "blocking_audits": blockers,
        "next_action": (
            "download one checksum-verified B8-P3-L9 TIFF solely to inspect its metadata; do not use its pixels as a target"
            if ready
            else "repair the spatial-join protocol before touching any physical target component"
        ),
    }


def build_payload(phase203: dict[str, Any]) -> dict[str, Any]:
    rows = build_protocol_rows()
    return {
        "phase": 204,
        "objective": "frozen_b8_physical_sample_plane_to_nominal_build_spatial_join_protocol",
        "protocol_rows": rows,
        "physical_target_boundary": (
            "No affine/rigid transform, layer association, image-derived target, or model comparison is made until all six "
            "evidence requirements are satisfied. The next phase may inspect TIFF metadata only."
        ),
        "gate": build_gate(phase203, rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROTOCOL_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase203", type=Path, default=DEFAULT_PHASE203)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-csv", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(_read_json(args.phase203))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.protocol_csv, payload["protocol_rows"])
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
