#!/usr/bin/env python3
"""Assess B8 TIFF coordinate evidence without estimating a spatial transform."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_PHASE205 = Path(
    os.environ.get(
        "AMB2022_01_PHASE205_B8_TIFF_METADATA",
        "/root/matsci-gnn-pinn-ops/phase205_amb2022_01_b8_tiff_metadata_intake.json",
    )
)
OFFICIAL_MEASUREMENT_DESCRIPTION = {
    "title": "AMB2022-01 Benchmark Measurements and Result Descriptions",
    "url": "https://www.nist.gov/system/files/documents/2022/07/27/AMB2022-01%20Measurement%20and%20Result%20Descriptions_v1.0.pdf",
    "evidence": "The document identifies AMB2022-718-AMMT-B8-P3-L9 as an as-built XY cross section and states that some XY cuts pass through the midplane of L9.",
}
REQUIRED_TRANSFORM_TAGS = ("ModelPixelScaleTag", "ModelTiepointTag", "GeoKeyDirectoryTag", "Orientation")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _phase205_ready(phase205: dict[str, Any]) -> bool:
    gate = phase205.get("gate", {})
    return (
        gate.get("status") == "phase205_b8_preview_metadata_intake_ready_phase206_coordinate_evidence_intake"
        and bool(gate.get("phase206_coordinate_evidence_intake_allowed"))
        and gate.get("physical_target_evaluation_allowed") is False
        and gate.get("coordinate_transform_estimation_allowed") is False
        and gate.get("model_training_allowed") is False
    )


def build_evidence(phase205: dict[str, Any]) -> dict[str, Any]:
    metadata = phase205.get("metadata", {})
    tags = metadata.get("metadata_tags", {})
    missing_transform_tags = [tag for tag in REQUIRED_TRANSFORM_TAGS if tag not in tags]
    return {
        "official_measurement_description": OFFICIAL_MEASUREMENT_DESCRIPTION,
        "sample_plane_identity_confirmed": True,
        "sample_plane_midplane_statement_confirmed": True,
        "tiff_raster_resolution_tags": {
            "XResolution": tags.get("XResolution"),
            "YResolution": tags.get("YResolution"),
            "ResolutionUnit": tags.get("ResolutionUnit"),
        },
        "missing_transform_metadata_tags": missing_transform_tags,
        "tiff_physical_origin_present": False,
        "tiff_nominal_build_transform_present": False,
        "section_elevation_in_nominal_build_mm_known": False,
        "section_layer_index_known": False,
        "coordinate_transform_estimated": False,
        "physical_target_evaluation_performed": False,
        "model_training_performed": False,
    }


def build_gate(phase205: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if not _phase205_ready(phase205):
        blockers.append("phase205_b8_metadata_gate_not_ready")
    if evidence.get("sample_plane_identity_confirmed") is not True:
        blockers.append("b8_sample_plane_identity_not_confirmed")
    if evidence.get("sample_plane_midplane_statement_confirmed") is not True:
        blockers.append("b8_sample_midplane_statement_not_confirmed")
    if not evidence.get("missing_transform_metadata_tags"):
        blockers.append("unexpected_complete_tiff_transform_metadata")
    if evidence.get("tiff_physical_origin_present") is not False:
        blockers.append("tiff_physical_origin_claimed_without_evidence")
    if evidence.get("tiff_nominal_build_transform_present") is not False:
        blockers.append("tiff_nominal_transform_claimed_without_evidence")
    if evidence.get("section_elevation_in_nominal_build_mm_known") is not False:
        blockers.append("section_elevation_claimed_without_evidence")
    if evidence.get("section_layer_index_known") is not False:
        blockers.append("section_layer_claimed_without_evidence")
    if evidence.get("coordinate_transform_estimated") is not False:
        blockers.append("coordinate_transform_estimated_before_evidence")
    if evidence.get("physical_target_evaluation_performed") is not False:
        blockers.append("physical_target_evaluation_before_registration")
    if evidence.get("model_training_performed") is not False:
        blockers.append("model_training_before_registration")
    blockers = sorted(set(blockers))
    ready = not blockers
    return {
        "status": (
            "phase206_coordinate_evidence_intake_complete_transform_blocked"
            if ready
            else "phase206_coordinate_evidence_intake_incomplete_or_overclaimed"
        ),
        "phase207_registration_documentation_resolution_allowed": ready,
        "coordinate_transform_estimation_allowed": False,
        "physical_target_evaluation_allowed": False,
        "calibrated_temperature_descriptor_allowed": False,
        "model_training_allowed": False,
        "hyperparameter_search_allowed": False,
        "simulation_as_external_validation_allowed": False,
        "blocking_audits": blockers,
        "next_action": (
            "obtain an official section elevation, physical origin/orientation, and nominal-build registration artifact before estimating any transform"
            if ready
            else "repair the coordinate evidence intake before making any registration statement"
        ),
    }


def build_payload(phase205: dict[str, Any]) -> dict[str, Any]:
    evidence = build_evidence(phase205)
    return {
        "phase": 206,
        "objective": "b8_physical_sample_coordinate_evidence_intake_without_transform_estimation",
        "coordinate_evidence": evidence,
        "boundary": (
            "A sample label, XY-midplane statement, and raster DPI do not determine a physical-to-nominal-build transform. "
            "No B8 EBSD pixel is used as a target until external registration evidence exists."
        ),
        "gate": build_gate(phase205, evidence),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase205", type=Path, default=DEFAULT_PHASE205)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(_read_json(args.phase205))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
