#!/usr/bin/env python3
"""Inspect one checksum-verified B8 EBSD TIFF metadata block without reading image pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_PHASE204 = Path(
    os.environ.get(
        "AMB2022_01_PHASE204_SPATIAL_JOIN_PROTOCOL",
        "/root/matsci-gnn-pinn-ops/phase204_amb2022_01_spatial_join_protocol.json",
    )
)
DEFAULT_TIFF = Path(
    os.environ.get(
        "AMB2022_01_B8_P3_L9_IPF_Z_TIFF",
        "/root/matsci-gnn-pinn-data/raw/ambench/2022_external_intake/AMB2022-01-microstructure/"
        "AMB22-718_B8-P3-LEG9_XY_as-built_IPF-Z.tif",
    )
)
EXPECTED_SIZE_BYTES = 30_826_644
EXPECTED_SHA256 = "842d6f93e4b7af4fd6b6f333ec57256bc4446e4ef0c8d68d413f9d8007a37495"
METADATA_TAGS = (
    "ImageWidth",
    "ImageLength",
    "BitsPerSample",
    "SamplesPerPixel",
    "XResolution",
    "YResolution",
    "ResolutionUnit",
    "Orientation",
    "ImageDescription",
    "Software",
    "DateTime",
    "ModelPixelScaleTag",
    "ModelTiepointTag",
    "GeoKeyDirectoryTag",
)


def _tifffile() -> Any:
    try:
        import tifffile
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("tifffile is required for Phase 205") from exc
    return tifffile


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tag_value(tag: Any) -> str:
    value = tag.value
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value)
    return text if len(text) <= 500 else f"{text[:500]}..."


def inspect_tiff_metadata(path: Path) -> dict[str, Any]:
    tifffile = _tifffile()
    with tifffile.TiffFile(path) as handle:
        page = handle.pages[0]
        tags = {
            name: _tag_value(page.tags[name])
            for name in METADATA_TAGS
            if name in page.tags
        }
        return {
            "file_name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "page_count": len(handle.pages),
            "first_page_shape": [int(value) for value in page.shape],
            "first_page_dtype": str(page.dtype),
            "metadata_tags": tags,
            "pixel_data_read": False,
            "physical_target_evaluation_performed": False,
            "coordinate_transform_estimated": False,
        }


def _phase204_ready(phase204: dict[str, Any]) -> bool:
    gate = phase204.get("gate", {})
    return (
        gate.get("status") == "phase204_spatial_join_protocol_ready_phase205_b8_preview_metadata_intake"
        and bool(gate.get("phase205_b8_preview_metadata_intake_allowed"))
        and gate.get("physical_target_download_allowed") is False
        and gate.get("physical_target_evaluation_allowed") is False
        and gate.get("coordinate_transform_estimation_allowed") is False
        and gate.get("model_training_allowed") is False
    )


def build_gate(phase204: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if not _phase204_ready(phase204):
        blockers.append("phase204_spatial_join_protocol_gate_not_ready")
    if int(metadata.get("size_bytes", 0)) != EXPECTED_SIZE_BYTES:
        blockers.append("b8_tiff_size_mismatch")
    if str(metadata.get("sha256", "")).lower() != EXPECTED_SHA256:
        blockers.append("b8_tiff_sha256_mismatch")
    if int(metadata.get("page_count", 0)) < 1 or not metadata.get("first_page_shape"):
        blockers.append("b8_tiff_page_metadata_missing")
    if metadata.get("pixel_data_read") is not False:
        blockers.append("b8_tiff_pixel_boundary_broken")
    if metadata.get("physical_target_evaluation_performed") is not False:
        blockers.append("physical_target_evaluation_before_spatial_join")
    if metadata.get("coordinate_transform_estimated") is not False:
        blockers.append("coordinate_transform_estimated_before_evidence")
    blockers = sorted(set(blockers))
    ready = not blockers
    return {
        "status": (
            "phase205_b8_preview_metadata_intake_ready_phase206_coordinate_evidence_intake"
            if ready
            else "phase205_b8_preview_metadata_intake_incomplete_or_premature"
        ),
        "phase206_coordinate_evidence_intake_allowed": ready,
        "physical_target_evaluation_allowed": False,
        "coordinate_transform_estimation_allowed": False,
        "calibrated_temperature_descriptor_allowed": False,
        "model_training_allowed": False,
        "hyperparameter_search_allowed": False,
        "simulation_as_external_validation_allowed": False,
        "blocking_audits": blockers,
        "next_action": (
            "use only documented pixel-scale/orientation metadata and official specimen geometry to assess whether coordinate evidence can be completed"
            if ready
            else "repair checksum or metadata-only boundary before any coordinate-evidence intake"
        ),
    }


def build_payload(phase204: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": 205,
        "objective": "checksum_verified_b8_p3_l9_ebsd_tiff_metadata_intake_without_pixel_analysis",
        "metadata": metadata,
        "coordinate_boundary": (
            "TIFF tags can establish image metadata only. They do not establish the section elevation, nominal build origin, "
            "or a physical-to-XYPT transform."
        ),
        "gate": build_gate(phase204, metadata),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase204", type=Path, default=DEFAULT_PHASE204)
    parser.add_argument("--tiff", type=Path, default=DEFAULT_TIFF)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(_read_json(args.phase204), inspect_tiff_metadata(args.tiff))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
