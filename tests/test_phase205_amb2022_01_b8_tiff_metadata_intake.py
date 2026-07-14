from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase205_amb2022_01_b8_tiff_metadata_intake.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase205_b8_tiff_metadata_intake", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase204() -> dict[str, object]:
    return {
        "gate": {
            "status": "phase204_spatial_join_protocol_ready_phase205_b8_preview_metadata_intake",
            "phase205_b8_preview_metadata_intake_allowed": True,
            "physical_target_download_allowed": False,
            "physical_target_evaluation_allowed": False,
            "coordinate_transform_estimation_allowed": False,
            "model_training_allowed": False,
        }
    }


def _metadata(module) -> dict[str, object]:
    return {
        "size_bytes": module.EXPECTED_SIZE_BYTES,
        "sha256": module.EXPECTED_SHA256,
        "page_count": 1,
        "first_page_shape": [512, 512, 3],
        "pixel_data_read": False,
        "physical_target_evaluation_performed": False,
        "coordinate_transform_estimated": False,
    }


def test_phase205_admits_coordinate_evidence_intake_but_not_pixel_evaluation():
    module = _load_module()
    payload = module.build_payload(_phase204(), _metadata(module))

    assert payload["gate"]["phase206_coordinate_evidence_intake_allowed"] is True
    assert payload["gate"]["physical_target_evaluation_allowed"] is False
    assert payload["gate"]["coordinate_transform_estimation_allowed"] is False


def test_phase205_blocks_pixel_read_or_checksum_mismatch():
    module = _load_module()
    metadata = deepcopy(_metadata(module))
    metadata["pixel_data_read"] = True
    metadata["sha256"] = "0" * 64
    gate = module.build_gate(_phase204(), metadata)

    assert gate["phase206_coordinate_evidence_intake_allowed"] is False
    assert "b8_tiff_pixel_boundary_broken" in gate["blocking_audits"]
    assert "b8_tiff_sha256_mismatch" in gate["blocking_audits"]
