from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase206_amb2022_01_coordinate_evidence_intake.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase206_coordinate_evidence_intake", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase205() -> dict[str, object]:
    return {
        "metadata": {
            "metadata_tags": {
                "XResolution": "(96000, 1000)",
                "YResolution": "(96000, 1000)",
                "ResolutionUnit": "2",
            }
        },
        "gate": {
            "status": "phase205_b8_preview_metadata_intake_ready_phase206_coordinate_evidence_intake",
            "phase206_coordinate_evidence_intake_allowed": True,
            "physical_target_evaluation_allowed": False,
            "coordinate_transform_estimation_allowed": False,
            "model_training_allowed": False,
        },
    }


def test_phase206_confirms_identity_but_blocks_transform_and_target_evaluation():
    module = _load_module()
    payload = module.build_payload(_phase205())

    assert payload["coordinate_evidence"]["sample_plane_identity_confirmed"] is True
    assert payload["coordinate_evidence"]["section_layer_index_known"] is False
    assert payload["gate"]["phase207_registration_documentation_resolution_allowed"] is True
    assert payload["gate"]["coordinate_transform_estimation_allowed"] is False
    assert payload["gate"]["physical_target_evaluation_allowed"] is False


def test_phase206_blocks_premature_section_elevation_claim():
    module = _load_module()
    evidence = module.build_evidence(_phase205())
    evidence = deepcopy(evidence)
    evidence["section_elevation_in_nominal_build_mm_known"] = True
    gate = module.build_gate(_phase205(), evidence)

    assert gate["phase207_registration_documentation_resolution_allowed"] is False
    assert "section_elevation_claimed_without_evidence" in gate["blocking_audits"]
