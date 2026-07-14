from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase204_amb2022_01_spatial_join_protocol.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase204_spatial_join_protocol", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase203(*, ready: bool = True) -> dict[str, object]:
    return {
        "gate": {
            "status": (
                "phase203_physical_target_intake_ready_phase204_spatial_join_protocol_design"
                if ready
                else "phase203_physical_target_intake_incomplete_or_premature"
            ),
            "phase204_spatial_join_protocol_design_allowed": ready,
            "physical_target_download_allowed": False,
            "physical_target_evaluation_allowed": False,
            "model_training_allowed": False,
        }
    }


def test_phase204_allows_metadata_preview_only_after_freezing_all_mapping_requirements():
    module = _load_module()
    payload = module.build_payload(_phase203())

    assert len(payload["protocol_rows"]) == 6
    assert payload["gate"]["phase205_b8_preview_metadata_intake_allowed"] is True
    assert payload["gate"]["physical_target_download_allowed"] is False
    assert payload["gate"]["coordinate_transform_estimation_allowed"] is False


def test_phase204_blocks_any_premature_transform_claim():
    module = _load_module()
    rows = module.build_protocol_rows()
    rows[0] = deepcopy(rows[0])
    rows[0]["current_state"] = "registered"
    gate = module.build_gate(_phase203(), rows)

    assert gate["phase205_b8_preview_metadata_intake_allowed"] is False
    assert "spatial_transform_claimed_before_evidence" in gate["blocking_audits"]
