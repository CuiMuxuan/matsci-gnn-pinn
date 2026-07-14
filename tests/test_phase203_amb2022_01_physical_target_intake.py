from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase203_amb2022_01_physical_target_intake.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase203_physical_target_intake", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase202() -> dict[str, object]:
    return {
        "gate": {
            "status": "phase202_formula_contract_complete_temperature_conversion_blocked",
            "phase203_calibration_documentation_resolution_allowed": True,
            "calibrated_temperature_descriptor_allowed": False,
            "model_training_allowed": False,
        }
    }


def _description(module, *, complete: bool = True) -> str:
    ids = [spec["sample_id"] for spec in module.SAMPLE_SPECS]
    return "\n".join(ids if complete else ids[:-1])


def _file_rows() -> list[dict[str, str]]:
    return [
        {
            "file path": "SEM_EBSD_Large_Area_Maps/Legs/LEG 9_XY plane/AMB22-718_B8-P3-LEG 9_XY_as-built_IPF-Z.tif",
            "file_size(bytes)": "30826644",
        }
    ]


def test_phase203_admits_spatial_join_protocol_but_not_target_download():
    module = _load_module()
    payload = module.build_payload(_phase202(), _description(module), _file_rows())

    assert payload["gate"]["phase204_spatial_join_protocol_design_allowed"] is True
    assert payload["gate"]["physical_target_download_allowed"] is False
    assert payload["gate"]["physical_target_evaluation_allowed"] is False
    assert payload["intake_audit"]["b8_p3_leg9_tiff_component_count"] == 1


def test_phase203_blocks_missing_sample_identifier():
    module = _load_module()
    payload = module.build_payload(_phase202(), _description(module, complete=False), _file_rows())

    assert payload["gate"]["phase204_spatial_join_protocol_design_allowed"] is False
    assert "microstructure_sample_identifier_missing_from_description" in payload["gate"]["blocking_audits"]
