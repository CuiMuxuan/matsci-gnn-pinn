from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase196_amb2022_03_raw_descriptor_extraction.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase196_raw_descriptor_extraction", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase195() -> dict[str, object]:
    return {
        "gate": {
            "status": "phase195_thermal_descriptor_design_ready_phase196_descriptor_extraction",
            "phase196_descriptor_extraction_allowed": True,
            "calibrated_temperature_descriptor_allowed": False,
            "calibration_fitting_allowed": False,
            "model_training_allowed": False,
        }
    }


def _audit(module, *, targets_read: bool = False, schema_mismatch_ids: list[str] | None = None) -> dict[str, object]:
    return {
        "line_signal_group_count": module.EXPECTED_LINE_SIGNAL_GROUP_COUNT,
        "non_line_signal_group_count": module.EXPECTED_NON_LINE_SIGNAL_GROUP_COUNT,
        "non_line_signal_group_ids": [f"Pad_{index:02d}" for index in range(6)],
        "unique_signal_shapes": [list(module.EXPECTED_SIGNAL_SHAPE)],
        "schema_mismatch_ids": schema_mismatch_ids or [],
        "cross_section_targets_read": targets_read,
        "thermalcal_read": False,
    }


def _rows(module) -> list[dict[str, object]]:
    return [
        {
            "thermal_group_id": f"Line_{index:02d}",
            **{descriptor_id: float(index + descriptor_index) for descriptor_index, descriptor_id in enumerate(module.DESCRIPTOR_IDS)},
        }
        for index in range(module.EXPECTED_LINE_SIGNAL_GROUP_COUNT)
    ]


def _linear_quantile(values: np.ndarray, quantile: float) -> float:
    ordered = np.sort(values.reshape(-1))
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return float(ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower]))


def test_phase196_streams_raw_signal_and_matches_full_array_reference():
    module = _load_module()
    signal = np.asarray(
        [
            [[0, 99], [100, 101]],
            [[1, 2], [3, 4095]],
        ],
        dtype=np.uint16,
    )
    descriptors = module.describe_raw_signal(signal, threshold_level=100.0)
    flat = signal.reshape(-1)
    frame_maximums = signal.max(axis=(1, 2))

    assert math.isclose(descriptors["signal_mean_dl"], float(np.mean(flat)))
    assert math.isclose(descriptors["signal_std_dl"], float(np.std(flat)))
    assert descriptors["signal_max_dl"] == 4095.0
    assert math.isclose(descriptors["signal_p99_dl"], _linear_quantile(flat, 0.99))
    assert math.isclose(descriptors["above_threshold_fraction"], 3 / 8)
    assert math.isclose(descriptors["active_frame_fraction"], 1.0)
    assert math.isclose(descriptors["frame_max_mean_dl"], float(np.mean(frame_maximums)))
    assert math.isclose(descriptors["frame_max_std_dl"], float(np.std(frame_maximums)))


def test_phase196_payload_admits_label_free_line_only_descriptor_table():
    module = _load_module()
    payload = module.build_payload(_phase195(), _rows(module), _audit(module))

    assert payload["gate"]["phase197_calibration_table_design_allowed"] is True
    assert payload["gate"]["calibration_fitting_allowed"] is False
    assert payload["gate"]["model_training_allowed"] is False
    assert payload["extraction_audit"]["cross_section_targets_read"] is False
    assert payload["extraction_audit"]["thermalcal_read"] is False
    assert tuple(payload["condition_descriptors"][0]) == module.CSV_FIELDS


def test_phase196_gate_blocks_target_reads_and_schema_mismatch():
    module = _load_module()
    target_gate = module.build_gate(_phase195(), _rows(module), _audit(module, targets_read=True))
    schema_gate = module.build_gate(
        _phase195(), _rows(module), _audit(module, schema_mismatch_ids=["Line_00"])
    )

    assert target_gate["phase197_calibration_table_design_allowed"] is False
    assert "cross_section_target_boundary_broken" in target_gate["blocking_audits"]
    assert schema_gate["phase197_calibration_table_design_allowed"] is False
    assert "raw_signal_schema_contract_broken" in schema_gate["blocking_audits"]
