from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase197_amb2022_03_calibration_table_design.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase197_calibration_table_design", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase193() -> dict[str, object]:
    return {
        "gate": {
            "status": "phase193_identifier_join_design_ready_phase194_calibration_protocol_design",
            "phase194_calibration_protocol_design_allowed": True,
            "calibration_fitting_allowed": False,
            "model_training_allowed": False,
        }
    }


def _phase194(settings: list[dict[str, object]], folds: list[dict[str, object]]) -> dict[str, object]:
    return {
        "settings": settings,
        "folds": folds,
        "gate": {
            "status": "phase194_calibration_protocol_design_ready_phase195_thermal_descriptor_extraction_design",
            "phase195_thermal_descriptor_extraction_design_allowed": True,
            "calibration_fitting_allowed": False,
            "model_training_allowed": False,
        },
    }


def _phase196() -> dict[str, object]:
    return {
        "gate": {
            "status": "phase196_raw_descriptor_extraction_ready_phase197_calibration_table_design",
            "phase197_calibration_table_design_allowed": True,
            "calibration_fitting_allowed": False,
            "model_training_allowed": False,
        }
    }


def _fixtures(module):
    descriptor_rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []
    settings: list[dict[str, object]] = []
    for setting_index in range(module.EXPECTED_PROCESS_SETTING_COUNT):
        power = 100.0 + setting_index
        speed = 200.0 + setting_index
        spot = 80.0 + setting_index
        setting_id = f"P{power:g}_V{speed:g}_D{spot:g}"
        condition_ids: list[str] = []
        for replicate_index in range(module.EXPECTED_CONDITIONS_PER_SETTING):
            group_id = f"Line_{setting_index}_{replicate_index}"
            condition_ids.append(group_id)
            descriptor_rows.append(
                {
                    "thermal_group_id": group_id,
                    **{
                        descriptor_id: str(setting_index * 10 + replicate_index + descriptor_index)
                        for descriptor_index, descriptor_id in enumerate(module.RAW_DESCRIPTOR_IDS)
                    },
                }
            )
            condition_rows.append(
                {
                    "thermal_group_id": group_id,
                    "laser_power_W": str(power),
                    "scan_speed_mm_s": str(speed),
                    "spot_size_um": str(spot),
                    "depth_um_mean": str(10 + setting_index + replicate_index),
                    "depth_um_std": "1.0",
                    "width_um_mean": str(20 + setting_index + replicate_index),
                    "width_um_std": "2.0",
                    "n_cross_section_replicates": "2",
                }
            )
        settings.append({"setting_id": setting_id})
    folds = [
        {
            "fold_id": f"leave_setting_out_{setting['setting_id']}",
            "held_out_setting_id": setting["setting_id"],
            "held_out_thermal_group_ids": [
                f"Line_{setting_index}_{replicate_index}"
                for replicate_index in range(module.EXPECTED_CONDITIONS_PER_SETTING)
            ],
            "training_setting_ids": [
                other["setting_id"] for other in settings if other["setting_id"] != setting["setting_id"]
            ],
        }
        for setting_index, setting in enumerate(settings)
    ]
    return descriptor_rows, condition_rows, _phase194(settings, folds)


def test_phase197_builds_exact_21_row_no_fit_calibration_table_and_folds():
    module = _load_module()
    descriptor_rows, condition_rows, phase194 = _fixtures(module)
    calibration_rows, calibration_audit = module.build_calibration_rows(descriptor_rows, condition_rows)
    fold_rows, fold_audit = module.build_fold_rows(phase194, calibration_rows)
    payload = module.build_payload(
        _phase193(), phase194, _phase196(), calibration_rows, calibration_audit, fold_rows, fold_audit
    )

    assert len(calibration_rows) == 21
    assert tuple(calibration_rows[0]) == module.CALIBRATION_FIELDS
    assert len(fold_rows) == 7
    assert payload["gate"]["phase198_baseline_contract_design_allowed"] is True
    assert payload["gate"]["calibration_fitting_allowed"] is False
    assert payload["calibration_audit"]["feature_normalization_fitted"] is False
    assert payload["calibration_audit"]["cross_section_targets_read"] is True


def test_phase197_blocks_missing_descriptor_and_invalid_fold_membership():
    module = _load_module()
    descriptor_rows, condition_rows, phase194 = _fixtures(module)
    incomplete_rows, incomplete_audit = module.build_calibration_rows(descriptor_rows[:-1], condition_rows)
    valid_rows, valid_audit = module.build_calibration_rows(descriptor_rows, condition_rows)
    broken_phase194 = deepcopy(phase194)
    broken_phase194["folds"][0]["held_out_thermal_group_ids"] = ["Line_0_0"]
    valid_folds, valid_fold_audit = module.build_fold_rows(phase194, valid_rows)
    broken_folds, broken_fold_audit = module.build_fold_rows(broken_phase194, valid_rows)

    incomplete_gate = module.build_gate(
        _phase193(), phase194, _phase196(), incomplete_rows, incomplete_audit, valid_folds, valid_fold_audit
    )
    broken_fold_gate = module.build_gate(
        _phase193(), broken_phase194, _phase196(), valid_rows, valid_audit, broken_folds, broken_fold_audit
    )

    assert incomplete_gate["phase198_baseline_contract_design_allowed"] is False
    assert "descriptor_condition_join_incomplete" in incomplete_gate["blocking_audits"]
    assert broken_fold_gate["phase198_baseline_contract_design_allowed"] is False
    assert "leave_setting_out_fold_membership_contract_broken" in broken_fold_gate["blocking_audits"]
