#!/usr/bin/env python3
"""Build the audited no-fit AMB2022-03 calibration table and grouped folds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_PHASE193 = Path(
    os.environ.get(
        "AMB2022_03_PHASE193_IDENTIFIER_JOIN",
        "/root/matsci-gnn-pinn-ops/phase193_amb2022_03_identifier_join_design.json",
    )
)
DEFAULT_PHASE194 = Path(
    os.environ.get(
        "AMB2022_03_PHASE194_CALIBRATION_PROTOCOL",
        "/root/matsci-gnn-pinn-ops/phase194_amb2022_03_calibration_protocol.json",
    )
)
DEFAULT_PHASE196 = Path(
    os.environ.get(
        "AMB2022_03_PHASE196_RAW_DESCRIPTOR_EXTRACTION",
        "/root/matsci-gnn-pinn-ops/phase196_amb2022_03_raw_descriptor_extraction.json",
    )
)
DEFAULT_CONDITIONS = Path(
    os.environ.get(
        "AMB2022_03_PHASE193_CONDITIONS",
        "/root/matsci-gnn-pinn-ops/phase193_amb2022_03_single_track_conditions.csv",
    )
)
DEFAULT_DESCRIPTORS = Path(
    os.environ.get(
        "AMB2022_03_PHASE196_RAW_DESCRIPTORS",
        "/root/matsci-gnn-pinn-ops/phase196_amb2022_03_raw_thermal_descriptors.csv",
    )
)
EXPECTED_CONDITION_COUNT = 21
EXPECTED_PROCESS_SETTING_COUNT = 7
EXPECTED_CONDITIONS_PER_SETTING = 3
RAW_DESCRIPTOR_IDS = (
    "signal_mean_dl",
    "signal_std_dl",
    "signal_max_dl",
    "signal_p99_dl",
    "above_threshold_fraction",
    "active_frame_fraction",
    "frame_max_mean_dl",
    "frame_max_std_dl",
)
PROCESS_FIELDS = ("laser_power_W", "scan_speed_mm_s", "spot_size_um")
TARGET_FIELDS = ("depth_um_mean", "depth_um_std", "width_um_mean", "width_um_std")
CALIBRATION_FIELDS = (
    "thermal_group_id",
    "setting_id",
    *PROCESS_FIELDS,
    *RAW_DESCRIPTOR_IDS,
    *TARGET_FIELDS,
    "n_cross_section_replicates",
)
FOLD_FIELDS = (
    "fold_id",
    "held_out_setting_id",
    "held_out_condition_count",
    "held_out_thermal_group_ids",
    "training_setting_ids",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _setting_id(row: dict[str, Any]) -> str:
    return (
        f"P{float(row['laser_power_W']):g}_V{float(row['scan_speed_mm_s']):g}_"
        f"D{float(row['spot_size_um']):g}"
    )


def _index_by_group(rows: list[dict[str, Any]], label: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for row in rows:
        group_id = str(row.get("thermal_group_id", ""))
        if group_id in indexed:
            duplicate_ids.append(group_id)
        else:
            indexed[group_id] = row
    return indexed, sorted(set(duplicate_ids))


def _phase193_ready(phase193: dict[str, Any]) -> bool:
    gate = phase193.get("gate", {})
    return (
        gate.get("status") == "phase193_identifier_join_design_ready_phase194_calibration_protocol_design"
        and bool(gate.get("phase194_calibration_protocol_design_allowed"))
        and gate.get("calibration_fitting_allowed") is False
        and gate.get("model_training_allowed") is False
    )


def _phase194_ready(phase194: dict[str, Any]) -> bool:
    gate = phase194.get("gate", {})
    return (
        gate.get("status") == "phase194_calibration_protocol_design_ready_phase195_thermal_descriptor_extraction_design"
        and bool(gate.get("phase195_thermal_descriptor_extraction_design_allowed"))
        and gate.get("calibration_fitting_allowed") is False
        and gate.get("model_training_allowed") is False
    )


def _phase196_ready(phase196: dict[str, Any]) -> bool:
    gate = phase196.get("gate", {})
    return (
        gate.get("status") == "phase196_raw_descriptor_extraction_ready_phase197_calibration_table_design"
        and bool(gate.get("phase197_calibration_table_design_allowed"))
        and gate.get("calibration_fitting_allowed") is False
        and gate.get("model_training_allowed") is False
    )


def build_calibration_rows(
    descriptor_rows: list[dict[str, Any]], condition_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    descriptor_by_group, duplicate_descriptor_ids = _index_by_group(descriptor_rows, "descriptor")
    condition_by_group, duplicate_condition_ids = _index_by_group(condition_rows, "condition")
    descriptor_ids = set(descriptor_by_group)
    condition_ids = set(condition_by_group)
    descriptor_fields = tuple(descriptor_rows[0].keys()) if descriptor_rows else ()
    descriptor_field_contract_ok = descriptor_fields == ("thermal_group_id", *RAW_DESCRIPTOR_IDS)
    rows: list[dict[str, Any]] = []
    for group_id in sorted(descriptor_ids & condition_ids):
        descriptor = descriptor_by_group[group_id]
        condition = condition_by_group[group_id]
        row = {
            "thermal_group_id": group_id,
            "setting_id": _setting_id(condition),
            **{field: float(condition[field]) for field in PROCESS_FIELDS},
            **{field: float(descriptor[field]) for field in RAW_DESCRIPTOR_IDS},
            **{field: float(condition[field]) for field in TARGET_FIELDS},
            "n_cross_section_replicates": int(condition["n_cross_section_replicates"]),
        }
        rows.append(row)
    target_rows_finite = all(
        all(math.isfinite(float(row[field])) for field in TARGET_FIELDS) for row in rows
    )
    descriptor_rows_finite = all(
        all(math.isfinite(float(row[field])) for field in RAW_DESCRIPTOR_IDS) for row in rows
    )
    setting_counts = Counter(str(row["setting_id"]) for row in rows)
    audit = {
        "descriptor_row_count": len(descriptor_rows),
        "condition_row_count": len(condition_rows),
        "joined_condition_count": len(rows),
        "missing_descriptor_for_condition_ids": sorted(condition_ids - descriptor_ids),
        "descriptor_ids_without_condition": sorted(descriptor_ids - condition_ids),
        "duplicate_descriptor_ids": duplicate_descriptor_ids,
        "duplicate_condition_ids": duplicate_condition_ids,
        "descriptor_field_contract_ok": descriptor_field_contract_ok,
        "descriptor_rows_finite": descriptor_rows_finite,
        "target_rows_finite": target_rows_finite,
        "cross_section_replicates_by_condition": {
            str(row["thermal_group_id"]): int(row["n_cross_section_replicates"]) for row in rows
        },
        "setting_counts": dict(sorted(setting_counts.items())),
        "cross_section_targets_read": True,
        "target_source_boundary": "Phase 193 per-condition summary CSV only; the raw workbook is not reopened.",
        "descriptor_source_boundary": "Phase 196 frozen raw-descriptor CSV only; raw HDF5 is not reopened.",
        "feature_normalization_fitted": False,
        "calibration_fitting_performed": False,
        "model_training_performed": False,
    }
    return rows, audit


def build_fold_rows(
    phase194: dict[str, Any], calibration_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ids_by_setting: dict[str, list[str]] = {}
    for row in calibration_rows:
        ids_by_setting.setdefault(str(row["setting_id"]), []).append(str(row["thermal_group_id"]))
    ids_by_setting = {setting_id: sorted(ids) for setting_id, ids in ids_by_setting.items()}
    expected_setting_ids = {str(row["setting_id"]) for row in phase194.get("settings", [])}
    rows: list[dict[str, Any]] = []
    invalid_fold_ids: list[str] = []
    for fold in sorted(phase194.get("folds", []), key=lambda row: str(row.get("fold_id", ""))):
        fold_id = str(fold.get("fold_id", ""))
        held_out_setting_id = str(fold.get("held_out_setting_id", ""))
        held_out_ids = sorted(str(value) for value in fold.get("held_out_thermal_group_ids", []))
        training_setting_ids = sorted(str(value) for value in fold.get("training_setting_ids", []))
        expected_held_out_ids = ids_by_setting.get(held_out_setting_id, [])
        if (
            held_out_setting_id not in expected_setting_ids
            or held_out_ids != expected_held_out_ids
            or set(training_setting_ids) != expected_setting_ids - {held_out_setting_id}
        ):
            invalid_fold_ids.append(fold_id)
        rows.append(
            {
                "fold_id": fold_id,
                "held_out_setting_id": held_out_setting_id,
                "held_out_condition_count": len(held_out_ids),
                "held_out_thermal_group_ids": ";".join(held_out_ids),
                "training_setting_ids": ";".join(training_setting_ids),
            }
        )
    audit = {
        "expected_setting_ids": sorted(expected_setting_ids),
        "calibration_setting_ids": sorted(ids_by_setting),
        "fold_count": len(rows),
        "invalid_fold_ids": sorted(set(invalid_fold_ids)),
    }
    return rows, audit


def build_gate(
    phase193: dict[str, Any],
    phase194: dict[str, Any],
    phase196: dict[str, Any],
    calibration_rows: list[dict[str, Any]],
    calibration_audit: dict[str, Any],
    fold_rows: list[dict[str, Any]],
    fold_audit: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    if not _phase193_ready(phase193):
        blockers.append("phase193_identifier_join_gate_not_ready")
    if not _phase194_ready(phase194):
        blockers.append("phase194_calibration_protocol_gate_not_ready")
    if not _phase196_ready(phase196):
        blockers.append("phase196_raw_descriptor_gate_not_ready")
    if int(calibration_audit.get("descriptor_row_count", 0)) != EXPECTED_CONDITION_COUNT:
        blockers.append("unexpected_descriptor_row_count")
    if int(calibration_audit.get("condition_row_count", 0)) != EXPECTED_CONDITION_COUNT:
        blockers.append("unexpected_condition_row_count")
    if int(calibration_audit.get("joined_condition_count", 0)) != EXPECTED_CONDITION_COUNT:
        blockers.append("descriptor_condition_join_incomplete")
    if calibration_audit.get("missing_descriptor_for_condition_ids"):
        blockers.append("condition_missing_phase196_descriptor")
    if calibration_audit.get("descriptor_ids_without_condition"):
        blockers.append("phase196_descriptor_without_condition")
    if calibration_audit.get("duplicate_descriptor_ids") or calibration_audit.get("duplicate_condition_ids"):
        blockers.append("duplicate_identifier_in_calibration_join")
    if calibration_audit.get("descriptor_field_contract_ok") is not True:
        blockers.append("raw_descriptor_field_contract_broken")
    if calibration_audit.get("descriptor_rows_finite") is not True:
        blockers.append("nonfinite_raw_descriptor")
    if calibration_audit.get("target_rows_finite") is not True:
        blockers.append("nonfinite_cross_section_target")
    replicate_counts = calibration_audit.get("cross_section_replicates_by_condition", {}).values()
    if any(int(value) != 2 for value in replicate_counts) or len(list(replicate_counts)) != EXPECTED_CONDITION_COUNT:
        blockers.append("cross_section_replicate_contract_not_two")
    setting_counts = calibration_audit.get("setting_counts", {})
    if len(setting_counts) != EXPECTED_PROCESS_SETTING_COUNT or any(
        int(value) != EXPECTED_CONDITIONS_PER_SETTING for value in setting_counts.values()
    ):
        blockers.append("process_setting_replicate_contract_broken")
    if calibration_audit.get("feature_normalization_fitted") is not False:
        blockers.append("pre_fold_feature_normalization_detected")
    if calibration_audit.get("calibration_fitting_performed") is not False:
        blockers.append("calibration_fit_performed_before_contract")
    if calibration_audit.get("model_training_performed") is not False:
        blockers.append("model_training_performed_before_contract")
    if int(fold_audit.get("fold_count", 0)) != EXPECTED_PROCESS_SETTING_COUNT:
        blockers.append("leave_setting_out_fold_count")
    if fold_audit.get("invalid_fold_ids"):
        blockers.append("leave_setting_out_fold_membership_contract_broken")
    if any(int(row["held_out_condition_count"]) != EXPECTED_CONDITIONS_PER_SETTING for row in fold_rows):
        blockers.append("leave_setting_out_heldout_count_contract_broken")
    blockers = sorted(set(blockers))
    ready = not blockers
    return {
        "status": (
            "phase197_calibration_table_design_ready_phase198_baseline_contract_design"
            if ready
            else "phase197_calibration_table_design_incomplete_or_leaky"
        ),
        "phase198_baseline_contract_design_allowed": ready,
        "calibration_fitting_allowed": False,
        "model_training_allowed": False,
        "hyperparameter_search_allowed": False,
        "post_b8_model_reselection_allowed": False,
        "descriptor_target_leakage_allowed": False,
        "simulation_as_external_validation_allowed": False,
        "blocking_audits": blockers,
        "next_action": (
            "freeze a low-capacity leave-one-setting-out baseline and uncertainty contract before fitting"
            if ready
            else "repair identifier, fold, or target-boundary audits before any calibration fit"
        ),
    }


def build_payload(
    phase193: dict[str, Any],
    phase194: dict[str, Any],
    phase196: dict[str, Any],
    calibration_rows: list[dict[str, Any]],
    calibration_audit: dict[str, Any],
    fold_rows: list[dict[str, Any]],
    fold_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": 197,
        "objective": "deterministic_no_fit_raw_descriptor_to_cross_section_calibration_table",
        "raw_descriptor_ids": list(RAW_DESCRIPTOR_IDS),
        "target_fields": list(TARGET_FIELDS),
        "feature_target_boundary": (
            "Phase 196 descriptors are frozen before the Phase 193 cross-section summary is joined. No raw HDF5, "
            "raw workbook, calibration fit, feature normalization, or model training is performed here."
        ),
        "calibration_rows": calibration_rows,
        "calibration_audit": calibration_audit,
        "fold_rows": fold_rows,
        "fold_audit": fold_audit,
        "gate": build_gate(
            phase193,
            phase194,
            phase196,
            calibration_rows,
            calibration_audit,
            fold_rows,
            fold_audit,
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase193", type=Path, default=DEFAULT_PHASE193)
    parser.add_argument("--phase194", type=Path, default=DEFAULT_PHASE194)
    parser.add_argument("--phase196", type=Path, default=DEFAULT_PHASE196)
    parser.add_argument("--conditions", type=Path, default=DEFAULT_CONDITIONS)
    parser.add_argument("--descriptors", type=Path, default=DEFAULT_DESCRIPTORS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table-csv", type=Path, required=True)
    parser.add_argument("--folds-csv", type=Path, required=True)
    args = parser.parse_args()
    calibration_rows, calibration_audit = build_calibration_rows(
        _read_csv(args.descriptors), _read_csv(args.conditions)
    )
    phase193 = _read_json(args.phase193)
    phase194 = _read_json(args.phase194)
    phase196 = _read_json(args.phase196)
    fold_rows, fold_audit = build_fold_rows(phase194, calibration_rows)
    payload = build_payload(
        phase193,
        phase194,
        phase196,
        calibration_rows,
        calibration_audit,
        fold_rows,
        fold_audit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.table_csv, calibration_rows, CALIBRATION_FIELDS)
    _write_csv(args.folds_csv, fold_rows, FOLD_FIELDS)
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
