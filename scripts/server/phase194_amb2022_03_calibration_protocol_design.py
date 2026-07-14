#!/usr/bin/env python3
"""Freeze grouped calibration and simulation-stress-test protocol for AMB2022-03."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PHASE193 = Path(
    os.environ.get(
        "AMB2022_01_PHASE193_IDENTIFIER_JOIN",
        "/root/matsci-gnn-pinn-ops/phase193_amb2022_03_identifier_join_design.json",
    )
)
DEFAULT_CONDITIONS = Path(
    os.environ.get(
        "AMB2022_01_PHASE193_CONDITIONS",
        "/root/matsci-gnn-pinn-ops/phase193_amb2022_03_single_track_conditions.csv",
    )
)
SETTING_FIELDS = (
    "setting_id",
    "laser_power_W",
    "scan_speed_mm_s",
    "spot_size_um",
    "condition_count",
    "thermal_group_ids",
    "target_columns",
)
TARGET_COLUMNS = ("depth_um_mean", "width_um_mean")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _setting_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row["laser_power_W"]),
        float(row["scan_speed_mm_s"]),
        float(row["spot_size_um"]),
    )


def _setting_id(key: tuple[float, float, float]) -> str:
    power, speed, spot = key
    return f"P{power:g}_V{speed:g}_D{spot:g}"


def build_setting_rows(condition_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in condition_rows:
        groups[_setting_key(row)].append(row)
    rows: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        power, speed, spot = key
        rows.append(
            {
                "setting_id": _setting_id(key),
                "laser_power_W": power,
                "scan_speed_mm_s": speed,
                "spot_size_um": spot,
                "condition_count": len(members),
                "thermal_group_ids": ";".join(sorted(str(row["thermal_group_id"]) for row in members)),
                "target_columns": ";".join(TARGET_COLUMNS),
            }
        )
    return rows


def build_folds(setting_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for held_out in setting_rows:
        training = [row["setting_id"] for row in setting_rows if row["setting_id"] != held_out["setting_id"]]
        folds.append(
            {
                "fold_id": f"leave_setting_out_{held_out['setting_id']}",
                "held_out_setting_id": held_out["setting_id"],
                "held_out_thermal_group_ids": held_out["thermal_group_ids"].split(";"),
                "training_setting_ids": training,
                "training_setting_count": len(training),
            }
        )
    return folds


def _phase193_ready(phase193: dict[str, Any]) -> bool:
    gate = phase193.get("gate", {})
    return (
        gate.get("status") == "phase193_identifier_join_design_ready_phase194_calibration_protocol_design"
        and bool(gate.get("phase194_calibration_protocol_design_allowed"))
        and gate.get("calibration_fitting_allowed") is False
        and gate.get("model_training_allowed") is False
    )


def build_gate(phase193: dict[str, Any], setting_rows: list[dict[str, Any]], folds: list[dict[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    if not _phase193_ready(phase193):
        blockers.append("phase193_identifier_join_gate_not_ready")
    if len(setting_rows) != 7:
        blockers.append("unexpected_unique_process_setting_count")
    if any(int(row["condition_count"]) != 3 for row in setting_rows):
        blockers.append("process_setting_replicate_contract_not_three")
    setting_ids = {str(row["setting_id"]) for row in setting_rows}
    if len(folds) != len(setting_rows):
        blockers.append("leave_setting_out_fold_count")
    for fold in folds:
        held_out = str(fold["held_out_setting_id"])
        training = set(fold["training_setting_ids"])
        if held_out not in setting_ids or held_out in training or training != setting_ids - {held_out}:
            blockers.append("leave_setting_out_leakage_contract")
            break
    blockers = sorted(set(blockers))
    ready = not blockers
    return {
        "status": (
            "phase194_calibration_protocol_design_ready_phase195_thermal_descriptor_extraction_design"
            if ready
            else "phase194_calibration_protocol_design_incomplete_or_leaky"
        ),
        "phase195_thermal_descriptor_extraction_design_allowed": ready,
        "calibration_fitting_allowed": False,
        "model_training_allowed": False,
        "hyperparameter_search_allowed": False,
        "post_b8_model_reselection_allowed": False,
        "grouped_leave_one_process_setting_out_required": True,
        "simulation_stress_test_requires_separate_holdout": True,
        "simulation_as_external_validation_allowed": False,
        "independent_3d_temperature_confirmation_allowed": False,
        "blocking_audits": blockers,
        "next_action": (
            "pre-register fixed raw-thermography descriptors before any calibration fit or synthetic stress test"
            if ready
            else "repair grouped split leakage or setting-replicate ambiguity before descriptor design"
        ),
    }


def build_protocol(phase193: dict[str, Any], condition_rows: list[dict[str, Any]]) -> dict[str, Any]:
    setting_rows = build_setting_rows(condition_rows)
    folds = build_folds(setting_rows)
    return {
        "phase": 194,
        "objective": "grouped_calibration_and_simulation_stress_test_protocol_design",
        "condition_unit": "One thermal signal group joined to the mean and standard deviation of two cross-section replicates.",
        "grouping_unit": "Exact (laser power, scan speed, spot size) process setting.",
        "targets": list(TARGET_COLUMNS),
        "frozen_evaluation": "Leave one complete process setting out across all seven settings; do not split within a setting.",
        "simulation_boundary": (
            "A future simulator may be calibrated only on designated calibration folds and stressed on a separate held-out setting. "
            "Synthetic results remain supporting evidence, not external experimental validation."
        ),
        "settings": setting_rows,
        "folds": folds,
        "gate": build_gate(phase193, setting_rows, folds),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SETTING_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase193", type=Path, default=DEFAULT_PHASE193)
    parser.add_argument("--conditions", type=Path, default=DEFAULT_CONDITIONS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings-csv", type=Path, required=True)
    args = parser.parse_args()
    payload = build_protocol(_read_json(args.phase193), _read_csv(args.conditions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.settings_csv, payload["settings"])
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
