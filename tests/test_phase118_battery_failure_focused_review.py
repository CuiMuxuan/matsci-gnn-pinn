from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_module():
    script = Path("scripts/server/build_phase118_battery_failure_focused_review.py")
    spec = importlib.util.spec_from_file_location("phase118_battery_failure", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _split_manifest(rows: list[dict[str, object]]) -> dict[str, object]:
    groups = sorted({str(row["Cell-Description"]) for row in rows})
    group_to_split = {}
    for index, group in enumerate(groups):
        if index < 18:
            group_to_split[group] = "train"
        elif index < 24:
            group_to_split[group] = "val"
        else:
            group_to_split[group] = "test"
    splits = {
        split: [index for index, row in enumerate(rows) if group_to_split[str(row["Cell-Description"])] == split]
        for split in ("train", "val", "test")
    }
    group_splits = {
        split: [group for group, label in group_to_split.items() if label == split]
        for split in ("train", "val", "test")
    }
    return {
        "split_strategy": "synthetic_group_hash_by_Cell-Description",
        "group_column": "Cell-Description",
        "n_groups": len(groups),
        "splits": splits,
        "group_splits": group_splits,
        "leakage_safe": True,
    }


def _rows(*, leaky: bool, n: int = 240) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(n):
        group_index = index % 30
        repeat = index // 30
        capacity = 2.0 + (group_index % 6) * 0.4
        cell_energy = capacity * 3.65
        pre_mass = 42.0 + capacity * 9.0 + (repeat % 4) * 0.25
        heater_power = 70.0 + (index % 10) * 15.0
        heater_time = 8.0 + (repeat % 5) * 4.0
        applied = heater_power * heater_time / 1000.0
        trigger_temp = 55.0 + (index % 12) * 6.0
        safe_target = 0.035 * heater_power + 0.18 * trigger_temp + 2.1 * capacity + (index % 4) * 0.2
        if leaky:
            target = 0.16 * pre_mass + 0.28 * safe_target + (index % 3) * 0.1
            corrected_energy = target * 3.4
            baseline_plus = target * 3.2
            baseline_total = target * 3.0
        else:
            target = safe_target
            corrected_energy = 20.0 + (index % 13) * 2.0
            baseline_plus = 25.0 + (index % 11) * 1.7
            baseline_total = 18.0 + (index % 7) * 2.5
        rows.append(
            {
                "phase117_row_id": f"BFDB-SYN-{index:04d}",
                "Cell-Description": f"Cell-{group_index:02d}",
                "Test-ID": f"T-{index:04d}",
                "Test-Series": f"Series-{index % 8}",
                "Cell-Format": "18650" if group_index % 2 == 0 else "21700",
                "Trigger-Mechanism": ["heater", "nail", "overcharge", "crush"][index % 4],
                "Pressure-Assisted-Seal-Configuration-Positive": f"P{index % 3}",
                "Pressure-Assisted-Seal-Configuration-Negative": f"N{(index + 1) % 3}",
                "S-FTRC-Generation": f"Gen-{index % 5}",
                "Cell-Capacity-Ah": capacity,
                "Cell-Nominal-Voltage-V": 3.65,
                "Cell-Energy-Wh": cell_energy,
                "Pre-Test-Cell-Open-Circuit-Voltage-V": 3.7 + (index % 4) * 0.01,
                "Pre-Test-Cell-Mass-g": pre_mass,
                "Heater-Power-W": heater_power,
                "Heater-Time-On-s": heater_time,
                "Energy-Applied-to-Trigger-kJ": applied,
                "Avg-Cell-Temp-At-Trigger-degC": trigger_temp,
                "Corrected-Total-Energy-Yield-kJ": corrected_energy,
                "Baseline-Plus-Heat-Loss-Total-Energy-Yield-kJ": baseline_plus,
                "Baseline-Total-Energy-Yield-kJ": baseline_total,
                "Energy-Percent-Positive-Ejecta-%": 20.0 + (index % 9) * 3.0,
                "Energy-Percent-Negative-Ejecta-%": 12.0 + (index % 7) * 2.0,
                "Post-Test-Mass-Unrecovered-g": target,
            }
        )
    return rows


def _write_phase117_inputs(path: Path, *, leaky: bool) -> Path:
    rows = _rows(leaky=leaky)
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        path / "phase117_battery_failure_databank_field_table.csv",
        index=False,
    )
    _write_json(path / "phase117_battery_failure_databank_split_manifest.json", _split_manifest(rows))
    _write_json(
        path / "phase117_battery_failure_databank_gate.json",
        {
            "status": "phase117_battery_failure_databank_gap_ready_focused_review",
            "selected_target": "Post-Test-Mass-Unrecovered-g",
            "selected_profile": "cell_trigger_safe",
            "selected_method": "hist_gradient_boosting",
            "phase117_model_mechanism_allowed": False,
            "phase117_model_training_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
    )
    (path / "phase117_battery_failure_databank_target_review_table.csv").write_text(
        "target,status\nPost-Test-Mass-Unrecovered-g,candidate_gap_ready_focused_review\n",
        encoding="utf-8",
    )
    return path


def test_phase118_closes_leaky_or_target_family_coupled_candidate(tmp_path: Path):
    module = _load_module()
    phase117_dir = _write_phase117_inputs(tmp_path / "phase117", leaky=True)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase117_dir=phase117_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 118
    assert gate["status"] == "phase118_battery_failure_focused_review_closed_leakage_or_split_sensitivity"
    assert gate["phase118_model_mechanism_allowed"] is False
    assert gate["phase118_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert "target_family_dependency" in gate["blocking_audits"]

    with (tmp_path / "out/phase118_battery_failure_leakage_shortcut_audit_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        audits = list(csv.DictReader(handle))
    assert any(row["audit"] == "target_family_dependency" and row["status"] == "block" for row in audits)


def test_phase118_can_allow_only_low_capacity_mechanism_design_for_stable_nonleaky_gap(tmp_path: Path):
    module = _load_module()
    phase117_dir = _write_phase117_inputs(tmp_path / "phase117", leaky=False)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase117_dir=phase117_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase118_battery_failure_focused_review_ready_low_capacity_mechanism_gate"
    assert gate["phase118_model_mechanism_allowed"] is True
    assert gate["phase118_low_capacity_mechanism_design_allowed"] is True
    assert gate["phase118_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert gate["split_pass_rate"] >= module.MIN_STABLE_SPLIT_PASS_RATE


def test_phase118_blocks_when_phase117_gate_is_not_open(tmp_path: Path):
    module = _load_module()
    phase117_dir = _write_phase117_inputs(tmp_path / "phase117", leaky=False)
    _write_json(
        phase117_dir / "phase117_battery_failure_databank_gate.json",
        {
            "status": "phase117_battery_failure_databank_closed_no_stable_guarded_gap",
            "selected_target": "Post-Test-Mass-Unrecovered-g",
        },
    )

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase117_dir=phase117_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase118_battery_failure_review_blocked_by_phase117"
    assert gate["phase118_model_mechanism_allowed"] is False
    assert gate["phase118_model_training_allowed"] is False
