from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_module():
    script = Path("scripts/server/build_phase119_battery_failure_candidate_sweep.py")
    spec = importlib.util.spec_from_file_location("phase119_battery_failure", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.phase118.MODEL_METHODS = ("knn",)
    module.phase118.SPLIT_PLAN = (
        ("phase117_registered_split", "phase117_manifest", "phase117"),
        ("cell_description_hash_0", "Cell-Description", "phase119_test_cell_0"),
        ("cell_description_hash_1", "Cell-Description", "phase119_test_cell_1"),
    )
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


def _rows(*, allow_ejecta: bool, n: int = 240) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(n):
        group_index = index % 30
        repeat = index // 30
        capacity = 2.0 + (group_index % 6) * 0.4
        cell_energy = capacity * 3.65
        pre_mass = 42.0 + capacity * 9.0 + (repeat % 4) * 0.2
        heater_power = 70.0 + (index % 10) * 15.0
        heater_time = 8.0 + (repeat % 5) * 4.0
        applied = heater_power * heater_time / 1000.0
        trigger_temp = 55.0 + (index % 12) * 6.0
        safe_signal = 0.05 * heater_power + 0.18 * trigger_temp + 2.0 * capacity
        if allow_ejecta:
            ejecta = safe_signal + (index % 3) * 0.1
            corrected_energy = 40.0 + (index % 17) * 1.3
            baseline_plus = 35.0 + (index % 11) * 1.1
            baseline_total = 30.0 + (index % 13) * 1.2
            mass_target = 8.0 + (index % 8) * 0.3
        else:
            mass_target = 0.18 * pre_mass + 0.4 * safe_signal
            corrected_energy = mass_target * 3.2
            baseline_plus = mass_target * 3.0
            baseline_total = mass_target * 2.8
            ejecta = 0.6 * mass_target + 10.0
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
                "Energy-Percent-Positive-Ejecta-%": ejecta,
                "Energy-Percent-Negative-Ejecta-%": 12.0 + (index % 7) * 2.0,
                "Post-Test-Mass-Unrecovered-g": mass_target,
            }
        )
    return rows


def _write_inputs(path: Path, *, allow_ejecta: bool, candidate_targets: list[str]) -> tuple[Path, Path]:
    rows = _rows(allow_ejecta=allow_ejecta)
    phase117_dir = path / "phase117"
    phase118_dir = path / "phase118"
    phase117_dir.mkdir(parents=True, exist_ok=True)
    phase118_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        phase117_dir / "phase117_battery_failure_databank_field_table.csv",
        index=False,
    )
    _write_json(phase117_dir / "phase117_battery_failure_databank_split_manifest.json", _split_manifest(rows))
    _write_json(
        phase117_dir / "phase117_battery_failure_databank_gate.json",
        {
            "status": "phase117_battery_failure_databank_gap_ready_focused_review",
            "candidate_targets": candidate_targets,
            "selected_target": candidate_targets[0],
        },
    )
    (phase117_dir / "phase117_battery_failure_databank_target_review_table.csv").write_text(
        "target,phase117_candidate\n"
        + "\n".join(f"{target},true" for target in candidate_targets)
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        phase118_dir / "phase118_battery_failure_focused_review_gate.json",
        {
            "status": "phase118_battery_failure_focused_review_closed_leakage_or_split_sensitivity",
            "selected_target": candidate_targets[0],
        },
    )
    return phase117_dir, phase118_dir


def test_phase119_closes_all_candidates_when_dependency_controls_block(tmp_path: Path):
    module = _load_module()
    phase117_dir, phase118_dir = _write_inputs(
        tmp_path,
        allow_ejecta=False,
        candidate_targets=[
            "Post-Test-Mass-Unrecovered-g",
            "Corrected-Total-Energy-Yield-kJ",
        ],
    )

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase117_dir=phase117_dir,
        phase118_dir=phase118_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 119
    assert gate["status"] == "phase119_battery_failure_candidate_sweep_closed_all_phase117_candidates"
    assert gate["allowed_candidate_targets"] == []
    assert "Post-Test-Mass-Unrecovered-g" in gate["closed_candidate_targets"]
    assert gate["phase119_model_mechanism_allowed"] is False
    assert gate["phase119_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False


def test_phase119_can_allow_nonleaky_candidate_without_opening_training(tmp_path: Path):
    module = _load_module()
    phase117_dir, phase118_dir = _write_inputs(
        tmp_path,
        allow_ejecta=True,
        candidate_targets=["Energy-Percent-Positive-Ejecta-%"],
    )

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase117_dir=phase117_dir,
        phase118_dir=phase118_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase119_battery_failure_candidate_sweep_ready_low_capacity_mechanism_gate"
    assert gate["allowed_candidate_targets"] == ["Energy-Percent-Positive-Ejecta-%"]
    assert gate["phase119_model_mechanism_allowed"] is True
    assert gate["phase119_low_capacity_mechanism_design_allowed"] is True
    assert gate["phase119_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
