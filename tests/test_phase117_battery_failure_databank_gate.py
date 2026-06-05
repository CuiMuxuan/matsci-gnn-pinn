from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    script = Path("scripts/server/build_phase117_battery_failure_databank_gate.py")
    spec = importlib.util.spec_from_file_location("phase117_battery_failure", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_rows(n: int = 210) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    descriptions = [f"Cell-{index:02d}" for index in range(30)]
    for index in range(n):
        group = descriptions[index % len(descriptions)]
        capacity = 2.0 + (index % 7) * 0.35
        energy = capacity * 3.65
        heater_power = 80.0 + (index % 11) * 12.0
        heater_time = 8.0 + (index % 5) * 4.0
        applied = heater_power * heater_time / 1000.0
        trigger_temp = 60.0 + (index % 13) * 5.0
        target = 4.0 * energy + 0.08 * heater_power + 0.12 * trigger_temp + (index % 3)
        rows.append(
            {
                "Cell-Description": group,
                "Test-ID": f"T-{index:04d}",
                "Test-Series": f"Series-{index % 5}",
                "S-FTRC-Generation": f"Gen-{index % 3}",
                "Cell-Format": "18650" if index % 2 == 0 else "21700",
                "Cell-Capacity-Ah": capacity,
                "Cell-Nominal-Voltage-V": 3.65,
                "Cell-Energy-Wh": energy,
                "Trigger-Mechanism": ["heater", "nail", "overcharge"][index % 3],
                "Pre-Test-Cell-Open-Circuit-Voltage-V": 3.7 + (index % 4) * 0.01,
                "Pre-Test-Cell-Mass-g": 45.0 + capacity * 8.0,
                "Pressure-Assisted-Seal-Configuration-Positive": f"P{index % 2}",
                "Pressure-Assisted-Seal-Configuration-Negative": f"N{index % 2}",
                "Baseline-Total-Energy-Yield-kJ": target * 0.9,
                "Baseline-Plus-Heat-Loss-Total-Energy-Yield-kJ": target * 0.96,
                "Corrected-Total-Energy-Yield-kJ": target,
                "Energy-Percent-Positive-Ejecta-%": 20.0 + (index % 9) * 3.0,
                "Energy-Percent-Negative-Ejecta-%": 15.0 + (index % 7) * 2.0,
                "Heater-Power-W": heater_power,
                "Heater-Time-On-s": heater_time,
                "Energy-Applied-to-Trigger-kJ": applied,
                "Avg-Cell-Temp-At-Trigger-degC": trigger_temp,
                "Post-Test-Mass-Unrecovered-g": 2.0 + (index % 6) * 0.4,
            }
        )
    return rows


def _write_workbook(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"Disclaimer": ["synthetic"]}).to_excel(
            writer, sheet_name="Disclaimer", index=False
        )
        pd.DataFrame({"ReadMe": ["synthetic"]}).to_excel(writer, sheet_name="ReadMe", index=False)
        pd.DataFrame(rows).to_excel(writer, sheet_name="Battery Failure Databank", index=False)
    return path


def test_phase117_finds_external_baseline_gap_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    module.EXPECTED_MIN_BYTES = 0
    workbook = _write_workbook(tmp_path / "battery.xlsx", _synthetic_rows())

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        raw_xlsx=workbook,
        source_url="file://synthetic",
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 117
    assert gate["status"] == "phase117_battery_failure_databank_gap_ready_focused_review"
    assert gate["phase117_focused_review_allowed"] is True
    assert gate["selected_target"] in gate["candidate_targets"]
    assert gate["phase117_model_mechanism_allowed"] is False
    assert gate["phase117_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["field_rows"] == 210
    assert manifest["counts"]["candidate_targets"] >= 1

    with (tmp_path / "out/phase117_battery_failure_databank_target_review_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        review_rows = list(csv.DictReader(handle))
    corrected = next(row for row in review_rows if row["target"] == "Corrected-Total-Energy-Yield-kJ")
    assert float(corrected["val_gain_vs_mean"]) > 0
    assert float(corrected["test_gain_vs_mean"]) > 0


def test_phase117_split_manifest_is_group_leakage_safe(tmp_path: Path):
    module = _load_module()
    module.EXPECTED_MIN_BYTES = 0
    workbook = _write_workbook(tmp_path / "battery.xlsx", _synthetic_rows())

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        raw_xlsx=workbook,
        source_url="file://synthetic",
    )

    split_manifest = module._read_json(tmp_path / "out/phase117_battery_failure_databank_split_manifest.json")
    train_groups = set(split_manifest["group_splits"]["train"])
    val_groups = set(split_manifest["group_splits"]["val"])
    test_groups = set(split_manifest["group_splits"]["test"])
    assert split_manifest["leakage_safe"] is True
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


def test_phase117_blocks_when_source_schema_is_missing(tmp_path: Path):
    module = _load_module()
    module.EXPECTED_MIN_BYTES = 0
    workbook = _write_workbook(
        tmp_path / "battery.xlsx",
        [{"Cell-Description": "Cell-01", "Corrected-Total-Energy-Yield-kJ": 1.0} for _ in range(160)],
    )

    try:
        module.build_package(
            root=Path(".").resolve(),
            output_dir=tmp_path / "out",
            raw_xlsx=workbook,
            source_url="file://synthetic",
        )
    except ValueError as exc:
        assert "Missing required Battery Failure columns" in str(exc)
    else:
        raise AssertionError("Expected missing schema to raise ValueError")
