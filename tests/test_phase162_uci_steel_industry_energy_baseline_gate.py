from __future__ import annotations

import importlib.util
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def _load_module():
    script = Path("scripts/server/build_phase162_uci_steel_industry_energy_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase162_uci_steel_energy", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXPECTED_MIN_BYTES = 1
    module.MIN_ROWS_FOR_REVIEW = 100
    module.MIN_SPLIT_ROWS = 10
    module.MODEL_METHODS = ("ridge", "extra_trees")
    return module


def _synthetic_source(path: Path, *, weeks: int = 18) -> Path:
    start = datetime(2018, 1, 1, 0, 15)
    rows = []
    for step in range(weeks * 7 * 8):
        timestamp = start + timedelta(minutes=180 * step)
        nsm = timestamp.hour * 3600 + timestamp.minute * 60
        weekday = timestamp.strftime("%A")
        week_status = "Weekend" if weekday in {"Saturday", "Sunday"} else "Weekday"
        load_type = "Light_Load" if timestamp.hour < 8 else "Medium_Load" if timestamp.hour < 17 else "Maximum_Load"
        base = 5.0 + (25.0 if load_type == "Maximum_Load" else 12.0 if load_type == "Medium_Load" else 0.0)
        usage = base + 0.04 * (nsm / 900.0) + (2.0 if week_status == "Weekday" else -1.0)
        lagging = usage * 0.35
        leading = max(0.0, 8.0 - usage * 0.03)
        rows.append(
            {
                "date": timestamp.strftime("%d/%m/%Y %H:%M"),
                "Usage_kWh": usage,
                "Lagging_Current_Reactive.Power_kVarh": lagging,
                "Leading_Current_Reactive_Power_kVarh": leading,
                "CO2(tCO2)": usage * 0.0004,
                "Lagging_Current_Power_Factor": 85.0,
                "Leading_Current_Power_Factor": 99.0,
                "NSM": nsm,
                "WeekStatus": week_status,
                "Day_of_week": weekday,
                "Load_Type": load_type,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Steel_industry_data.csv", pd.DataFrame(rows).to_csv(index=False))
    return path


def test_phase162_builds_baseline_gate_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    raw_path = _synthetic_source(tmp_path / "raw" / "steel_industry_energy_consumption.zip")

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        raw_path=raw_path,
        source_url="https://example.invalid/steel-energy.zip",
        allow_download=False,
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 162
    assert gate["status"] in {
        "phase162_uci_steel_industry_energy_ready_focused_review",
        "phase162_uci_steel_industry_energy_closed_no_stable_guarded_gap",
    }
    assert gate["selected_target"] == "Usage_kWh"
    assert gate["phase162_model_mechanism_allowed"] is False
    assert gate["phase162_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase162_baseline_metric_table.csv").exists()
    assert (tmp_path / "out" / "phase162_uci_steel_industry_energy_baseline_gate.json").exists()


def test_phase162_week_split_keeps_weeks_together(tmp_path: Path):
    module = _load_module()
    raw_path = _synthetic_source(tmp_path / "raw" / "steel_industry_energy_consumption.zip")
    frame = module.load_steel_energy_table(raw_path)
    split = module.split_by_group(frame)

    assert split["group_column"] == "week_key"
    for week_key, group in frame.groupby("week_key"):
        assigned = {split["assignments"][int(index)] for index in group.index}
        assert len(assigned) == 1, week_key


def test_phase162_profiles_treat_direct_power_quantities_as_shortcuts():
    module = _load_module()
    profiles = module.profile_columns()

    assert profiles["direct_electrical_proxy_control"]["role"] == "shortcut_control"
    assert "CO2(tCO2)" in profiles["co2_direct_control"]["numeric"]
    assert profiles["load_type_shortcut_control"]["role"] == "shortcut_control"
    assert profiles["row_order_control"]["role"] == "shortcut_control"
