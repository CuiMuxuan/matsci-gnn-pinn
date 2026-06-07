from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pandas as pd


def _load_module():
    script = Path("scripts/server/build_phase158_uci_concrete_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase158_uci_concrete", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXPECTED_MIN_BYTES = 1
    module.MIN_ROWS_FOR_REVIEW = 80
    module.MIN_SPLIT_ROWS = 5
    module.MODEL_METHODS = ("knn",)
    return module


def _synthetic_source(path: Path, *, n_designs: int = 90) -> Path:
    rows = []
    for design in range(n_designs):
        cement = 250.0 + (design % 9) * 18.0
        slag = float((design % 5) * 22.5)
        fly_ash = float((design % 4) * 17.0)
        water = 145.0 + (design % 6) * 9.0
        superplasticizer = float((design % 3) * 2.5)
        coarse = 850.0 + (design % 7) * 12.0
        fine = 620.0 + (design % 8) * 10.0
        binder = cement + slag + fly_ash
        water_binder = water / binder
        for age in (7.0, 28.0):
            strength = (
                8.0
                + 0.055 * cement
                + 0.020 * slag
                + 0.012 * fly_ash
                - 18.0 * water_binder
                + 4.2 * (age ** 0.25)
                + 0.02 * superplasticizer
            )
            rows.append(
                {
                    "Cement (component 1)(kg in a m^3 mixture)": cement,
                    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": slag,
                    "Fly Ash (component 3)(kg in a m^3 mixture)": fly_ash,
                    "Water  (component 4)(kg in a m^3 mixture)": water,
                    "Superplasticizer (component 5)(kg in a m^3 mixture)": superplasticizer,
                    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": coarse,
                    "Fine Aggregate (component 7)(kg in a m^3 mixture)": fine,
                    "Age (day)": age,
                    "Concrete compressive strength(MPa, megapascals) ": strength,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Concrete_Data.csv", pd.DataFrame(rows).to_csv(index=False))
        archive.writestr("Concrete_Readme.txt", "synthetic phase158 fixture\n")
    return path


def test_phase158_builds_baseline_gate_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    raw_path = _synthetic_source(tmp_path / "raw" / "concrete_compressive_strength.zip")

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        raw_path=raw_path,
        source_url="https://example.invalid/concrete.zip",
        allow_download=False,
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 158
    assert gate["status"] in {
        "phase158_uci_concrete_ready_focused_review",
        "phase158_uci_concrete_closed_no_stable_guarded_gap",
    }
    assert manifest["counts"]["field_rows"] == 180
    assert gate["selected_target"] == "target_compressive_strength_mpa"
    assert gate["phase158_model_mechanism_allowed"] is False
    assert gate["phase158_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase158_baseline_metric_table.csv").exists()
    assert (tmp_path / "out" / "phase158_uci_concrete_baseline_gate.json").exists()


def test_phase158_group_split_excludes_age_from_mix_design(tmp_path: Path):
    module = _load_module()
    raw_path = _synthetic_source(tmp_path / "raw" / "concrete_compressive_strength.zip")
    frame = module.load_concrete_table(raw_path)
    split = module.split_by_group(frame)

    assert split["group_column"] == "mix_design_key"
    assert split["group_count"] == 90
    for design_key, group in frame.groupby("mix_design_key"):
        assigned = {split["assignments"][int(index)] for index in group.index}
        assert len(assigned) == 1, design_key
        assert set(group["age_day"]) == {7.0, 28.0}


def test_phase158_rejects_missing_concrete_table(tmp_path: Path):
    module = _load_module()
    raw_path = tmp_path / "raw" / "bad.zip"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Concrete_Readme.txt", "missing table\n")

    try:
        module.ensure_source(
            raw_path,
            source_url="https://example.invalid/concrete.zip",
            allow_download=False,
        )
    except ValueError as exc:
        assert "Missing Concrete_Data.xls" in str(exc)
    else:
        raise AssertionError("Expected missing concrete table to raise ValueError")
