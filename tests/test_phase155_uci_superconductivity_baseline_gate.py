from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pandas as pd


ELEMENT_COLUMNS = ("H", "Li", "B", "C", "N", "O", "Mg", "Al", "Si", "Fe", "Cu", "Y")


def _load_module():
    script = Path("scripts/server/build_phase155_uci_superconductivity_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase155_uci_superconductivity", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXPECTED_MIN_BYTES = 1
    module.MIN_ROWS_FOR_REVIEW = 80
    module.MIN_SPLIT_ROWS = 5
    module.MODEL_METHODS = ("knn",)
    return module


def _synthetic_source(path: Path, *, n_rows: int = 180) -> Path:
    train_rows = []
    unique_rows = []
    for index in range(n_rows):
        first = index % len(ELEMENT_COLUMNS)
        second = (index // len(ELEMENT_COLUMNS) + 1 + first) % len(ELEMENT_COLUMNS)
        third = (index // (len(ELEMENT_COLUMNS) * 2) + 3 + first) % len(ELEMENT_COLUMNS)
        active = sorted({first, second, third})
        fractions = {element: 0.0 for element in ELEMENT_COLUMNS}
        for pos, element_index in enumerate(active):
            fractions[ELEMENT_COLUMNS[element_index]] = (pos + 1.0) / sum(range(1, len(active) + 1))
        number_of_elements = len(active)
        mean_atomic_mass = 20.0 + sum(active) * 1.7
        wtd_mean_atomic_mass = mean_atomic_mass + 0.5 * fractions["O"] + 0.2 * fractions["Cu"]
        wtd_std_atomic_mass = 0.1 + (max(active) - min(active)) * 0.05
        mean_fie = 5.0 + active[0] * 0.3
        wtd_mean_fie = mean_fie + fractions["O"] * 1.5 - fractions["Fe"] * 0.3
        target = (
            8.0
            + 0.9 * wtd_mean_atomic_mass
            + 4.0 * fractions["O"]
            - 2.5 * fractions["Fe"]
            + (index % 4) * 0.05
        )
        train_rows.append(
            {
                "number_of_elements": number_of_elements,
                "mean_atomic_mass": mean_atomic_mass,
                "wtd_mean_atomic_mass": wtd_mean_atomic_mass,
                "wtd_std_atomic_mass": wtd_std_atomic_mass,
                "mean_fie": mean_fie,
                "wtd_mean_fie": wtd_mean_fie,
                "critical_temp": target,
            }
        )
        material = "".join(
            f"{ELEMENT_COLUMNS[element_index]}{pos + 1}"
            for pos, element_index in enumerate(active)
        )
        unique_rows.append({**fractions, "critical_temp": target, "material": material})

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("train.csv", pd.DataFrame(train_rows).to_csv(index=False))
        archive.writestr("unique_m.csv", pd.DataFrame(unique_rows).to_csv(index=False))
    return path


def test_phase155_builds_baseline_gate_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    raw_path = _synthetic_source(tmp_path / "raw" / "superconductivty_data.zip")

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        raw_path=raw_path,
        source_url="https://example.invalid/superconductivty_data.zip",
        allow_download=False,
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 155
    assert gate["status"] in {
        "phase155_uci_superconductivity_ready_focused_review",
        "phase155_uci_superconductivity_closed_no_stable_guarded_gap",
    }
    assert manifest["counts"]["field_rows"] == 180
    assert gate["selected_target"] == "target_critical_temp_K"
    assert gate["phase155_model_mechanism_allowed"] is False
    assert gate["phase155_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase155_baseline_metric_table.csv").exists()
    assert (tmp_path / "out" / "phase155_uci_superconductivity_baseline_gate.json").exists()


def test_phase155_keeps_shortcuts_out_of_admissible_full_profile(tmp_path: Path):
    module = _load_module()
    raw_path = _synthetic_source(tmp_path / "raw" / "superconductivty_data.zip")
    frame = module.load_superconductivity_table(raw_path)
    profiles = module.profile_columns(frame)

    full_columns = set(profiles["uci_feature_full"]["columns"])
    assert "wtd_mean_atomic_mass" in full_columns
    assert "element_set_hash" not in full_columns
    assert "dominant_element_hash" not in full_columns
    assert "row_order_fraction" not in full_columns
    assert "formula_length" not in full_columns
    assert "max_element_fraction" not in full_columns
    assert any(column.startswith("frac_") for column in profiles["element_fraction_vector"]["columns"])


def test_phase155_rejects_missing_zip_members(tmp_path: Path):
    module = _load_module()
    raw_path = tmp_path / "raw" / "bad.zip"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("train.csv", "critical_temp\n1.0\n")

    try:
        module.ensure_source(
            raw_path,
            source_url="https://example.invalid/superconductivty_data.zip",
            allow_download=False,
        )
    except ValueError as exc:
        assert "Missing required UCI members" in str(exc)
    else:
        raise AssertionError("Expected missing ZIP members to raise ValueError")
