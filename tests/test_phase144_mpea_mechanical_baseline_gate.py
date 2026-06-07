from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase144_mpea_mechanical_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase144_mpea", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MIN_ROWS_FOR_REVIEW = 30
    module.MIN_SPLIT_ROWS = 8
    module.MODEL_METHODS = ("knn",)
    module.PROFILE_METHODS = {
        "formula_hash_shortcut": ("knn",),
        "reference_shortcut": ("knn",),
        "dominant_element_shortcut": ("knn",),
        "process_only_control": ("knn",),
    }
    module.EXPECTED_MIN_BYTES = 1
    return module


def _write_payload(path: Path, *, n_rows: int = 120) -> None:
    module = _load_module()
    formulas = [
        "Al0.25 Co1 Fe1 Ni1",
        "Al0.5 Co1 Fe1 Ni1",
        "Al0.75 Co1 Fe1 Ni1",
        "Al1 Co1 Fe1 Ni1",
        "Co1 Cr1 Fe1 Ni1",
        "Co1 Cr1 Mn1 Ni1",
        "Co1 Fe1 Ni1",
        "Co1 Fe1 Ni1 Si0.25",
        "Co1 Fe1 Ni1 Si0.5",
        "Co1 Fe1 Ni1 Si0.75",
        "Fe1 Mn1 Ni1",
        "Cr1 Fe1 Mn1 Ni1",
        "Al1 Cr1 Fe1 Ni1",
        "Al1 Co1 Cr1 Fe1",
        "Al1 Co1 Cr1 Ni1",
        "Co1 Cr1 Fe1 Mn1",
        "Co1 Cr1 Fe1 Mo0.2",
        "Co1 Cr1 Fe1 Ti0.2",
        "Co1 Cr1 Fe1 W0.1",
        "Co1 Cr1 Fe1 V0.2",
        "Al1 Fe1 Ni1 Ti0.2",
        "Al1 Co1 Fe1 Ti0.2",
        "Al1 Cr1 Fe1 Ti0.2",
        "Al1 Co1 Ni1 Ti0.2",
        "Co1 Fe1 Ni1 V0.2",
        "Co1 Fe1 Ni1 W0.1",
        "Co1 Fe1 Ni1 Mo0.2",
        "Cr1 Fe1 Ni1 V0.2",
        "Cr1 Fe1 Ni1 W0.1",
        "Cr1 Fe1 Ni1 Mo0.2",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(module.SOURCE_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for index in range(n_rows):
            formula = formulas[index % len(formulas)]
            normalized = formula.replace(" ", "")
            al_level = 1.0 if "Al1" in normalized else 0.5 if "Al0.5" in normalized else 0.25
            process = "CAST" if index % 3 else "WROUGHT"
            phase = "BCC" if al_level >= 0.75 else "FCC"
            grain_size = 20.0 + (index % 7) * 3.0
            hardness = 110.0 + 120.0 * al_level + (20.0 if process == "WROUGHT" else 0.0)
            ys = 180.0 + 420.0 * al_level + grain_size * 1.5
            writer.writerow(
                {
                    "IDENTIFIER: Reference ID": str(20 + index % 12),
                    "FORMULA": formula,
                    "PROPERTY: Microstructure": phase,
                    "PROPERTY: Processing method": process,
                    "PROPERTY: BCC/FCC/other": phase,
                    "PROPERTY: grain size ($\\mu$m)": f"{grain_size:.1f}",
                    "PROPERTY: Exp. Density (g/cm$^3$)": "",
                    "PROPERTY: Calculated Density (g/cm$^3$)": "7.8",
                    "PROPERTY: HV": f"{hardness:.1f}",
                    "PROPERTY: Type of test": "T",
                    "PROPERTY: Test temperature ($^\\circ$C)": "25",
                    "PROPERTY: YS (MPa)": f"{ys:.1f}",
                    "PROPERTY: UTS (MPa)": f"{ys * 1.3:.1f}",
                    "PROPERTY: Elongation (%)": f"{45.0 - al_level * 12.0:.1f}",
                    "PROPERTY: Elongation plastic (%)": "",
                    "PROPERTY: Exp. Young modulus (GPa)": f"{180.0 + al_level * 20.0:.1f}",
                    "PROPERTY: Calculated Young modulus (GPa)": "205",
                    "PROPERTY: O content (wppm)": "",
                    "PROPERTY: N content (wppm)": "",
                    "PROPERTY: C content (wppm)": "",
                    "REFERENCE: doi": f"10.example/{index % 12}",
                    "REFERENCE: year": str(2010 + index % 10),
                    "REFERENCE: title": "synthetic MPEA row",
                }
            )


def test_phase144_builds_mpea_gate_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    raw_path = tmp_path / "raw" / "MPEA_dataset.csv"
    _write_payload(raw_path)

    manifest = module.build_package(
        root=Path(".").resolve(),
        raw_path=raw_path,
        output_dir=tmp_path / "out",
        source_url="https://example.invalid/MPEA_dataset.csv",
        force_download=False,
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 144
    assert gate["status"] in {
        "phase144_mpea_mechanical_ready_focused_review",
        "phase144_mpea_mechanical_closed_no_stable_guarded_gap",
        "phase144_mpea_mechanical_incomplete_insufficient_split_rows",
    }
    assert gate["selected_target"] in module.TARGET_COLUMNS
    assert gate["parsed_row_count"] == 120
    assert gate["phase144_model_mechanism_allowed"] is False
    assert gate["phase144_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase144_mpea_mechanical_field_table.csv").exists()
    assert (tmp_path / "out" / "phase144_mpea_mechanical_gate.json").exists()
    markdown = (tmp_path / "out" / "phase144_mpea_mechanical.md").read_text(encoding="utf-8")
    assert "Phase 144 MPEA Mechanical Baseline Gate" in markdown
    assert "yield_strength_mpa" in markdown


def test_phase144_skips_unparseable_formula_rows(tmp_path: Path):
    module = _load_module()
    raw_path = tmp_path / "raw" / "MPEA_dataset.csv"
    _write_payload(raw_path, n_rows=40)
    rows = list(csv.DictReader(raw_path.open(encoding="utf-8", newline="")))
    rows.append({**rows[0], "FORMULA": "Xx2 O1"})
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(module.SOURCE_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    manifest = module.build_package(
        root=Path(".").resolve(),
        raw_path=raw_path,
        output_dir=tmp_path / "out",
        source_url="https://example.invalid/MPEA_dataset.csv",
        force_download=False,
    )

    assert manifest["counts"]["raw_rows"] == 41
    assert manifest["counts"]["skipped_rows"] == 1
    assert manifest["gate"]["skipped_row_count"] == 1


def test_phase144_rejects_missing_required_columns(tmp_path: Path):
    module = _load_module()
    raw_path = tmp_path / "raw" / "bad.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text("FORMULA,wrong\nAl1 Co1,1\n", encoding="utf-8")

    try:
        module.load_source_table(raw_path)
    except ValueError as exc:
        assert "Missing MPEA source columns" in str(exc)
    else:
        raise AssertionError("expected ValueError")
