from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase123_matbench_expt_gap_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase123_matbench_expt_gap", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODEL_METHODS = ("knn",)
    return module


def _write_source(path: Path, rows: list[list[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"index": list(range(len(rows))), "columns": ["composition", "gap expt"], "data": rows}
    path.write_bytes(gzip.compress(json.dumps(payload).encode("utf-8")))
    return path


def _synthetic_rows(n: int = 1200) -> list[list[object]]:
    rows: list[list[object]] = []
    families = [
        "Li2O",
        "Na2O",
        "BaTiO3",
        "ZnSe",
        "CdTe",
        "GaAs",
        "InP",
        "SiO2",
        "Ag(AuS)2",
        "PbSe",
    ]
    for index in range(n):
        formula = families[index % len(families)]
        if index % 10 == 0:
            formula = f"Li{1 + index % 3}O{1 + index % 2}"
        target = 0.4 + (index % 10) * 0.22
        if "O" in formula:
            target += 1.0
        if "Se" in formula or "Te" in formula:
            target -= 0.25
        if "AuS" in formula:
            target += 0.35
        target += (index % 5) * 0.01
        rows.append([formula, max(0.0, target)])
    return rows


def test_phase123_finds_external_baseline_gap_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    module.EXPECTED_MIN_BYTES = 0
    module.MIN_SPLIT_ROWS = 20
    source = _write_source(tmp_path / "matbench_expt_gap.json.gz", _synthetic_rows())

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        raw_path=source,
        source_url="file://synthetic",
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 123
    assert gate["status"] in {
        "phase123_matbench_expt_gap_gap_ready_focused_review",
        "phase123_matbench_expt_gap_closed_no_stable_guarded_gap",
    }
    assert manifest["counts"]["field_rows"] == 1200
    assert gate["phase123_model_mechanism_allowed"] is False
    assert gate["phase123_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    field_table = (tmp_path / "out/phase123_matbench_expt_gap_field_table.csv").read_text(encoding="utf-8")
    assert "frac_O" in field_table
    assert "gap_expt_ev" in field_table


def test_phase123_parser_handles_parenthesized_formula():
    module = _load_module()
    fractions = module.parse_composition("Ag(AuS)2")
    assert abs(sum(fractions.values()) - 1.0) < 1e-12
    assert abs(fractions["Ag"] - 0.2) < 1e-12
    assert abs(fractions["Au"] - 0.4) < 1e-12
    assert abs(fractions["S"] - 0.4) < 1e-12


def test_phase123_parser_covers_real_expt_gap_elements():
    module = _load_module()
    fractions = module.parse_composition("Al3Tc")
    assert abs(fractions["Al"] - 0.75) < 1e-12
    assert abs(fractions["Tc"] - 0.25) < 1e-12
    assert module.ATOMIC_NUMBER["Tc"] == 43
    assert module.ATOMIC_NUMBER["U"] == 92
    assert module.ELECTRONEGATIVITY["Xe"] > 0.0


def test_phase123_blocks_unexpected_source_schema(tmp_path: Path):
    module = _load_module()
    module.EXPECTED_MIN_BYTES = 0
    path = tmp_path / "bad.json.gz"
    path.write_bytes(gzip.compress(json.dumps({"columns": ["x"], "data": []}).encode("utf-8")))
    try:
        module.build_package(
            root=Path(".").resolve(),
            output_dir=tmp_path / "out",
            raw_path=path,
            source_url="file://synthetic",
        )
    except ValueError as exc:
        assert "Unexpected Matbench expt gap columns" in str(exc)
    else:
        raise AssertionError("Expected unexpected schema to raise ValueError")
