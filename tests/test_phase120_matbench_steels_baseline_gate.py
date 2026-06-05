from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase120_matbench_steels_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase120_matbench_steels", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_source(path: Path, rows: list[list[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"index": list(range(len(rows))), "columns": ["composition", "yield strength"], "data": rows}
    path.write_bytes(gzip.compress(json.dumps(payload).encode("utf-8")))
    return path


def _synthetic_rows(n: int = 300) -> list[list[object]]:
    rows: list[list[object]] = []
    families = [
        {"Ni": 0.08, "Cr": 0.03},
        {"Ni": 0.14},
        {"Co": 0.10, "Cr": 0.03},
        {"Cr": 0.12, "Mo": 0.03},
        {"Mo": 0.05, "Ni": 0.03},
        {"Al": 0.03, "Ni": 0.08},
        {"Ti": 0.03, "Ni": 0.06},
        {"Co": 0.08, "Mo": 0.04, "Ni": 0.03},
        {"Cr": 0.08, "Mn": 0.03, "Ni": 0.03},
        {"Co": 0.08, "Cr": 0.06, "Ti": 0.03},
    ]
    for index in range(n):
        family = families[index % len(families)]
        c = 0.001 + (index % 7) * 0.0005
        ni = family.get("Ni", 0.004 + (index % 4) * 0.001)
        co = family.get("Co", 0.004 + (index % 5) * 0.001)
        cr = family.get("Cr", 0.004 + (index % 6) * 0.001)
        mo = family.get("Mo", 0.003 + (index % 5) * 0.001)
        ti = family.get("Ti", 0.002 + (index % 4) * 0.001)
        mn = family.get("Mn", 0.001 + (index % 3) * 0.0005)
        si = 0.001 + (index % 4) * 0.0004
        al = family.get("Al", 0.002 + (index % 5) * 0.0008)
        fe = max(0.5, 1.0 - sum([c, ni, co, cr, mo, ti, mn, si, al]))
        composition = (
            f"Fe{fe:.6f}C{c:.6f}Mn{mn:.6f}Si{si:.6f}Cr{cr:.6f}"
            f"Ni{ni:.6f}Mo{mo:.6f}Co{co:.6f}Al{al:.6f}Ti{ti:.6f}"
        )
        target = 900.0 + 2100.0 * c + 1600.0 * ni + 950.0 * co + 700.0 * mo + 500.0 * ti
        target += (index % 6) * 3.0
        rows.append([composition, target])
    return rows


def test_phase120_finds_composition_baseline_gap_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    module.EXPECTED_MIN_BYTES = 0
    source = _write_source(tmp_path / "matbench_steels.json.gz", _synthetic_rows())

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        raw_path=source,
        source_url="file://synthetic",
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 120
    assert gate["status"] in {
        "phase120_matbench_steels_gap_ready_focused_review",
        "phase120_matbench_steels_closed_no_stable_guarded_gap",
    }
    assert manifest["counts"]["field_rows"] == 300
    assert gate["phase120_model_mechanism_allowed"] is False
    assert gate["phase120_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False

    field_table = (tmp_path / "out/phase120_matbench_steels_field_table.csv").read_text(encoding="utf-8")
    assert "frac_Ni" in field_table
    assert "yield_strength_mpa" in field_table


def test_phase120_parser_normalizes_composition_fractions():
    module = _load_module()
    fractions = module.parse_composition("Fe0.7Ni0.2Mo0.1")
    assert abs(sum(fractions.values()) - 1.0) < 1e-12
    assert abs(fractions["Fe"] - 0.7) < 1e-12
    assert abs(fractions["Ni"] - 0.2) < 1e-12
    assert abs(fractions["Mo"] - 0.1) < 1e-12


def test_phase120_blocks_unexpected_source_schema(tmp_path: Path):
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
        assert "Unexpected Matbench steels columns" in str(exc)
    else:
        raise AssertionError("Expected unexpected schema to raise ValueError")
