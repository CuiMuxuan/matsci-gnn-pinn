from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase126_matbench_phonons_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase126_matbench_phonons", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MIN_ROWS_FOR_REVIEW = 30
    module.MIN_SPLIT_ROWS = 8
    module.MODEL_METHODS = ("knn",)
    module.PROFILE_METHODS = {
        "composition_hash_shortcut": ("knn",),
        "chemistry_family_shortcut": ("knn",),
    }
    module.EXPECTED_MIN_BYTES = 1
    return module


def _structure(elements: list[str], *, volume: float, scale: float = 1.0) -> dict[str, object]:
    sites = []
    for index, element in enumerate(elements):
        frac = (index + 1) / (len(elements) + 1)
        sites.append(
            {
                "species": [{"element": element, "occu": 1}],
                "abc": [frac, frac / 2.0, frac / 3.0],
                "xyz": [frac * scale, frac * scale * 2.0, frac * scale * 3.0],
                "label": element,
                "properties": {},
            }
        )
    side = volume ** (1.0 / 3.0)
    return {
        "@module": "pymatgen.core.structure",
        "@class": "Structure",
        "@version": "test",
        "charge": None,
        "lattice": {
            "matrix": [[side, 0.0, 0.0], [0.0, side, 0.0], [0.0, 0.0, side]],
            "a": side,
            "b": side * (1.0 + 0.01 * len(elements)),
            "c": side * (1.0 + 0.02 * len(elements)),
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
            "volume": volume,
        },
        "sites": sites,
    }


def _write_payload(path: Path, *, n_rows: int = 90) -> None:
    families = [
        (["Li", "O"], 780.0),
        (["Na", "O"], 720.0),
        (["Mg", "O"], 690.0),
        (["Si", "O"], 960.0),
        (["Zn", "S"], 430.0),
        (["Cd", "Te"], 260.0),
        (["Ga", "As"], 340.0),
        (["Pb", "Se"], 210.0),
        (["Ba", "Ti", "O"], 610.0),
        (["Al", "N"], 890.0),
    ]
    rows = []
    for index in range(n_rows):
        elements, base = families[index % len(families)]
        volume = 20.0 + 4.0 * len(elements) + (index % 7)
        target = base + 2.5 * volume + (index % 5) * 3.0
        rows.append([_structure(elements, volume=volume, scale=1.0 + 0.01 * index), target])
    payload = {"columns": ["structure", "last phdos peak"], "index": list(range(n_rows)), "data": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_phase126_builds_phonon_gate_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    raw_path = tmp_path / "raw" / "matbench_phonons.json.gz"
    _write_payload(raw_path)

    manifest = module.build_package(
        root=Path(".").resolve(),
        raw_path=raw_path,
        output_dir=tmp_path / "out",
        source_url="https://example.invalid/matbench_phonons.json.gz",
        force_download=False,
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 126
    assert gate["status"] in {
        "phase126_matbench_phonons_peak_ready_focused_review",
        "phase126_matbench_phonons_peak_closed_no_stable_guarded_gap",
    }
    assert gate["selected_target"] == "last_phdos_peak"
    assert gate["row_count"] == 90
    assert gate["phase126_model_mechanism_allowed"] is False
    assert gate["phase126_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase126_matbench_phonons_field_table.csv").exists()
    assert (tmp_path / "out" / "phase126_matbench_phonons_gate.json").exists()


def test_phase126_rejects_unexpected_payload_columns(tmp_path: Path):
    module = _load_module()
    raw_path = tmp_path / "raw" / "bad.json.gz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        json.dump({"columns": ["structure", "wrong"], "index": [0], "data": []}, handle)

    try:
        module.load_matbench_payload(raw_path)
    except ValueError as exc:
        assert "Unexpected matbench_phonons columns" in str(exc)
    else:
        raise AssertionError("expected ValueError")
