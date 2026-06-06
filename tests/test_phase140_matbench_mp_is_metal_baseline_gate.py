from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase140_matbench_mp_is_metal_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase140_matbench_mp_is_metal", script)
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
        "dominant_element_shortcut": ("knn",),
    }
    module.EXPECTED_MIN_BYTES = 1
    return module


def _structure(elements: list[str], *, volume: float) -> dict[str, object]:
    sites = []
    for index, element in enumerate(elements):
        frac = (index + 1) / (len(elements) + 1)
        sites.append(
            {
                "species": [{"element": element, "occu": 1}],
                "abc": [frac, frac / 2.0, frac / 3.0],
                "xyz": [frac, frac * 2.0, frac * 3.0],
                "label": element,
                "properties": {},
            }
        )
    side = volume ** (1.0 / 3.0)
    return {
        "lattice": {
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


def _write_payload(path: Path, *, n_rows: int = 96) -> None:
    families = [
        (["Na", "Cl"], 0),
        (["Mg", "O"], 0),
        (["Si", "O"], 0),
        (["Al", "N"], 0),
        (["Cu"], 1),
        (["Fe"], 1),
        (["Ni"], 1),
        (["Ti"], 1),
        (["Ga", "As"], 0),
        (["Zn", "S"], 0),
        (["Mo"], 1),
        (["W"], 1),
    ]
    rows = []
    for index in range(n_rows):
        elements, label = families[index % len(families)]
        volume = 20.0 + 4.0 * len(elements) + (index % 11)
        if index % 19 == 0:
            label = 1 - label
        rows.append([_structure(elements, volume=volume), bool(label)])
    payload = {"columns": ["structure", "is_metal"], "index": list(range(n_rows)), "data": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_phase140_builds_capped_is_metal_gate_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    raw_path = tmp_path / "raw" / "matbench_mp_is_metal.json.gz"
    _write_payload(raw_path, n_rows=96)

    manifest = module.build_package(
        root=Path(".").resolve(),
        raw_path=raw_path,
        output_dir=tmp_path / "out",
        source_url="https://example.invalid/matbench_mp_is_metal.json.gz",
        force_download=False,
        max_rows=60,
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 140
    assert gate["status"] in {
        "phase140_matbench_mp_is_metal_triage_ready_focused_review",
        "phase140_matbench_mp_is_metal_closed_no_stable_guarded_gap",
        "phase140_matbench_mp_is_metal_incomplete_insufficient_split_rows",
    }
    assert gate["selected_target"] == "is_metal"
    assert gate["raw_row_count"] == 96
    assert gate["selected_raw_row_count"] == 60
    assert gate["row_cap_applied"] is True
    assert gate["phase140_model_mechanism_allowed"] is False
    assert gate["phase140_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase140_matbench_mp_is_metal_field_table.csv").exists()
    assert (tmp_path / "out" / "phase140_matbench_mp_is_metal_gate.json").exists()


def test_phase140_rejects_unexpected_payload_columns(tmp_path: Path):
    module = _load_module()
    raw_path = tmp_path / "raw" / "bad.json.gz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        json.dump({"columns": ["structure", "wrong"], "index": [0], "data": []}, handle)

    try:
        module.load_matbench_payload(raw_path)
    except ValueError as exc:
        assert "Unexpected matbench_mp_is_metal columns" in str(exc)
    else:
        raise AssertionError("expected ValueError")
