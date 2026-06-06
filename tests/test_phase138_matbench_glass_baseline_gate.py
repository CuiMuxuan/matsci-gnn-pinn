from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase138_matbench_glass_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase138_matbench_glass", script)
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


def _write_payload(path: Path, *, n_rows: int = 96) -> None:
    families = [
        ("Al", 0),
        ("Al(NiB)2", 1),
        ("Al10Co21B19", 1),
        ("Al10Co23B17", 1),
        ("Zr50Cu40Al10", 1),
        ("Zr60Ni25Al15", 1),
        ("Fe80B20", 0),
        ("Cu50Zr50", 1),
        ("Ni80P20", 0),
        ("Pd40Ni40P20", 1),
        ("SiO2", 0),
        ("Mg65Cu25Y10", 1),
    ]
    rows = []
    for index in range(n_rows):
        composition, label = families[index % len(families)]
        if index % 17 == 0:
            label = 1 - label
        rows.append([composition, bool(label)])
    payload = {"columns": ["composition", "gfa"], "index": list(range(n_rows)), "data": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_phase138_builds_glass_gate_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    raw_path = tmp_path / "raw" / "matbench_glass.json.gz"
    _write_payload(raw_path)

    manifest = module.build_package(
        root=Path(".").resolve(),
        raw_path=raw_path,
        output_dir=tmp_path / "out",
        source_url="https://example.invalid/matbench_glass.json.gz",
        force_download=False,
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 138
    assert gate["status"] in {
        "phase138_matbench_glass_ready_focused_review",
        "phase138_matbench_glass_closed_no_stable_guarded_gap",
        "phase138_matbench_glass_incomplete_insufficient_split_rows",
    }
    assert gate["selected_target"] == "gfa"
    assert gate["row_count"] == 96
    assert gate["phase138_model_mechanism_allowed"] is False
    assert gate["phase138_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase138_matbench_glass_field_table.csv").exists()
    assert (tmp_path / "out" / "phase138_matbench_glass_gate.json").exists()


def test_phase138_skips_unparseable_composition_rows(tmp_path: Path):
    module = _load_module()
    raw_path = tmp_path / "raw" / "matbench_glass.json.gz"
    _write_payload(raw_path, n_rows=40)
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["data"].append(["Xx2O", True])
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    manifest = module.build_package(
        root=Path(".").resolve(),
        raw_path=raw_path,
        output_dir=tmp_path / "out",
        source_url="https://example.invalid/matbench_glass.json.gz",
        force_download=False,
    )

    assert manifest["phase"] == 138
    assert manifest["counts"]["raw_rows"] == 41
    assert manifest["counts"]["skipped_rows"] == 1
    assert manifest["gate"]["skipped_row_count"] == 1


def test_phase138_rejects_unexpected_payload_columns(tmp_path: Path):
    module = _load_module()
    raw_path = tmp_path / "raw" / "bad.json.gz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        json.dump({"columns": ["composition", "wrong"], "index": [0], "data": []}, handle)

    try:
        module.load_source_table(raw_path)
    except ValueError as exc:
        assert "Unexpected matbench_glass columns" in str(exc)
    else:
        raise AssertionError("expected ValueError")
