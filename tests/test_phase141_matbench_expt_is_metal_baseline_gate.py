from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase141_matbench_expt_is_metal_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase141_matbench_expt_is_metal", script)
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
        ("Ag(AuS)2", 1),
        ("Al2O3", 0),
        ("BaTiO3", 0),
        ("Cu", 1),
        ("Fe", 1),
        ("GaAs", 0),
        ("MgO", 0),
        ("MoS2", 0),
        ("Ni", 1),
        ("SiO2", 0),
        ("Ti", 1),
        ("W", 1),
    ]
    rows = []
    for index in range(n_rows):
        composition, label = families[index % len(families)]
        if index % 17 == 0:
            label = 1 - label
        rows.append([composition, bool(label)])
    payload = {"columns": ["composition", "is_metal"], "index": list(range(n_rows)), "data": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_phase141_builds_expt_is_metal_gate_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    raw_path = tmp_path / "raw" / "matbench_expt_is_metal.json.gz"
    _write_payload(raw_path)

    manifest = module.build_package(
        root=Path(".").resolve(),
        raw_path=raw_path,
        output_dir=tmp_path / "out",
        source_url="https://example.invalid/matbench_expt_is_metal.json.gz",
        force_download=False,
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 141
    assert gate["status"] in {
        "phase141_matbench_expt_is_metal_ready_focused_review",
        "phase141_matbench_expt_is_metal_closed_no_stable_guarded_gap",
        "phase141_matbench_expt_is_metal_incomplete_insufficient_split_rows",
    }
    assert gate["selected_target"] == "is_metal"
    assert gate["row_count"] == 96
    assert gate["phase141_model_mechanism_allowed"] is False
    assert gate["phase141_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase141_matbench_expt_is_metal_field_table.csv").exists()
    assert (tmp_path / "out" / "phase141_matbench_expt_is_metal_gate.json").exists()


def test_phase141_skips_unparseable_composition_rows(tmp_path: Path):
    module = _load_module()
    raw_path = tmp_path / "raw" / "matbench_expt_is_metal.json.gz"
    _write_payload(raw_path, n_rows=40)
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["data"].append(["Xx2O", False])
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    manifest = module.build_package(
        root=Path(".").resolve(),
        raw_path=raw_path,
        output_dir=tmp_path / "out",
        source_url="https://example.invalid/matbench_expt_is_metal.json.gz",
        force_download=False,
    )

    assert manifest["phase"] == 141
    assert manifest["counts"]["raw_rows"] == 41
    assert manifest["counts"]["skipped_rows"] == 1
    assert manifest["gate"]["skipped_row_count"] == 1


def test_phase141_rejects_unexpected_payload_columns(tmp_path: Path):
    module = _load_module()
    raw_path = tmp_path / "raw" / "bad.json.gz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        json.dump({"columns": ["composition", "wrong"], "index": [0], "data": []}, handle)

    try:
        module.load_source_table(raw_path)
    except ValueError as exc:
        assert "Unexpected matbench_expt_is_metal columns" in str(exc)
    else:
        raise AssertionError("expected ValueError")
