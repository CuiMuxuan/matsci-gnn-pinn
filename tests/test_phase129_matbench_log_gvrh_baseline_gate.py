from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase129_matbench_log_gvrh_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase129_matbench_log_gvrh", script)
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
        (["Li", "O"], 1.8),
        (["Na", "O"], 1.9),
        (["Mg", "O"], 2.1),
        (["Si", "O"], 2.0),
        (["Zn", "S"], 2.6),
        (["Cd", "Te"], 3.0),
        (["Ga", "As"], 3.3),
        (["Pb", "Se"], 3.6),
        (["Ba", "Ti", "O"], 2.4),
        (["Al", "N"], 2.2),
    ]
    rows = []
    for index in range(n_rows):
        elements, base = families[index % len(families)]
        volume = 25.0 + 5.0 * len(elements) + (index % 7)
        target = base + 0.005 * volume + (index % 5) * 0.01
        rows.append([_structure(elements, volume=volume, scale=1.0 + 0.01 * index), target])
    payload = {"columns": ["structure", "log10(G_VRH)"], "index": list(range(n_rows)), "data": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_phase129_builds_log_gvrh_gate_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    raw_path = tmp_path / "raw" / "matbench_log_gvrh.json.gz"
    _write_payload(raw_path)

    manifest = module.build_package(
        root=Path(".").resolve(),
        raw_path=raw_path,
        output_dir=tmp_path / "out",
        source_url="https://example.invalid/matbench_log_gvrh.json.gz",
        force_download=False,
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 129
    assert gate["status"] in {
        "phase129_matbench_log_gvrh_ready_focused_review",
        "phase129_matbench_log_gvrh_closed_no_stable_guarded_gap",
    }
    assert gate["selected_target"] == "log10_g_vrh"
    assert gate["row_count"] == 90
    assert gate["phase129_model_mechanism_allowed"] is False
    assert gate["phase129_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase129_matbench_log_gvrh_field_table.csv").exists()
    assert (tmp_path / "out" / "phase129_matbench_log_gvrh_gate.json").exists()


def test_phase129_skips_unparseable_structure_rows(tmp_path: Path):
    module = _load_module()
    raw_path = tmp_path / "raw" / "matbench_log_gvrh.json.gz"
    _write_payload(raw_path, n_rows=40)
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["data"].append([_structure(["Pu"], volume=40.0), 1.5])
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    manifest = module.build_package(
        root=Path(".").resolve(),
        raw_path=raw_path,
        output_dir=tmp_path / "out",
        source_url="https://example.invalid/matbench_log_gvrh.json.gz",
        force_download=False,
    )

    assert manifest["phase"] == 129
    assert manifest["counts"]["raw_rows"] == 41
    assert manifest["counts"]["skipped_rows"] == 1
    assert manifest["gate"]["skipped_row_count"] == 1


def test_phase129_rejects_unexpected_payload_columns(tmp_path: Path):
    module = _load_module()
    raw_path = tmp_path / "raw" / "bad.json.gz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        json.dump({"columns": ["structure", "wrong"], "index": [0], "data": []}, handle)

    try:
        module.load_matbench_payload(raw_path)
    except ValueError as exc:
        assert "Unexpected matbench_log_gvrh columns" in str(exc)
    else:
        raise AssertionError("expected ValueError")

