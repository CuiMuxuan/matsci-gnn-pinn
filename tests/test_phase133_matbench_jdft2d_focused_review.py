from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_module():
    script = Path("scripts/server/build_phase133_matbench_jdft2d_focused_review.py")
    spec = importlib.util.spec_from_file_location("phase133_matbench_jdft2d", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODEL_METHODS = ("knn",)
    module.PROFILE_METHODS = {
        "composition_hash_shortcut": ("knn",),
        "chemistry_family_shortcut": ("knn",),
        "dominant_element_shortcut": ("knn",),
    }
    module.MIN_SPLIT_ROWS = 8
    module.SPLIT_PLAN = (
        ("phase132_registered_split", "phase132_manifest", "phase132"),
        ("chemistry_family_hash_0", "group:chemistry_family_key", "test_family_0"),
        ("lattice_volume_bins", "bins:lattice_volume", "test_lattice"),
    )
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _structure(elements: list[str], *, volume: float) -> dict[str, object]:
    sites = []
    for index, element in enumerate(elements):
        frac = (index + 1) / (len(elements) + 1)
        sites.append(
            {
                "species": [{"element": element, "occu": 1}],
                "abc": [frac, frac / 2.0, frac / 3.0],
                "xyz": [frac, frac, frac],
            }
        )
    side = volume ** (1.0 / 3.0)
    return {
        "@module": "pymatgen.core.structure",
        "@class": "Structure",
        "charge": 0,
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


def _write_phase132(path: Path, *, ready: bool = True) -> Path:
    module = _load_module()
    phase132_dir = path / "phase132"
    phase132_dir.mkdir(parents=True, exist_ok=True)
    families = [
        (["Li", "O"], 80.0),
        (["Na", "O"], 95.0),
        (["Mg", "O"], 120.0),
        (["Si", "O"], 135.0),
        (["Zn", "S"], 150.0),
        (["Cd", "Te"], 175.0),
        (["Ga", "As"], 190.0),
        (["Pb", "Se"], 210.0),
        (["Ba", "Ti", "O"], 160.0),
        (["Al", "N"], 130.0),
    ]
    raw_rows = []
    for index in range(120):
        elements, base = families[index % len(families)]
        volume = 18.0 + 5.0 * len(elements) + (index % 9)
        target = base + 0.7 * volume + (index % 5) * 2.0
        raw_rows.append([_structure(elements, volume=volume), target])
    field, _ = module.phase132.build_field_table(raw_rows)
    field.to_csv(phase132_dir / "phase132_matbench_jdft2d_field_table.csv", index=False)
    splits = {
        "train": [index for index in range(len(field)) if index % 5 not in {3, 4}],
        "val": [index for index in range(len(field)) if index % 5 == 3],
        "test": [index for index in range(len(field)) if index % 5 == 4],
    }
    _write_json(
        phase132_dir / "phase132_matbench_jdft2d_split_manifest.json",
        {
            "split_strategy": "synthetic_group_split",
            "group_column": "chemistry_family_key",
            "n_groups": 10,
            "splits": splits,
            "group_splits": {"train": [], "val": [], "test": []},
            "leakage_safe": True,
        },
    )
    _write_json(
        phase132_dir / "phase132_matbench_jdft2d_gate.json",
        {
            "status": "phase132_matbench_jdft2d_ready_focused_review"
            if ready
            else "phase132_matbench_jdft2d_closed_no_stable_guarded_gap",
            "selected_target": "exfoliation_en",
            "selected_profile": "composition_lattice_descriptors",
            "selected_method": "extra_trees",
            "phase132_model_mechanism_allowed": False,
            "phase132_model_training_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
    )
    return phase132_dir


def test_phase133_reviews_phase132_candidate_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    phase132_dir = _write_phase132(tmp_path, ready=True)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase132_dir=phase132_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 133
    assert gate["status"] in {
        "phase133_matbench_jdft2d_focused_review_ready_low_capacity_mechanism_gate",
        "phase133_matbench_jdft2d_focused_review_closed_split_sensitivity_or_shortcut",
    }
    assert gate["phase133_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase133_matbench_jdft2d_split_sensitivity_table.csv").exists()
    split_table = pd.read_csv(tmp_path / "out" / "phase133_matbench_jdft2d_split_sensitivity_table.csv")
    assert set(split_table["split_id"]) == {
        "phase132_registered_split",
        "chemistry_family_hash_0",
        "lattice_volume_bins",
    }


def test_phase133_blocks_when_phase132_gate_not_open(tmp_path: Path):
    module = _load_module()
    phase132_dir = _write_phase132(tmp_path, ready=False)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase132_dir=phase132_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase133_matbench_jdft2d_review_blocked_by_phase132"
    assert gate["phase133_model_mechanism_allowed"] is False
    assert gate["phase133_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
