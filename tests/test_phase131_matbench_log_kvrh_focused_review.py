from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_module():
    script = Path("scripts/server/build_phase131_matbench_log_kvrh_focused_review.py")
    spec = importlib.util.spec_from_file_location("phase131_matbench_log_kvrh", script)
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
        ("phase130_registered_split", "phase130_manifest", "phase130"),
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


def _write_phase130(path: Path, *, ready: bool = True) -> Path:
    module = _load_module()
    phase130_dir = path / "phase130"
    phase130_dir.mkdir(parents=True, exist_ok=True)
    families = [
        (["Li", "O"], 1.80),
        (["Na", "O"], 1.90),
        (["Mg", "O"], 2.05),
        (["Si", "O"], 2.00),
        (["Zn", "S"], 2.55),
        (["Cd", "Te"], 3.00),
        (["Ga", "As"], 3.25),
        (["Pb", "Se"], 3.55),
        (["Ba", "Ti", "O"], 2.35),
        (["Al", "N"], 2.15),
    ]
    raw_rows = []
    for index in range(120):
        elements, base = families[index % len(families)]
        volume = 20.0 + 4.0 * len(elements) + (index % 7)
        target = base + 0.004 * volume + (index % 5) * 0.01
        raw_rows.append([_structure(elements, volume=volume), target])
    field, _ = module.phase130.build_field_table(raw_rows)
    field.to_csv(phase130_dir / "phase130_matbench_log_kvrh_field_table.csv", index=False)
    splits = {
        "train": [index for index in range(len(field)) if index % 5 not in {3, 4}],
        "val": [index for index in range(len(field)) if index % 5 == 3],
        "test": [index for index in range(len(field)) if index % 5 == 4],
    }
    _write_json(
        phase130_dir / "phase130_matbench_log_kvrh_split_manifest.json",
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
        phase130_dir / "phase130_matbench_log_kvrh_gate.json",
        {
            "status": "phase130_matbench_log_kvrh_ready_focused_review"
            if ready
            else "phase130_matbench_log_kvrh_closed_no_stable_guarded_gap",
            "selected_target": "log10_k_vrh",
            "selected_profile": "composition_lattice_descriptors",
            "selected_method": "hist_gradient_boosting",
            "phase130_model_mechanism_allowed": False,
            "phase130_model_training_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
    )
    return phase130_dir


def test_phase131_reviews_phase130_candidate_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    phase130_dir = _write_phase130(tmp_path, ready=True)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase130_dir=phase130_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 131
    assert gate["status"] in {
        "phase131_matbench_log_kvrh_focused_review_ready_low_capacity_mechanism_gate",
        "phase131_matbench_log_kvrh_focused_review_closed_split_sensitivity_or_shortcut",
    }
    assert gate["phase131_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase131_matbench_log_kvrh_split_sensitivity_table.csv").exists()
    split_table = pd.read_csv(tmp_path / "out" / "phase131_matbench_log_kvrh_split_sensitivity_table.csv")
    assert "target_distribution_shift_z" in split_table.columns
    assert "nearest_neighbor_dominates" in split_table.columns


def test_phase131_blocks_if_phase130_gate_is_closed(tmp_path: Path):
    module = _load_module()
    phase130_dir = _write_phase130(tmp_path, ready=False)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase130_dir=phase130_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase131_matbench_log_kvrh_review_blocked_by_phase130"
    assert gate["phase131_model_mechanism_allowed"] is False
    assert gate["phase131_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
