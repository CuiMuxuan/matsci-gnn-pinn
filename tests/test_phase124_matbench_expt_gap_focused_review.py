from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_module():
    script = Path("scripts/server/build_phase124_matbench_expt_gap_focused_review.py")
    spec = importlib.util.spec_from_file_location("phase124_matbench_expt_gap", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODEL_METHODS = ("knn",)
    module.PROFILE_METHODS = {"composition_hash_shortcut": ("knn",)}
    module.MIN_SPLIT_ROWS = 10
    module.SPLIT_PLAN = (
        ("phase123_registered_split", "phase123_manifest", "phase123"),
        ("chemistry_family_hash_0", "group:chemistry_family_key", "test_family_0"),
        ("anion_electronegativity_bins", "bins:anion_fraction,mean_electronegativity,electronegativity_range", "test_bins"),
    )
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rows(n: int = 180) -> list[list[object]]:
    formulas = [
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
        "Al3Tc",
        "Cu2O",
    ]
    rows: list[list[object]] = []
    for index in range(n):
        formula = formulas[index % len(formulas)]
        target = 0.5 + (index % len(formulas)) * 0.12
        if "O" in formula:
            target += 1.0
        if "Se" in formula or "Te" in formula:
            target -= 0.25
        target += (index % 5) * 0.01
        rows.append([formula, max(0.0, target)])
    return rows


def _write_phase123(path: Path, *, ready: bool = True) -> Path:
    phase123_dir = path / "phase123"
    phase123_dir.mkdir(parents=True, exist_ok=True)
    source_df = pd.DataFrame(_rows(), columns=["composition", "gap_expt_ev"])
    phase123 = _load_module().phase123
    field = phase123.build_field_table(source_df)
    field.to_csv(phase123_dir / "phase123_matbench_expt_gap_field_table.csv", index=False)
    splits = {
        "train": [index for index in range(len(field)) if index % 5 not in {3, 4}],
        "val": [index for index in range(len(field)) if index % 5 == 3],
        "test": [index for index in range(len(field)) if index % 5 == 4],
    }
    _write_json(
        phase123_dir / "phase123_matbench_expt_gap_split_manifest.json",
        {
            "split_strategy": "synthetic_group_split",
            "n_groups": 12,
            "splits": splits,
            "group_splits": {"train": [], "val": [], "test": []},
            "leakage_safe": True,
        },
    )
    _write_json(
        phase123_dir / "phase123_matbench_expt_gap_gate.json",
        {
            "status": "phase123_matbench_expt_gap_gap_ready_focused_review" if ready else "phase123_matbench_expt_gap_closed_no_stable_guarded_gap",
            "selected_target": "gap_expt_ev",
            "selected_profile": "chemistry_descriptors",
            "selected_method": "extra_trees",
            "phase123_model_mechanism_allowed": False,
            "phase123_model_training_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
    )
    return phase123_dir


def test_phase124_reviews_phase123_candidate_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    phase123_dir = _write_phase123(tmp_path, ready=True)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase123_dir=phase123_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 124
    assert gate["status"] in {
        "phase124_matbench_expt_gap_focused_review_ready_low_capacity_mechanism_gate",
        "phase124_matbench_expt_gap_focused_review_closed_split_sensitivity_or_shortcut",
    }
    assert gate["phase124_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase124_matbench_expt_gap_split_sensitivity_table.csv").exists()


def test_phase124_blocks_if_phase123_gate_is_closed(tmp_path: Path):
    module = _load_module()
    phase123_dir = _write_phase123(tmp_path, ready=False)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase123_dir=phase123_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase124_matbench_expt_gap_review_blocked_by_phase123"
    assert gate["phase124_model_mechanism_allowed"] is False
    assert gate["phase124_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
