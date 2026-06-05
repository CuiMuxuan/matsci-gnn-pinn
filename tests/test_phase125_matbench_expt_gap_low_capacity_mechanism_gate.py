from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_module():
    script = Path("scripts/server/build_phase125_matbench_expt_gap_low_capacity_mechanism_gate.py")
    spec = importlib.util.spec_from_file_location("phase125_matbench_expt_gap", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODEL_SPECS = (
        module.ModelSpec("ordinary_least_squares"),
        module.ModelSpec("ridge", alpha=1.0),
    )
    module.MECHANISM_PROFILES = {
        "ionicity_proxy": module.MECHANISM_PROFILES["ionicity_proxy"],
        "mechanism_compact": module.MECHANISM_PROFILES["mechanism_compact"],
    }
    module.MIN_SPLIT_ROWS = 10
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_rows(n: int = 180) -> list[list[object]]:
    formulas = [
        "Li2O",
        "Na2O",
        "MgO",
        "BaTiO3",
        "ZnSe",
        "CdTe",
        "GaAs",
        "InP",
        "SiO2",
        "PbSe",
        "Cu2O",
        "Ag(AuS)2",
    ]
    rows: list[list[object]] = []
    for index in range(n):
        formula = formulas[index % len(formulas)]
        target = 1.2
        if "O" in formula:
            target += 1.4
        if "F" in formula or "Cl" in formula:
            target += 0.8
        if "Se" in formula or "Te" in formula:
            target -= 0.4
        if "Pb" in formula or "Cd" in formula or "Ag" in formula:
            target -= 0.2
        target += (index % 5) * 0.015
        rows.append([formula, max(0.0, target)])
    return rows


def _write_inputs(
    path: Path,
    *,
    phase124_guard_val: float,
    phase124_guard_test: float,
    phase124_ready: bool = True,
) -> tuple[Path, Path]:
    phase123_dir = path / "phase123"
    phase124_dir = path / "phase124"
    phase123_dir.mkdir(parents=True, exist_ok=True)

    phase123 = _load_module().phase123
    source_df = pd.DataFrame(_source_rows(), columns=["composition", "gap_expt_ev"])
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
            "status": "phase123_matbench_expt_gap_gap_ready_focused_review",
            "selected_target": "gap_expt_ev",
            "selected_profile": "chemistry_descriptors",
            "selected_method": "extra_trees",
            "selected_validation_rmse": phase124_guard_val,
            "selected_test_rmse": phase124_guard_test,
            "phase123_model_training_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
    )

    phase124_dir.mkdir(parents=True, exist_ok=True)
    status = (
        "phase124_matbench_expt_gap_focused_review_ready_low_capacity_mechanism_gate"
        if phase124_ready
        else "phase124_matbench_expt_gap_focused_review_closed_split_sensitivity_or_shortcut"
    )
    _write_json(
        phase124_dir / "phase124_matbench_expt_gap_focused_review_gate.json",
        {
            "status": status,
            "selected_target": "gap_expt_ev",
            "original_best_admissible_profile": "chemistry_descriptors",
            "original_best_admissible_method": "extra_trees",
            "original_best_admissible_val_rmse": phase124_guard_val,
            "original_best_admissible_test_rmse": phase124_guard_test,
            "original_nearest_neighbor_val_rmse": phase124_guard_val * 1.4,
            "original_nearest_neighbor_test_rmse": phase124_guard_test * 1.4,
            "phase124_model_mechanism_allowed": phase124_ready,
            "phase124_model_training_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
    )
    pd.DataFrame(
        [
            {
                "split_id": "phase123_registered_split",
                "best_admissible_val_rmse": phase124_guard_val,
                "best_admissible_test_rmse": phase124_guard_test,
            }
        ]
    ).to_csv(phase124_dir / "phase124_matbench_expt_gap_split_sensitivity_table.csv", index=False)
    return phase123_dir, phase124_dir


def test_phase125_can_open_focused_validation_without_training(tmp_path: Path):
    module = _load_module()
    phase123_dir, phase124_dir = _write_inputs(tmp_path, phase124_guard_val=0.5, phase124_guard_test=0.5)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase123_dir=phase123_dir,
        phase124_dir=phase124_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 125
    assert gate["status"] == "phase125_matbench_expt_gap_low_capacity_mechanism_ready_focused_validation"
    assert gate["phase125_focused_validation_allowed"] is True
    assert gate["phase125_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase125_matbench_expt_gap_mechanism_coefficient_table.csv").exists()


def test_phase125_closes_when_low_capacity_does_not_beat_phase124_guard(tmp_path: Path):
    module = _load_module()
    phase123_dir, phase124_dir = _write_inputs(tmp_path, phase124_guard_val=0.0, phase124_guard_test=0.0)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase123_dir=phase123_dir,
        phase124_dir=phase124_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase125_matbench_expt_gap_low_capacity_mechanism_closed_no_guarded_gain"
    assert "phase124_validation_guard_gain" in gate["blocking_audits"]
    assert gate["phase125_model_mechanism_allowed"] is False
    assert gate["phase125_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False


def test_phase125_blocks_if_phase124_gate_is_closed(tmp_path: Path):
    module = _load_module()
    phase123_dir, phase124_dir = _write_inputs(
        tmp_path,
        phase124_guard_val=0.5,
        phase124_guard_test=0.5,
        phase124_ready=False,
    )

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase123_dir=phase123_dir,
        phase124_dir=phase124_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase125_matbench_expt_gap_low_capacity_mechanism_blocked_by_phase124"
    assert gate["phase125_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
