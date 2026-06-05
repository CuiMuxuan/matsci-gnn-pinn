from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_module():
    script = Path("scripts/server/build_phase122_matbench_steels_low_capacity_mechanism_gate.py")
    spec = importlib.util.spec_from_file_location("phase122_matbench_steels", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODEL_SPECS = (
        module.ModelSpec("ordinary_least_squares"),
        module.ModelSpec("ridge", alpha=1.0),
    )
    module.MECHANISM_PROFILES = {
        "minor_element_linear": module.MECHANISM_PROFILES["minor_element_linear"],
        "mechanism_compact": module.MECHANISM_PROFILES["mechanism_compact"],
    }
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rows(n: int = 125) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(n):
        c = 0.001 + (index % 5) * 0.0008
        ni = 0.06 + (index % 7) * 0.01
        co = 0.04 + (index % 3) * 0.012
        cr = 0.05 + (index % 4) * 0.008
        mo = 0.004 + (index % 6) * 0.002
        v = 0.001 * (index % 2)
        nb = 0.0005 * (index % 3)
        w = 0.0007 * (index % 4)
        al = 0.003 + (index % 2) * 0.002
        ti = 0.002 + (index % 3) * 0.001
        mn = 0.002
        si = 0.002
        n_frac = 0.0005
        fe = 1.0 - sum([c, ni, co, cr, mo, v, nb, w, al, ti, mn, si, n_frac])
        target = 500.0 + 2500.0 * c + 800.0 * ni + 400.0 * co + 1200.0 * mo + 600.0 * ti
        rows.append(
            {
                "phase120_row_id": f"SYN-{index:04d}",
                "composition": f"synthetic-{index}",
                "yield_strength_mpa": target,
                "dominant_non_fe_element": "Ni",
                "alloy_family_key": ["Ni", "Co+Cr", "Co+Mo", "Ni+Ti", "Cr+Mo"][index % 5],
                "composition_hash16": f"h{index:04d}",
                "element_count": 13,
                "non_fe_fraction": 1.0 - fe,
                "transition_fraction": fe + cr + ni + mn + co + v,
                "light_fraction": c + n_frac + al + si,
                "refractory_fraction": mo + w + nb + ti + v,
                "entropy_fraction": 1.0,
                "frac_Fe": fe,
                "frac_C": c,
                "frac_Mn": mn,
                "frac_Si": si,
                "frac_Cr": cr,
                "frac_Ni": ni,
                "frac_Mo": mo,
                "frac_V": v,
                "frac_N": n_frac,
                "frac_Nb": nb,
                "frac_Co": co,
                "frac_W": w,
                "frac_Al": al,
                "frac_Ti": ti,
            }
        )
    return rows


def _write_inputs(path: Path, *, phase121_guard_val: float, phase121_guard_test: float, phase121_ready: bool = True) -> tuple[Path, Path]:
    phase120_dir = path / "phase120"
    phase121_dir = path / "phase121"
    rows = _rows()
    phase120_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(phase120_dir / "phase120_matbench_steels_field_table.csv", index=False)
    splits = {
        "train": [index for index in range(len(rows)) if index % 5 not in {3, 4}],
        "val": [index for index in range(len(rows)) if index % 5 == 3],
        "test": [index for index in range(len(rows)) if index % 5 == 4],
    }
    _write_json(
        phase120_dir / "phase120_matbench_steels_split_manifest.json",
        {
            "split_strategy": "synthetic_group_split",
            "n_groups": 5,
            "splits": splits,
            "group_splits": {"train": [], "val": [], "test": []},
            "leakage_safe": True,
        },
    )
    _write_json(
        phase120_dir / "phase120_matbench_steels_gate.json",
        {
            "status": "phase120_matbench_steels_gap_ready_focused_review",
            "selected_target": "yield_strength_mpa",
            "selected_profile": "all_element_fractions",
            "selected_method": "extra_trees",
            "selected_validation_rmse": 70.0,
            "selected_test_rmse": 70.0,
            "phase120_model_training_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
    )
    phase121_dir.mkdir(parents=True, exist_ok=True)
    status = (
        "phase121_matbench_steels_focused_review_ready_low_capacity_mechanism_gate"
        if phase121_ready
        else "phase121_matbench_steels_focused_review_closed_split_sensitivity_or_shortcut"
    )
    _write_json(
        phase121_dir / "phase121_matbench_steels_focused_review_gate.json",
        {
            "status": status,
            "selected_target": "yield_strength_mpa",
            "original_best_admissible_profile": "minor_elements_only",
            "original_best_admissible_method": "extra_trees",
            "original_best_admissible_val_rmse": phase121_guard_val,
            "original_best_admissible_test_rmse": phase121_guard_test,
            "phase121_model_mechanism_allowed": phase121_ready,
            "phase121_model_training_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
    )
    pd.DataFrame(
        [
            {
                "split_id": "phase120_registered_split",
                "best_admissible_val_rmse": phase121_guard_val,
                "best_admissible_test_rmse": phase121_guard_test,
            }
        ]
    ).to_csv(phase121_dir / "phase121_matbench_steels_split_sensitivity_table.csv", index=False)
    return phase120_dir, phase121_dir


def test_phase122_can_open_focused_validation_without_training(tmp_path: Path):
    module = _load_module()
    phase120_dir, phase121_dir = _write_inputs(tmp_path, phase121_guard_val=40.0, phase121_guard_test=40.0)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase120_dir=phase120_dir,
        phase121_dir=phase121_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 122
    assert gate["status"] == "phase122_matbench_steels_low_capacity_mechanism_ready_focused_validation"
    assert gate["phase122_focused_validation_allowed"] is True
    assert gate["phase122_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase122_matbench_steels_mechanism_coefficient_table.csv").exists()


def test_phase122_closes_when_low_capacity_does_not_beat_phase121_guard(tmp_path: Path):
    module = _load_module()
    phase120_dir, phase121_dir = _write_inputs(tmp_path, phase121_guard_val=0.0, phase121_guard_test=0.0)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase120_dir=phase120_dir,
        phase121_dir=phase121_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase122_matbench_steels_low_capacity_mechanism_closed_no_guarded_gain"
    assert "phase121_validation_guard_gain" in gate["blocking_audits"]
    assert gate["phase122_model_mechanism_allowed"] is False
    assert gate["phase122_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False


def test_phase122_blocks_if_phase121_gate_is_closed(tmp_path: Path):
    module = _load_module()
    phase120_dir, phase121_dir = _write_inputs(
        tmp_path,
        phase121_guard_val=40.0,
        phase121_guard_test=40.0,
        phase121_ready=False,
    )

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase120_dir=phase120_dir,
        phase121_dir=phase121_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase122_matbench_steels_low_capacity_mechanism_blocked_by_phase121"
    assert gate["phase122_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
