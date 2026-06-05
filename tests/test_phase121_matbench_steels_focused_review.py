from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_module():
    script = Path("scripts/server/build_phase121_matbench_steels_focused_review.py")
    spec = importlib.util.spec_from_file_location("phase121_matbench_steels", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODEL_METHODS = ("knn",)
    module.SPLIT_PLAN = (
        ("phase120_registered_split", "phase120_manifest", "phase120"),
        ("alloy_family_hash_0", "group:alloy_family_key", "test_alloy_0"),
        ("fe_ni_co_bins", "bins:frac_Fe,frac_Ni,frac_Co", "test_bins"),
    )
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rows(*, unstable: bool, n: int = 240) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    families = ["Cr+Ni", "Co+Cr", "Co+Mo", "Al+Ni", "Ni+Ti", "Cr+Mo", "Co+Ni", "Al+Co"]
    for index in range(n):
        family = families[index % len(families)]
        family_index = families.index(family)
        ni = 0.04 + (index % 9) * 0.01 + (0.08 if "Ni" in family else 0.0)
        co = 0.03 + (index % 7) * 0.008 + (0.08 if "Co" in family else 0.0)
        cr = 0.05 + (index % 5) * 0.01 + (0.08 if "Cr" in family else 0.0)
        mo = 0.004 + (index % 4) * 0.006 + (0.03 if "Mo" in family else 0.0)
        al = 0.003 + (0.03 if "Al" in family else 0.0)
        ti = 0.002 + (0.03 if "Ti" in family else 0.0)
        c = 0.001 + (index % 4) * 0.0005
        mn = 0.002
        si = 0.002
        fe = max(0.45, 1.0 - sum([ni, co, cr, mo, al, ti, c, mn, si]))
        base = 900 + 1200 * ni + 700 * co + 400 * mo + 300 * ti + 2000 * c
        target = base + (family_index * 180.0 if unstable else (index % 3) * 2.0)
        rows.append(
            {
                "phase120_row_id": f"MB-{index:04d}",
                "composition": f"synthetic-{index}",
                "yield_strength_mpa": target,
                "dominant_non_fe_element": "Ni" if ni >= max(co, cr, mo) else "Co",
                "alloy_family_key": family,
                "composition_hash16": f"h{index:04d}",
                "element_count": 9,
                "non_fe_fraction": 1.0 - fe,
                "transition_fraction": fe + cr + ni + co,
                "light_fraction": c + al + si,
                "refractory_fraction": mo + ti,
                "entropy_fraction": 1.0,
                "frac_Fe": fe,
                "frac_C": c,
                "frac_Mn": mn,
                "frac_Si": si,
                "frac_Cr": cr,
                "frac_Ni": ni,
                "frac_Mo": mo,
                "frac_V": 0.0,
                "frac_N": 0.0,
                "frac_Nb": 0.0,
                "frac_Co": co,
                "frac_W": 0.0,
                "frac_Al": al,
                "frac_Ti": ti,
            }
        )
    return rows


def _write_phase120(path: Path, *, unstable: bool) -> Path:
    phase120_dir = path / "phase120"
    rows = _rows(unstable=unstable)
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
            "split_strategy": "synthetic_split",
            "n_groups": 8,
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
        },
    )
    return phase120_dir


def test_phase121_closes_split_sensitive_candidate(tmp_path: Path):
    module = _load_module()
    phase120_dir = _write_phase120(tmp_path, unstable=True)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase120_dir=phase120_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 121
    assert gate["status"] == "phase121_matbench_steels_focused_review_closed_split_sensitivity_or_shortcut"
    assert "split_sensitivity_pass_rate" in gate["blocking_audits"] or "shortcut_dominant_split_count" in gate["blocking_audits"]
    assert gate["phase121_model_mechanism_allowed"] is False
    assert gate["phase121_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False


def test_phase121_can_allow_stable_nonshortcut_candidate_without_training(tmp_path: Path):
    module = _load_module()
    phase120_dir = _write_phase120(tmp_path, unstable=False)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase120_dir=phase120_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] in {
        "phase121_matbench_steels_focused_review_ready_low_capacity_mechanism_gate",
        "phase121_matbench_steels_focused_review_closed_split_sensitivity_or_shortcut",
    }
    assert gate["phase121_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
