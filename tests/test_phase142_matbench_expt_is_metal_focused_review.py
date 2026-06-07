from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from tests.test_phase141_matbench_expt_is_metal_baseline_gate import _write_payload


def _load_phase141():
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


def _load_module():
    script = Path("scripts/server/build_phase142_matbench_expt_is_metal_focused_review.py")
    spec = importlib.util.spec_from_file_location("phase142_matbench_expt_is_metal", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.phase139.MODEL_METHODS = ("knn",)
    module.phase139.PROFILE_METHODS = {
        "composition_hash_shortcut": ("knn",),
        "chemistry_family_shortcut": ("knn",),
        "dominant_element_shortcut": ("knn",),
    }
    module.phase139.MIN_SPLIT_ROWS = 8
    module.phase139.SPLIT_PLAN = (
        ("phase141_registered_split", "phase141_manifest", "phase141"),
        ("chemistry_family_hash_0", "group:chemistry_family_key", "test_family_0"),
        ("element_count_bins", "bins:element_count", "test_element_count"),
    )
    return module


def _write_phase141(tmp_path: Path, *, ready: bool = True) -> Path:
    phase141 = _load_phase141()
    raw_path = tmp_path / "raw" / "matbench_expt_is_metal.json.gz"
    _write_payload(raw_path, n_rows=120)
    phase141_dir = tmp_path / "phase141"
    manifest = phase141.build_package(
        root=Path(".").resolve(),
        raw_path=raw_path,
        output_dir=phase141_dir,
        source_url="https://example.invalid/matbench_expt_is_metal.json.gz",
        force_download=False,
    )
    gate = manifest["gate"]
    if ready:
        gate["status"] = "phase141_matbench_expt_is_metal_ready_focused_review"
        gate["phase141_focused_review_allowed"] = True
        gate["selected_target"] = "is_metal"
    else:
        gate["status"] = "phase141_matbench_expt_is_metal_closed_no_stable_guarded_gap"
        gate["phase141_focused_review_allowed"] = False
    (phase141_dir / "phase141_matbench_expt_is_metal_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return phase141_dir


def test_phase142_reviews_phase141_candidate_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    phase141_dir = _write_phase141(tmp_path, ready=True)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase141_dir=phase141_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 142
    assert gate["status"] in {
        "phase142_matbench_expt_is_metal_focused_review_ready_low_capacity_mechanism_gate",
        "phase142_matbench_expt_is_metal_focused_review_closed_split_sensitivity_or_shortcut",
    }
    assert gate["phase142_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase142_matbench_expt_is_metal_split_sensitivity_table.csv").exists()
    split_table = pd.read_csv(tmp_path / "out" / "phase142_matbench_expt_is_metal_split_sensitivity_table.csv")
    assert "class_balance_shift" in split_table.columns
    assert "nearest_neighbor_dominates" in split_table.columns
    audit_table = pd.read_csv(tmp_path / "out" / "phase142_matbench_expt_is_metal_shortcut_audit_table.csv")
    assert {
        "original_split_shortcut_dominance",
        "original_split_nearest_neighbor_dominance",
        "original_split_class_balance",
    }.issubset(set(audit_table["audit"]))


def test_phase142_blocks_if_phase141_gate_is_closed(tmp_path: Path):
    module = _load_module()
    phase141_dir = _write_phase141(tmp_path, ready=False)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase141_dir=phase141_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase142_matbench_expt_is_metal_review_blocked_by_phase141"
    assert gate["phase142_model_mechanism_allowed"] is False
    assert gate["phase142_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
