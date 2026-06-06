from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from tests.test_phase138_matbench_glass_baseline_gate import _write_payload


def _load_phase138():
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


def _load_module():
    script = Path("scripts/server/build_phase139_matbench_glass_focused_review.py")
    spec = importlib.util.spec_from_file_location("phase139_matbench_glass", script)
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
        ("phase138_registered_split", "phase138_manifest", "phase138"),
        ("chemistry_family_hash_0", "group:chemistry_family_key", "test_family_0"),
        ("element_count_bins", "bins:element_count", "test_element_count"),
    )
    return module


def _write_phase138(tmp_path: Path, *, ready: bool = True) -> Path:
    phase138 = _load_phase138()
    raw_path = tmp_path / "raw" / "matbench_glass.json.gz"
    _write_payload(raw_path, n_rows=120)
    phase138_dir = tmp_path / "phase138"
    manifest = phase138.build_package(
        root=Path(".").resolve(),
        raw_path=raw_path,
        output_dir=phase138_dir,
        source_url="https://example.invalid/matbench_glass.json.gz",
        force_download=False,
    )
    if ready:
        gate = manifest["gate"]
        gate["status"] = "phase138_matbench_glass_ready_focused_review"
        gate["phase138_focused_review_allowed"] = True
        gate["selected_target"] = "gfa"
    else:
        gate = manifest["gate"]
        gate["status"] = "phase138_matbench_glass_closed_no_stable_guarded_gap"
        gate["phase138_focused_review_allowed"] = False
    (phase138_dir / "phase138_matbench_glass_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return phase138_dir


def test_phase139_reviews_phase138_candidate_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    phase138_dir = _write_phase138(tmp_path, ready=True)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase138_dir=phase138_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 139
    assert gate["status"] in {
        "phase139_matbench_glass_focused_review_ready_low_capacity_mechanism_gate",
        "phase139_matbench_glass_focused_review_closed_split_sensitivity_or_shortcut",
    }
    assert gate["phase139_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase139_matbench_glass_split_sensitivity_table.csv").exists()
    split_table = pd.read_csv(tmp_path / "out" / "phase139_matbench_glass_split_sensitivity_table.csv")
    assert "class_balance_shift" in split_table.columns
    assert "nearest_neighbor_dominates" in split_table.columns


def test_phase139_blocks_if_phase138_gate_is_closed(tmp_path: Path):
    module = _load_module()
    phase138_dir = _write_phase138(tmp_path, ready=False)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase138_dir=phase138_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase139_matbench_glass_review_blocked_by_phase138"
    assert gate["phase139_model_mechanism_allowed"] is False
    assert gate["phase139_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
