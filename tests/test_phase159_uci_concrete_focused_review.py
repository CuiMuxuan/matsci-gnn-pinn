from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from tests.test_phase158_uci_concrete_baseline_gate import _synthetic_source


def _load_phase158():
    script = Path("scripts/server/build_phase158_uci_concrete_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase158_uci_concrete", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXPECTED_MIN_BYTES = 1
    module.MIN_ROWS_FOR_REVIEW = 80
    module.MIN_SPLIT_ROWS = 5
    module.MODEL_METHODS = ("knn",)
    return module


def _load_module():
    script = Path("scripts/server/build_phase159_uci_concrete_focused_review.py")
    spec = importlib.util.spec_from_file_location("phase159_uci_concrete", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.p158.EXPECTED_MIN_BYTES = 1
    module.p158.MIN_ROWS_FOR_REVIEW = 80
    module.p158.MIN_SPLIT_ROWS = 5
    module.p158.MODEL_METHODS = ("knn",)
    module.MIN_SPLIT_ROWS = 5
    module.SPLIT_SPECS = (
        ("phase158_registered_mix_design", "mix_design_hash", "phase158_split", True),
        ("mix_design_hash_0", "mix_design_hash", "phase159_mix_design_0", True),
        ("water_binder_bins", "bins:water_binder_ratio", "phase159_water_binder", False),
    )
    return module


def _write_phase158(tmp_path: Path, *, ready: bool = True) -> tuple[Path, Path]:
    phase158 = _load_phase158()
    raw_path = _synthetic_source(tmp_path / "raw" / "concrete_compressive_strength.zip")
    phase158_dir = tmp_path / "phase158"
    manifest = phase158.build_package(
        root=Path(".").resolve(),
        output_dir=phase158_dir,
        raw_path=raw_path,
        source_url="https://example.invalid/concrete.zip",
        allow_download=False,
    )
    gate = manifest["gate"]
    if ready:
        gate["status"] = "phase158_uci_concrete_ready_focused_review"
        gate["phase159_focused_review_allowed"] = True
    else:
        gate["status"] = "phase158_uci_concrete_closed_no_stable_guarded_gap"
        gate["phase159_focused_review_allowed"] = False
    (phase158_dir / "phase158_uci_concrete_baseline_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return phase158_dir, raw_path


def test_phase159_reviews_phase158_candidate_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    phase158_dir, raw_path = _write_phase158(tmp_path, ready=True)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase158_dir=phase158_dir,
        raw_path=raw_path,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 159
    assert gate["status"] in {
        "phase159_uci_concrete_focused_review_ready_low_capacity_mechanism_gate",
        "phase159_uci_concrete_focused_review_closed_split_sensitivity_or_shortcut",
    }
    assert gate["phase159_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase159_uci_concrete_focused_review_gate.json").exists()
    split_table = pd.read_csv(tmp_path / "out" / "phase159_uci_concrete_split_review_table.csv")
    assert "nearest_neighbor_dominant" in split_table.columns
    assert "target_distribution_shift_z" in split_table.columns


def test_phase159_blocks_if_phase158_gate_is_closed(tmp_path: Path):
    module = _load_module()
    phase158_dir, raw_path = _write_phase158(tmp_path, ready=False)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase158_dir=phase158_dir,
        raw_path=raw_path,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase159_uci_concrete_focused_review_closed_split_sensitivity_or_shortcut"
    assert "phase158_gate_consistency" in gate["blocking_audits"]
    assert gate["phase159_model_mechanism_allowed"] is False
    assert gate["phase159_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
