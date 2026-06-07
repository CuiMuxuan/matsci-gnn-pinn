from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from tests.test_phase155_uci_superconductivity_baseline_gate import _synthetic_source


def _load_phase155():
    script = Path("scripts/server/build_phase155_uci_superconductivity_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase155_uci_superconductivity", script)
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
    script = Path("scripts/server/build_phase156_uci_superconductivity_focused_review.py")
    spec = importlib.util.spec_from_file_location("phase156_uci_superconductivity", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.p155.EXPECTED_MIN_BYTES = 1
    module.p155.MIN_ROWS_FOR_REVIEW = 80
    module.p155.MIN_SPLIT_ROWS = 5
    module.p155.MODEL_METHODS = ("knn",)
    module.MIN_SPLIT_ROWS = 5
    module.SPLIT_SPECS = (
        ("phase155_registered_element_set", "element_set_hash", "phase155_split", True),
        ("element_set_hash_0", "element_set_hash", "phase156_element_set_0", True),
        ("number_of_elements_bins", "bins:number_of_elements", "phase156_number_elements", False),
    )
    return module


def _write_phase155(tmp_path: Path, *, ready: bool = True) -> tuple[Path, Path]:
    phase155 = _load_phase155()
    raw_path = _synthetic_source(tmp_path / "raw" / "superconductivty_data.zip")
    phase155_dir = tmp_path / "phase155"
    manifest = phase155.build_package(
        root=Path(".").resolve(),
        output_dir=phase155_dir,
        raw_path=raw_path,
        source_url="https://example.invalid/superconductivty_data.zip",
        allow_download=False,
    )
    gate = manifest["gate"]
    if ready:
        gate["status"] = "phase155_uci_superconductivity_ready_focused_review"
        gate["phase156_focused_review_allowed"] = True
    else:
        gate["status"] = "phase155_uci_superconductivity_closed_no_stable_guarded_gap"
        gate["phase156_focused_review_allowed"] = False
    (phase155_dir / "phase155_uci_superconductivity_baseline_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return phase155_dir, raw_path


def test_phase156_reviews_phase155_candidate_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    phase155_dir, raw_path = _write_phase155(tmp_path, ready=True)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase155_dir=phase155_dir,
        raw_path=raw_path,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 156
    assert gate["status"] in {
        "phase156_uci_superconductivity_focused_review_ready_low_capacity_mechanism_gate",
        "phase156_uci_superconductivity_focused_review_closed_split_sensitivity_or_shortcut",
    }
    assert gate["phase156_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase156_uci_superconductivity_focused_review_gate.json").exists()
    split_table = pd.read_csv(tmp_path / "out" / "phase156_uci_superconductivity_split_review_table.csv")
    assert "nearest_neighbor_dominant" in split_table.columns
    assert "target_distribution_shift_z" in split_table.columns


def test_phase156_blocks_if_phase155_gate_is_closed(tmp_path: Path):
    module = _load_module()
    phase155_dir, raw_path = _write_phase155(tmp_path, ready=False)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase155_dir=phase155_dir,
        raw_path=raw_path,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase156_uci_superconductivity_focused_review_closed_split_sensitivity_or_shortcut"
    assert "phase155_gate_consistency" in gate["blocking_audits"]
    assert gate["phase156_model_mechanism_allowed"] is False
    assert gate["phase156_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
