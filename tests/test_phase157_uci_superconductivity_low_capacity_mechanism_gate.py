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


def _load_phase156():
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
    )
    return module


def _load_module():
    script = Path("scripts/server/build_phase157_uci_superconductivity_low_capacity_mechanism_gate.py")
    spec = importlib.util.spec_from_file_location("phase157_uci_superconductivity", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.p155.EXPECTED_MIN_BYTES = 1
    module.MODEL_SPECS = (
        module.ModelSpec("ordinary_least_squares"),
        module.ModelSpec("ridge", alpha=1.0),
    )
    module.MECHANISM_PROFILES = {
        "weighted_property_linear": module.MECHANISM_PROFILES["weighted_property_linear"],
        "mechanism_compact": module.MECHANISM_PROFILES["mechanism_compact"],
    }
    return module


def _write_phase156(tmp_path: Path, *, ready: bool = True, guard: float = 25.0) -> tuple[Path, Path]:
    phase155 = _load_phase155()
    phase156 = _load_phase156()
    raw_path = _synthetic_source(tmp_path / "raw" / "superconductivty_data.zip")
    phase155_dir = tmp_path / "phase155"
    phase155_manifest = phase155.build_package(
        root=Path(".").resolve(),
        output_dir=phase155_dir,
        raw_path=raw_path,
        source_url="https://example.invalid/superconductivty_data.zip",
        allow_download=False,
    )
    phase155_gate = phase155_manifest["gate"]
    phase155_gate["status"] = "phase155_uci_superconductivity_ready_focused_review"
    phase155_gate["phase156_focused_review_allowed"] = True
    (phase155_dir / "phase155_uci_superconductivity_baseline_gate.json").write_text(
        json.dumps(phase155_gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    phase156_dir = tmp_path / "phase156"
    phase156_manifest = phase156.build_package(
        root=Path(".").resolve(),
        phase155_dir=phase155_dir,
        raw_path=raw_path,
        output_dir=phase156_dir,
    )
    gate = phase156_manifest["gate"]
    gate["status"] = (
        "phase156_uci_superconductivity_focused_review_ready_low_capacity_mechanism_gate"
        if ready
        else "phase156_uci_superconductivity_focused_review_closed_split_sensitivity_or_shortcut"
    )
    gate["phase156_model_mechanism_allowed"] = ready
    gate["registered_replay_validation_rmse"] = guard
    gate["registered_replay_test_rmse"] = guard
    (phase156_dir / "phase156_uci_superconductivity_focused_review_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return phase156_dir, raw_path


def test_phase157_runs_low_capacity_gate_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    phase156_dir, raw_path = _write_phase156(tmp_path, ready=True, guard=30.0)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase156_dir=phase156_dir,
        raw_path=raw_path,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 157
    assert gate["status"] in {
        "phase157_uci_superconductivity_low_capacity_mechanism_ready_focused_validation",
        "phase157_uci_superconductivity_low_capacity_mechanism_closed_no_guarded_gain",
    }
    assert gate["phase157_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase157_uci_superconductivity_candidate_table.csv").exists()
    assert (tmp_path / "out" / "phase157_uci_superconductivity_coefficient_table.csv").exists()


def test_phase157_blocks_if_phase156_gate_is_closed(tmp_path: Path):
    module = _load_module()
    phase156_dir, raw_path = _write_phase156(tmp_path, ready=False, guard=30.0)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase156_dir=phase156_dir,
        raw_path=raw_path,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase157_uci_superconductivity_low_capacity_mechanism_closed_no_guarded_gain"
    assert "phase156_gate_consistency" in gate["blocking_audits"]
    assert gate["phase157_model_mechanism_allowed"] is False
    assert gate["phase157_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False


def test_phase157_adds_named_superconductivity_features(tmp_path: Path):
    module = _load_module()
    raw_path = _synthetic_source(tmp_path / "raw" / "superconductivty_data.zip")
    frame = module.p155.load_superconductivity_table(raw_path)
    enriched, schema = module.add_mechanism_features(frame)

    assert "phase157_cu_o_interaction" in enriched.columns
    assert "phase157_fe_as_interaction" in enriched.columns
    assert "phase157_fraction_entropy" in enriched.columns
    assert "phase157_cuprate_layer_proxy" in enriched.columns
    assert any(row["feature"] == "phase157_cu_o_interaction" for row in schema)
