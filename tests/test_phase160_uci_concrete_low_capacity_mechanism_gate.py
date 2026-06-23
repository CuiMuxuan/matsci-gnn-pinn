from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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


def _load_phase159():
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
    )
    return module


def _load_module():
    script = Path("scripts/server/build_phase160_uci_concrete_low_capacity_mechanism_gate.py")
    spec = importlib.util.spec_from_file_location("phase160_uci_concrete", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODEL_SPECS = (
        module.ModelSpec("ordinary_least_squares"),
        module.ModelSpec("ridge", alpha=1.0),
    )
    module.MECHANISM_PROFILES = {
        "abrams_age_core": module.MECHANISM_PROFILES["abrams_age_core"],
        "mechanism_compact": module.MECHANISM_PROFILES["mechanism_compact"],
    }
    return module


def _write_phase159(tmp_path: Path, *, ready: bool = True, guard: float = 10.0) -> tuple[Path, Path]:
    phase158 = _load_phase158()
    phase159 = _load_phase159()
    raw_path = _synthetic_source(tmp_path / "raw" / "concrete_compressive_strength.zip")
    phase158_dir = tmp_path / "phase158"
    phase158_manifest = phase158.build_package(
        root=Path(".").resolve(),
        output_dir=phase158_dir,
        raw_path=raw_path,
        source_url="https://example.invalid/concrete.zip",
        allow_download=False,
    )
    phase158_gate = phase158_manifest["gate"]
    phase158_gate["status"] = "phase158_uci_concrete_ready_focused_review"
    phase158_gate["phase159_focused_review_allowed"] = True
    (phase158_dir / "phase158_uci_concrete_baseline_gate.json").write_text(
        json.dumps(phase158_gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    phase159_dir = tmp_path / "phase159"
    phase159_manifest = phase159.build_package(
        root=Path(".").resolve(),
        phase158_dir=phase158_dir,
        raw_path=raw_path,
        output_dir=phase159_dir,
    )
    gate = phase159_manifest["gate"]
    gate["status"] = (
        "phase159_uci_concrete_focused_review_ready_low_capacity_mechanism_gate"
        if ready
        else "phase159_uci_concrete_focused_review_closed_split_sensitivity_or_shortcut"
    )
    gate["phase159_model_mechanism_allowed"] = ready
    gate["registered_replay_validation_rmse"] = guard
    gate["registered_replay_test_rmse"] = guard
    (phase159_dir / "phase159_uci_concrete_focused_review_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return phase159_dir, raw_path


def test_phase160_runs_low_capacity_gate_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    phase159_dir, raw_path = _write_phase159(tmp_path, ready=True, guard=20.0)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase159_dir=phase159_dir,
        raw_path=raw_path,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 160
    assert gate["status"] in {
        "phase160_uci_concrete_low_capacity_mechanism_ready_focused_validation",
        "phase160_uci_concrete_low_capacity_mechanism_closed_no_guarded_gain",
    }
    assert gate["phase160_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase160_uci_concrete_candidate_table.csv").exists()
    assert (tmp_path / "out" / "phase160_uci_concrete_coefficient_table.csv").exists()


def test_phase160_blocks_if_phase159_gate_is_closed(tmp_path: Path):
    module = _load_module()
    phase159_dir, raw_path = _write_phase159(tmp_path, ready=False, guard=20.0)

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase159_dir=phase159_dir,
        raw_path=raw_path,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase160_uci_concrete_low_capacity_mechanism_closed_no_guarded_gain"
    assert "phase159_gate_consistency" in gate["blocking_audits"]
    assert gate["phase160_model_mechanism_allowed"] is False
    assert gate["phase160_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False


def test_phase160_adds_named_concrete_mechanism_features(tmp_path: Path):
    module = _load_module()
    raw_path = _synthetic_source(tmp_path / "raw" / "concrete_compressive_strength.zip")
    frame = module.p158.load_concrete_table(raw_path)
    enriched, schema = module.add_mechanism_features(frame)

    assert "phase160_effective_water_binder_ratio" in enriched.columns
    assert "phase160_water_binder_age_interaction" in enriched.columns
    assert "phase160_scm_age_interaction" in enriched.columns
    assert "phase160_paste_aggregate_ratio" in enriched.columns
    assert any(row["feature"] == "phase160_water_binder_age_interaction" for row in schema)
