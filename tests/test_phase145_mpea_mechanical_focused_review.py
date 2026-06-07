from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase145_mpea_mechanical_focused_review.py")
    spec = importlib.util.spec_from_file_location("phase145_mpea", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.phase144.MODEL_METHODS = ("knn",)
    module.phase144.PROFILE_METHODS = {
        "formula_hash_shortcut": ("knn",),
        "reference_shortcut": ("knn",),
        "dominant_element_shortcut": ("knn",),
        "process_only_control": ("knn",),
    }
    return module


def test_phase145_reviews_phase144_mpea_artifacts_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    phase144_dir = Path("docs/results/phase144_mpea_mechanical_baseline_gate")

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase144_dir=phase144_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 145
    assert gate["selected_target"] == "hardness_hv"
    assert gate["viable_split_reviews"] >= 1
    assert gate["phase145_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase145_mpea_mechanical_focused_review_gate.json").exists()
    assert (tmp_path / "out" / "phase145_mpea_mechanical_focused_profile_table.csv").exists()
    markdown = (tmp_path / "out" / "phase145_mpea_mechanical_focused_review.md").read_text(
        encoding="utf-8"
    )
    assert "Phase 145 MPEA Mechanical Focused Review" in markdown
    assert "registered_formula_hash" in markdown


def test_phase145_blocks_if_phase144_gate_is_not_ready(tmp_path: Path):
    module = _load_module()
    phase144_src = Path("docs/results/phase144_mpea_mechanical_baseline_gate")
    phase144_dir = tmp_path / "phase144"
    shutil.copytree(phase144_src, phase144_dir)
    gate_path = phase144_dir / "phase144_mpea_mechanical_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["status"] = "phase144_mpea_mechanical_closed_no_stable_guarded_gap"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase144_dir=phase144_dir,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase145_mpea_mechanical_focused_review_closed_split_sensitivity_or_shortcut"
    assert "phase144_gate_consistency" in gate["blocking_audits"]
    assert gate["phase145_model_mechanism_allowed"] is False
    assert gate["phase145_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
