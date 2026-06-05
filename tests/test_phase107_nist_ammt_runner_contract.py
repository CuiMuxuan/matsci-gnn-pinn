from __future__ import annotations

from pathlib import Path


def test_phase107_source_region_runner_is_no_training_gate():
    script = Path("scripts/server/run_phase107_nist_ammt_source_region_feature_gate_a100.sh").read_text(
        encoding="utf-8"
    )

    assert "phase107_nist_ammt_source_region_feature_gate.py" in script
    assert "phase107_nist_ammt_source_region_feature_gate_a100_manifest.json" in script
    assert "selected_feature_profile" in script
    assert "phase107_focused_review_allowed" in script
    assert "phase107_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
