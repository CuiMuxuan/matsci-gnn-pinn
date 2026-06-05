from __future__ import annotations

from pathlib import Path


def test_phase106_spatial_target_runner_is_no_training_gate():
    script = Path(
        "scripts/server/run_phase106_nist_ammt_spatial_target_representation_gate_a100.sh"
    ).read_text(encoding="utf-8")

    assert "phase106_nist_ammt_spatial_target_representation_gate.py" in script
    assert "phase106_nist_ammt_spatial_target_representation_gate_a100_manifest.json" in script
    assert "selected_target" in script
    assert "phase106_seed7_focused_validation_allowed" in script
    assert "phase106_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
