from __future__ import annotations

from pathlib import Path


def test_phase110_layer_mean_runner_is_no_training_gate():
    script = Path(
        "scripts/server/run_phase110_nist_ammt_layer_mean_target_review_gate_a100.sh"
    ).read_text(encoding="utf-8")

    assert "phase110_nist_ammt_layer_mean_target_review_gate.py" in script
    assert "phase110_nist_ammt_layer_mean_target_review_gate_a100_manifest.json" in script
    assert "layer_time_shortcut_detected" in script
    assert "phase110_model_mechanism_allowed" in script
    assert "phase110_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
