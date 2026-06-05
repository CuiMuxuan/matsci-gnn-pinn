from __future__ import annotations

from pathlib import Path


def test_phase109_focused_review_runner_is_no_training_gate():
    script = Path(
        "scripts/server/run_phase109_nist_ammt_sequence_target_focused_review_gate_a100.sh"
    ).read_text(encoding="utf-8")

    assert "phase109_nist_ammt_sequence_target_focused_review_gate.py" in script
    assert "phase109_nist_ammt_sequence_target_focused_review_gate_a100_manifest.json" in script
    assert "camera_shortcut_detected" in script
    assert "phase109_model_mechanism_allowed" in script
    assert "phase109_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
