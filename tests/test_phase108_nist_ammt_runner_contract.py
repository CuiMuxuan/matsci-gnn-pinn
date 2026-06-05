from __future__ import annotations

from pathlib import Path


def test_phase108_sequence_target_runner_is_no_training_gate():
    script = Path("scripts/server/run_phase108_nist_ammt_sequence_target_gate_a100.sh").read_text(
        encoding="utf-8"
    )

    assert "phase108_nist_ammt_sequence_target_gate.py" in script
    assert "phase108_nist_ammt_sequence_target_gate_a100_manifest.json" in script
    assert "selected_target" in script
    assert "phase108_focused_review_allowed" in script
    assert "phase108_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
