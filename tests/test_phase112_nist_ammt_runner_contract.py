from __future__ import annotations

from pathlib import Path


def test_phase112_melt_pool_runner_is_no_training_gate():
    script = Path("scripts/server/run_phase112_nist_ammt_melt_pool_target_gate_a100.sh").read_text(
        encoding="utf-8"
    )

    assert "phase112_nist_ammt_melt_pool_target_gate.py" in script
    assert "phase112_nist_ammt_melt_pool_target_gate_a100_manifest.json" in script
    assert "--discover-from-zip" in script
    assert "phase112_focused_review_allowed" in script
    assert "phase112_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
