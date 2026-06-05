from __future__ import annotations

from pathlib import Path


def test_phase113_melt_pool_review_runner_is_no_training_review():
    script = Path("scripts/server/run_phase113_nist_ammt_melt_pool_focused_review_a100.sh").read_text(
        encoding="utf-8"
    )

    assert "build_phase113_nist_ammt_melt_pool_focused_review.py" in script
    assert "phase113_nist_ammt_melt_pool_focused_review_a100_manifest.json" in script
    assert "phase113_model_mechanism_allowed" in script
    assert "phase113_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "data/raw/nist_ammt" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
