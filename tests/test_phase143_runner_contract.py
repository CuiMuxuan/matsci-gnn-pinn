from __future__ import annotations

from pathlib import Path


def test_phase143_paper_evidence_refresh_runner_is_no_training_package():
    script = Path("scripts/server/run_phase143_paper_evidence_refresh.sh").read_text(
        encoding="utf-8"
    )

    assert "build_phase143_paper_evidence_refresh.py" in script
    assert "phase143_paper_evidence_refresh_manifest.json" in script
    assert "paper_evidence_refresh_phase143_gate" in script
    assert "phase143_model_mechanism_allowed" in script
    assert "phase143_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "data/raw" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
