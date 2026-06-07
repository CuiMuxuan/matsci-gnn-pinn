from pathlib import Path


def test_phase146_paper_evidence_runner_is_no_training_refresh():
    script = Path("scripts/server/run_phase146_paper_evidence_refresh.sh").read_text(
        encoding="utf-8"
    )
    assert "build_phase146_paper_evidence_refresh.py" in script
    assert "phase146_paper_evidence_refresh_manifest.json" in script
    assert "paper_evidence_refresh_phase146_gate" in script
    assert "phase146_model_mechanism_allowed" in script
    assert "phase146_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "data/raw/nist_ammt" not in script
    assert "data/raw/external" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
