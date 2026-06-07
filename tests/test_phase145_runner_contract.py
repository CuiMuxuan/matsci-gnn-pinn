from pathlib import Path


def test_phase145_mpea_mechanical_runner_is_no_training_focused_review():
    script = Path("scripts/server/run_phase145_mpea_mechanical_focused_review.sh").read_text(
        encoding="utf-8"
    )
    assert "build_phase145_mpea_mechanical_focused_review.py" in script
    assert "phase145_mpea_mechanical_focused_review_manifest.json" in script
    assert "mpea_mechanical_phase145_gate" in script
    assert "phase145_model_mechanism_allowed" in script
    assert "phase145_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "phase144_mpea_mechanical_baseline_gate" in script
    assert "data/raw/nist_ammt" not in script
    assert "data/raw/external/mpea_dataset" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
