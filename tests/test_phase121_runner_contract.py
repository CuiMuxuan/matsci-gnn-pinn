from pathlib import Path


def test_phase121_matbench_steels_runner_is_no_training_focused_review():
    script = Path("scripts/server/run_phase121_matbench_steels_focused_review.sh").read_text(
        encoding="utf-8"
    )
    assert "build_phase121_matbench_steels_focused_review.py" in script
    assert "phase121_matbench_steels_focused_review_manifest.json" in script
    assert "matbench_steels_focused_review_gate" in script
    assert "phase121_model_mechanism_allowed" in script
    assert "phase121_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "phase120_matbench_steels_baseline_gate" in script
    assert "data/raw/nist_ammt" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
