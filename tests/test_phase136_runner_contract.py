from pathlib import Path


def test_phase136_matbench_perovskites_runner_is_no_training_focused_review():
    script = Path("scripts/server/run_phase136_matbench_perovskites_focused_review.sh").read_text(
        encoding="utf-8"
    )
    assert "build_phase136_matbench_perovskites_focused_review.py" in script
    assert "phase136_matbench_perovskites_focused_review_manifest.json" in script
    assert "matbench_perovskites_focused_review_gate" in script
    assert "phase136_model_mechanism_allowed" in script
    assert "phase136_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "phase135_matbench_perovskites_baseline_gate" in script
    assert "data/raw/nist_ammt" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
