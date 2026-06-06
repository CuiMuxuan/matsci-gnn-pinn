from pathlib import Path


def test_phase134_matbench_log_gvrh_runner_is_no_training_focused_review():
    script = Path("scripts/server/run_phase134_matbench_log_gvrh_focused_review.sh").read_text(
        encoding="utf-8"
    )
    assert "build_phase134_matbench_log_gvrh_focused_review.py" in script
    assert "phase134_matbench_log_gvrh_focused_review_manifest.json" in script
    assert "matbench_log_gvrh_focused_review_gate" in script
    assert "phase134_model_mechanism_allowed" in script
    assert "phase134_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "phase129_matbench_log_gvrh_baseline_gate" in script
    assert "data/raw/nist_ammt" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
