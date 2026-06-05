from pathlib import Path


def test_phase120_matbench_steels_runner_is_no_training_baseline_gate():
    script = Path("scripts/server/run_phase120_matbench_steels_baseline_gate.sh").read_text(
        encoding="utf-8"
    )
    assert "build_phase120_matbench_steels_baseline_gate.py" in script
    assert "phase120_matbench_steels_baseline_gate_manifest.json" in script
    assert "matbench_steels_baseline_gate" in script
    assert "phase120_focused_review_allowed" in script
    assert "phase120_model_mechanism_allowed" in script
    assert "phase120_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "data/raw/nist_ammt" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
