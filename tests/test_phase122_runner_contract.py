from pathlib import Path


def test_phase122_matbench_steels_runner_is_no_training_low_capacity_gate():
    script = Path("scripts/server/run_phase122_matbench_steels_low_capacity_mechanism_gate.sh").read_text(
        encoding="utf-8"
    )
    assert "build_phase122_matbench_steels_low_capacity_mechanism_gate.py" in script
    assert "phase122_matbench_steels_low_capacity_mechanism_manifest.json" in script
    assert "matbench_steels_low_capacity_mechanism_gate" in script
    assert "phase122_model_mechanism_allowed" in script
    assert "phase122_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "phase120_matbench_steels_baseline_gate" in script
    assert "phase121_matbench_steels_focused_review" in script
    assert "data/raw/nist_ammt" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
