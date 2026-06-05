from pathlib import Path


def test_phase125_matbench_expt_gap_runner_is_no_training_mechanism_gate():
    script = Path("scripts/server/run_phase125_matbench_expt_gap_low_capacity_mechanism_gate.sh").read_text(
        encoding="utf-8"
    )
    assert "build_phase125_matbench_expt_gap_low_capacity_mechanism_gate.py" in script
    assert "phase125_matbench_expt_gap_low_capacity_mechanism_manifest.json" in script
    assert "matbench_expt_gap_low_capacity_mechanism_gate" in script
    assert "phase125_model_mechanism_allowed" in script
    assert "phase125_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "phase123_matbench_expt_gap_baseline_gate" in script
    assert "phase124_matbench_expt_gap_focused_review" in script
    assert "data/raw/nist_ammt" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
