from pathlib import Path


def test_phase119_battery_failure_runner_is_no_training_candidate_sweep():
    script = Path("scripts/server/run_phase119_battery_failure_candidate_sweep.sh").read_text(
        encoding="utf-8"
    )
    assert "build_phase119_battery_failure_candidate_sweep.py" in script
    assert "phase119_battery_failure_candidate_sweep_manifest.json" in script
    assert "battery_failure_candidate_sweep_gate" in script
    assert "phase119_model_mechanism_allowed" in script
    assert "phase119_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "phase117_battery_failure_databank_gate" in script
    assert "phase118_battery_failure_focused_review" in script
    assert "data/raw/nist_ammt" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
