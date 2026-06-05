from pathlib import Path


def test_phase130_matbench_log_kvrh_runner_is_no_training_baseline_gate():
    script = Path("scripts/server/run_phase130_matbench_log_kvrh_baseline_gate.sh").read_text(
        encoding="utf-8"
    )
    assert "build_phase130_matbench_log_kvrh_baseline_gate.py" in script
    assert "phase130_matbench_log_kvrh_manifest.json" in script
    assert "matbench_log_kvrh_gate" in script
    assert "phase130_model_mechanism_allowed" in script
    assert "phase130_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "matbench_log_kvrh" in script
    assert "data/raw/nist_ammt" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script


