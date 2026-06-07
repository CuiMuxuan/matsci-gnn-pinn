from pathlib import Path


def test_phase149_runner_is_no_training_readiness_gate():
    script = Path("scripts/server/run_phase149_neural_operator_readiness_gate.sh").read_text(
        encoding="utf-8"
    )
    assert "build_phase149_neural_operator_readiness_gate.py" in script
    assert "phase149_neural_operator_readiness_manifest.json" in script
    assert "neural_operator_readiness_phase149_gate" in script
    assert "operator_training_allowed_now" in script
    assert "phase149_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "data/raw/nist_ammt" not in script
    assert "data/raw/external" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
