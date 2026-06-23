from pathlib import Path


def test_phase174_runner_is_design_only_gate():
    script = Path(
        "scripts/server/run_phase174_low_capacity_hidden_closure_design_gate.sh"
    ).read_text(encoding="utf-8")

    assert "build_phase174_low_capacity_hidden_closure_design_gate.py" in script
    assert "phase174_low_capacity_hidden_closure_design_manifest.json" in script
    assert "low_capacity_hidden_closure_phase174_gate" in script
    assert "phase175_low_capacity_smoke_allowed" in script
    assert "phase174_model_mechanism_allowed" in script
    assert "phase174_model_training_allowed" in script
    assert "phase175_training_allowed_now" in script
    assert "bayesian_pinn_training_allowed_now" in script
    assert "adaptive_sampling_training_allowed_now" in script
    assert "gcn_pinn_training_allowed_now" in script
    assert "cnn_operator_training_allowed_now" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "data/raw/nist_ammt" not in script
    assert "data/raw/ambench" not in script
    assert "data/raw/external" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
