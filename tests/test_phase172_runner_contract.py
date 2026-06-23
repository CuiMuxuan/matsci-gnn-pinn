from pathlib import Path


def test_phase172_runner_is_design_only_gate():
    script = Path(
        "scripts/server/run_phase172_trainable_hidden_closure_smoke_design_gate.sh"
    ).read_text(encoding="utf-8")

    assert "build_phase172_trainable_hidden_closure_smoke_design_gate.py" in script
    assert "phase172_trainable_hidden_closure_smoke_design_manifest.json" in script
    assert "trainable_hidden_closure_phase172_gate" in script
    assert "phase173_low_budget_trainable_smoke_allowed" in script
    assert "phase172_model_mechanism_allowed" in script
    assert "phase172_model_training_allowed" in script
    assert "phase173_training_allowed_now" in script
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
