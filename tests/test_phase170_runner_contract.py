from pathlib import Path


def test_phase170_runner_is_design_only_gate():
    script = Path(
        "scripts/server/run_phase170_hidden_closure_mechanism_smoke_design_gate.sh"
    ).read_text(encoding="utf-8")

    assert "build_phase170_hidden_closure_mechanism_smoke_design_gate.py" in script
    assert "phase170_hidden_closure_mechanism_smoke_design_manifest.json" in script
    assert "hidden_closure_mechanism_phase170_gate" in script
    assert "phase171_low_budget_hidden_closure_smoke_allowed" in script
    assert "phase170_model_mechanism_allowed" in script
    assert "phase170_model_training_allowed" in script
    assert "phase171_training_allowed_now" in script
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
