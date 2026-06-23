from pathlib import Path


def test_phase171_runner_is_bounded_numpy_smoke():
    script = Path("scripts/server/run_phase171_hidden_closure_low_budget_smoke.sh").read_text(
        encoding="utf-8"
    )

    assert "build_phase171_hidden_closure_low_budget_smoke.py" in script
    assert "phase171_hidden_closure_low_budget_smoke_manifest.json" in script
    assert "hidden_closure_phase171_gate" in script
    assert "phase172_trainable_hidden_closure_design_allowed" in script
    assert "phase171_model_mechanism_allowed" in script
    assert "phase171_model_training_allowed" in script
    assert "phase172_training_allowed_now" in script
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
