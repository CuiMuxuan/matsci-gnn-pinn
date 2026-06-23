from pathlib import Path


def test_phase175_runner_keeps_tiny_smoke_boundaries():
    script = Path("scripts/server/run_phase175_low_capacity_hidden_closure_smoke.sh").read_text(
        encoding="utf-8"
    )

    assert "build_phase175_low_capacity_hidden_closure_smoke.py" in script
    assert "phase175_low_capacity_hidden_closure_smoke_manifest.json" in script
    assert "low_capacity_hidden_closure_phase175_gate" in script
    assert "phase176_focused_review_allowed" in script
    assert "phase175_model_mechanism_allowed" in script
    assert "phase175_model_training_allowed" in script
    assert "phase176_training_allowed_now" in script
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
