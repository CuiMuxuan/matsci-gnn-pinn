from pathlib import Path


def test_phase163_runner_is_no_training_pinn_roadmap():
    script = Path("scripts/server/run_phase163_pinn_bayesian_hybrid_roadmap.sh").read_text(
        encoding="utf-8"
    )

    assert "build_phase163_pinn_bayesian_hybrid_roadmap.py" in script
    assert "phase163_pinn_bayesian_hybrid_roadmap_manifest.json" in script
    assert "pinn_bayesian_hybrid_roadmap_phase163_gate" in script
    assert "phase164_no_training_design_allowed" in script
    assert "phase163_model_mechanism_allowed" in script
    assert "phase163_model_training_allowed" in script
    assert "bayesian_pinn_training_allowed_now" in script
    assert "adaptive_sampling_training_allowed_now" in script
    assert "gcn_pinn_training_allowed_now" in script
    assert "cnn_pinn_training_allowed_now" in script
    assert "operator_training_allowed_now" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "data/raw/nist_ammt" not in script
    assert "data/raw/ambench" not in script
    assert "data/raw/external" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
