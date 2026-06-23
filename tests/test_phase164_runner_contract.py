from pathlib import Path


def test_phase164_runner_is_no_training_identifiability_gate():
    script = Path(
        "scripts/server/run_phase164_synthetic_bayesian_inverse_heat_identifiability_gate.sh"
    ).read_text(encoding="utf-8")

    assert "build_phase164_synthetic_bayesian_inverse_heat_identifiability_gate.py" in script
    assert "phase164_synthetic_bayesian_inverse_heat_identifiability_manifest.json" in script
    assert "synthetic_bayesian_inverse_heat_phase164_gate" in script
    assert "phase165_adaptive_sampler_gate_allowed" in script
    assert "phase164_low_capacity_training_allowed" in script
    assert "phase164_model_mechanism_allowed" in script
    assert "phase164_model_training_allowed" in script
    assert "bayesian_pinn_training_allowed_now" in script
    assert "adaptive_sampling_training_allowed_now" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "data/raw/nist_ammt" not in script
    assert "data/raw/ambench" not in script
    assert "data/raw/external" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
