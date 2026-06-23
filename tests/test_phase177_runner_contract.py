from pathlib import Path


def test_phase177_runner_is_design_only_and_blocks_training():
    script = Path(
        "scripts/server/run_phase177_uncertainty_guided_latent_acquisition_design_gate.sh"
    ).read_text(encoding="utf-8")

    assert "build_phase177_uncertainty_guided_latent_acquisition_design_gate.py" in script
    assert "phase177_uncertainty_guided_latent_acquisition_design_manifest.json" in script
    assert "uncertainty_guided_latent_acquisition_phase177_gate" in script
    assert "materially_different_from_phase175" in script
    assert "phase178_no_training_acquisition_smoke_allowed" in script
    assert "phase177_model_mechanism_allowed" in script
    assert "phase177_model_training_allowed" in script
    assert "phase178_training_allowed_now" in script
    assert "bayesian_pinn_training_allowed_now" in script
    assert "adaptive_sampling_training_allowed_now" in script
    assert "gcn_pinn_training_allowed_now" in script
    assert "cnn_operator_training_allowed_now" in script
    assert "am_bench_training_allowed_now" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "data/raw/nist_ammt" not in script
    assert "data/raw/ambench" not in script
    assert "data/raw/external" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
