from pathlib import Path


def test_phase178_runner_is_no_training_utility_smoke():
    script = Path(
        "scripts/server/run_phase178_uncertainty_guided_acquisition_utility_smoke.sh"
    ).read_text(encoding="utf-8")

    assert "build_phase178_uncertainty_guided_acquisition_utility_smoke.py" in script
    assert "phase178_uncertainty_guided_acquisition_utility_manifest.json" in script
    assert "uncertainty_guided_acquisition_phase178_gate" in script
    assert "phase179_training_design_allowed" in script
    assert "phase178_model_mechanism_allowed" in script
    assert "phase178_model_training_allowed" in script
    assert "phase179_training_allowed_now" in script
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
