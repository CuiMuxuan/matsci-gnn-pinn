from pathlib import Path


def test_phase168_runner_is_no_training_redesign_gate():
    script = Path("scripts/server/run_phase168_hidden_source_closure_redesign_gate.sh").read_text(
        encoding="utf-8"
    )

    assert "build_phase168_hidden_source_closure_redesign_gate.py" in script
    assert "phase168_hidden_source_closure_redesign_manifest.json" in script
    assert "hidden_source_closure_phase168_gate" in script
    assert "phase169_hidden_source_closure_identifiability_gate_allowed" in script
    assert "phase168_retrain_same_sampler_route_allowed" in script
    assert "phase168_model_mechanism_allowed" in script
    assert "phase168_model_training_allowed" in script
    assert "phase169_training_allowed_now" in script
    assert "bayesian_pinn_training_allowed_now" in script
    assert "adaptive_sampling_training_allowed_now" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "data/raw/nist_ammt" not in script
    assert "data/raw/ambench" not in script
    assert "data/raw/external" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
