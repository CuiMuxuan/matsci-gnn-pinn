from pathlib import Path


def test_phase166_runner_is_design_only_gate():
    script = Path("scripts/server/run_phase166_low_budget_pinn_smoke_design_gate.sh").read_text(
        encoding="utf-8"
    )

    assert "build_phase166_low_budget_pinn_smoke_design_gate.py" in script
    assert "phase166_low_budget_pinn_smoke_design_manifest.json" in script
    assert "low_budget_pinn_smoke_phase166_gate" in script
    assert "phase167_local_low_budget_pinn_smoke_allowed" in script
    assert "phase166_model_mechanism_allowed" in script
    assert "phase166_model_training_allowed" in script
    assert "phase167_training_allowed_now" in script
    assert "bayesian_pinn_training_allowed_now" in script
    assert "adaptive_sampling_training_allowed_now" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "data/raw/nist_ammt" not in script
    assert "data/raw/ambench" not in script
    assert "data/raw/external" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
