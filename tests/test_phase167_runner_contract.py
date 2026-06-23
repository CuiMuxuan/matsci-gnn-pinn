from pathlib import Path


def test_phase167_runner_is_low_budget_synthetic_only():
    script = Path("scripts/server/run_phase167_low_budget_pinn_smoke.sh").read_text(
        encoding="utf-8"
    )

    assert "build_phase167_low_budget_pinn_smoke.py" in script
    assert "phase167_low_budget_pinn_smoke_manifest.json" in script
    assert "low_budget_pinn_smoke_phase167_gate" in script
    assert "phase168_focused_review_allowed" in script
    assert "phase167_model_mechanism_allowed" in script
    assert "phase167_model_claim_allowed" in script
    assert "bayesian_pinn_training_allowed_now" in script
    assert "adaptive_sampling_training_allowed_now" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "data/raw/nist_ammt" not in script
    assert "data/raw/ambench" not in script
    assert "data/raw/external" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
