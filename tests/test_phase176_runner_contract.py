from pathlib import Path


def test_phase176_runner_is_evidence_refresh_only():
    script = Path("scripts/server/run_phase176_hidden_closure_evidence_refresh.sh").read_text(
        encoding="utf-8"
    )

    assert "build_phase176_hidden_closure_evidence_refresh.py" in script
    assert "phase176_hidden_closure_evidence_refresh_manifest.json" in script
    assert "hidden_closure_evidence_phase176_gate" in script
    assert "synthetic_hidden_closure_claim_allowed_now" in script
    assert "second_paper_core_claim_ready" in script
    assert "low_capacity_head_claim_ready" in script
    assert "phase177_materially_different_mechanism_design_allowed" in script
    assert "phase176_model_mechanism_allowed" in script
    assert "phase176_model_training_allowed" in script
    assert "phase177_training_allowed_now" in script
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
