from pathlib import Path


def test_phase152_route_closure_runner_is_no_training_refresh():
    script = Path(
        "scripts/server/run_phase152_paper_evidence_neural_operator_route_closure.sh"
    ).read_text(encoding="utf-8")

    assert "build_phase152_paper_evidence_neural_operator_route_closure.py" in script
    assert "phase152_paper_evidence_neural_operator_route_closure_manifest.json" in script
    assert "paper_evidence_refresh_phase152_gate" in script
    assert "new_neural_operator_model_claim_ready" in script
    assert "phase152_model_mechanism_allowed" in script
    assert "phase152_model_training_allowed" in script
    assert "operator_training_allowed_now" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "data/raw/nist_ammt" not in script
    assert "data/raw/ambench" not in script
    assert "data/raw/external" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
