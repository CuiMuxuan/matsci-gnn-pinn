from pathlib import Path


def test_phase153_runner_is_no_training_contribution_refinement():
    script = Path("scripts/server/run_phase153_first_paper_contribution_refinement.sh").read_text(
        encoding="utf-8"
    )

    assert "build_phase153_first_paper_contribution_refinement.py" in script
    assert "phase153_first_paper_contribution_refinement_manifest.json" in script
    assert "first_paper_contribution_refinement_phase153_gate" in script
    assert "new_model_claim_ready" in script
    assert "phase153_model_mechanism_allowed" in script
    assert "phase153_model_training_allowed" in script
    assert "operator_training_allowed_now" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "data/raw/nist_ammt" not in script
    assert "data/raw/ambench" not in script
    assert "data/raw/external" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
