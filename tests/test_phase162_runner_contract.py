from pathlib import Path


def test_phase162_uci_steel_energy_runner_is_no_training_baseline_gate():
    script = Path("scripts/server/run_phase162_uci_steel_industry_energy_baseline_gate.sh").read_text(
        encoding="utf-8"
    )

    assert "build_phase162_uci_steel_industry_energy_baseline_gate.py" in script
    assert "phase162_uci_steel_industry_energy_baseline_manifest.json" in script
    assert "uci_steel_energy_phase162_gate" in script
    assert "phase163_focused_review_allowed" in script
    assert "phase162_model_mechanism_allowed" in script
    assert "phase162_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
    assert "python -m gnnpinn.train" not in script
    assert "data/raw/nist_ammt" not in script
    assert "data/raw/ambench" not in script
