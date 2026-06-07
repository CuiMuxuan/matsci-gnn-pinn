from pathlib import Path


def test_phase157_uci_superconductivity_runner_is_no_training_mechanism_gate():
    script = Path(
        "scripts/server/run_phase157_uci_superconductivity_low_capacity_mechanism_gate.sh"
    ).read_text(encoding="utf-8")

    assert "build_phase157_uci_superconductivity_low_capacity_mechanism_gate.py" in script
    assert "phase157_uci_superconductivity_low_capacity_mechanism_manifest.json" in script
    assert "uci_superconductivity_phase157_gate" in script
    assert "phase157_model_mechanism_allowed" in script
    assert "phase157_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
    assert "python -m gnnpinn.train" not in script
    assert "data/raw/nist_ammt" not in script
    assert "data/raw/ambench" not in script
