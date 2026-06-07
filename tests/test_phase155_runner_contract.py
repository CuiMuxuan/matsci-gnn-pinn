from pathlib import Path


def test_phase155_uci_superconductivity_runner_is_no_training_baseline_gate():
    script = Path("scripts/server/run_phase155_uci_superconductivity_baseline_gate.sh").read_text(
        encoding="utf-8"
    )

    assert "build_phase155_uci_superconductivity_baseline_gate.py" in script
    assert "phase155_uci_superconductivity_baseline_manifest.json" in script
    assert "uci_superconductivity_phase155_gate" in script
    assert "phase156_focused_review_allowed" in script
    assert "phase155_model_mechanism_allowed" in script
    assert "phase155_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "superconductivty_data.zip" in script
    assert "data/raw/nist_ammt" not in script
    assert "data/raw/ambench" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
    assert "python -m gnnpinn.train" not in script
