from pathlib import Path


def test_phase140_matbench_mp_is_metal_runner_is_no_training_baseline_triage():
    script = Path("scripts/server/run_phase140_matbench_mp_is_metal_baseline_gate.sh").read_text(
        encoding="utf-8"
    )
    assert "build_phase140_matbench_mp_is_metal_baseline_gate.py" in script
    assert "phase140_matbench_mp_is_metal_manifest.json" in script
    assert "matbench_mp_is_metal_gate" in script
    assert "MAX_ROWS" in script
    assert "phase140_model_mechanism_allowed" in script
    assert "phase140_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "matbench_mp_is_metal" in script
    assert "data/raw/nist_ammt" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
