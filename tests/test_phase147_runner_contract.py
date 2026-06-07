from pathlib import Path


def test_phase147_runner_is_no_training_roadmap():
    script = Path("scripts/server/run_phase147_literature_guided_model_roadmap.sh").read_text(
        encoding="utf-8"
    )
    assert "build_phase147_literature_guided_model_roadmap.py" in script
    assert "phase147_literature_guided_model_roadmap_manifest.json" in script
    assert "literature_guided_model_roadmap_phase147_gate" in script
    assert "phase148_no_training_design_allowed" in script
    assert "phase147_model_mechanism_allowed" in script
    assert "phase147_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "data/raw/nist_ammt" not in script
    assert "data/raw/external" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
