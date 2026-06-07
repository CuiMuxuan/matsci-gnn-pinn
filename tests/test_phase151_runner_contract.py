from pathlib import Path


def test_phase151_runner_is_no_training_dense_baseline_review():
    script = Path("scripts/server/run_phase151_fixed_grid_dense_baseline_review.sh").read_text(
        encoding="utf-8"
    )
    assert "build_phase151_fixed_grid_dense_baseline_review.py" in script
    assert "phase151_fixed_grid_dense_baseline_manifest.json" in script
    assert "fixed_grid_dense_phase151_gate" in script
    assert "phase152_low_capacity_dense_design_candidates" in script
    assert "phase151_model_training_allowed" in script
    assert "operator_training_allowed_now" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
    assert "python -m gnnpinn.train" not in script
    assert "data/raw/nist_ammt" not in script
