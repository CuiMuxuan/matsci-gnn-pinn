from pathlib import Path


def test_phase150_runner_is_no_training_inventory_gate():
    script = Path("scripts/server/run_phase150_dense_tensorization_inventory_gate.sh").read_text(
        encoding="utf-8"
    )
    assert "build_phase150_dense_tensorization_inventory_gate.py" in script
    assert "phase150_dense_tensorization_inventory_manifest.json" in script
    assert "dense_tensorization_phase150_gate" in script
    assert "phase151_fixed_grid_baseline_review_allowed" in script
    assert "operator_training_allowed_now" in script
    assert "phase150_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
    assert "python -m gnnpinn.train" not in script
    assert "rm -rf" not in script
