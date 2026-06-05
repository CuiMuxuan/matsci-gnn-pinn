from __future__ import annotations

from pathlib import Path


def test_phase115_nist_ammt_diagnostic_closure_runner_is_no_training_package():
    script = Path("scripts/server/run_phase115_nist_ammt_diagnostic_closure_package_a100.sh").read_text(
        encoding="utf-8"
    )

    assert "build_phase115_nist_ammt_diagnostic_closure_package.py" in script
    assert "phase115_nist_ammt_diagnostic_closure_package_a100_manifest.json" in script
    assert "all_training_locks_verified" in script
    assert "phase115_model_mechanism_allowed" in script
    assert "phase115_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "data/raw/nist_ammt" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
