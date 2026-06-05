from __future__ import annotations

from pathlib import Path


def test_phase111_registered_target_closure_runner_is_no_training_package():
    script = Path(
        "scripts/server/run_phase111_nist_ammt_registered_target_closure_package_a100.sh"
    ).read_text(encoding="utf-8")

    assert "build_phase111_nist_ammt_registered_target_closure_package.py" in script
    assert "phase111_nist_ammt_registered_target_closure_package_a100_manifest.json" in script
    assert "nist_ammt_sequence_branch_closed" in script
    assert "appendix_diagnostic_package_ready" in script
    assert "phase111_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
