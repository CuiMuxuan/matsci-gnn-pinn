from __future__ import annotations

from pathlib import Path


def test_phase105_source_path_runner_is_no_training_gate():
    script = Path("scripts/server/run_phase105_nist_ammt_source_path_feature_gate_a100.sh").read_text(
        encoding="utf-8"
    )

    assert "phase105_nist_ammt_source_path_feature_gate.py" in script
    assert "phase105_nist_ammt_source_path_feature_gate_a100_manifest.json" in script
    assert "selected_feature_profile" in script
    assert "phase105_cpu_smoke_allowed" in script
    assert "a100_training_allowed_now" in script
