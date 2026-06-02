from __future__ import annotations

from pathlib import Path


def test_phase58_clean_repro_runner_rebuilds_governance_stack():
    script = Path("scripts/server/run_phase58_clean_repro_package_a100.sh")
    text = script.read_text(encoding="utf-8")

    assert "matsci-gnn-pinn-phase58-clean" in text
    assert "summarize_phase54_process_route_claim_boundary.py" in text
    assert "summarize_phase55_spot_size_seed_check.py" in text
    assert "build_phase56_manuscript_package.py" in text
    assert "build_phase57_claim_governance.py" in text
    assert "--require-complete" in text
    assert "--require-pass" in text
    assert "copy_required_artifact" in text
    assert "_seed${seed}_macro_pinn_minmax_${run_tag}_v1/metrics.json" in text
    assert "${base_run_id}_${tag}_regions_q90.json" in text
    assert "phase58_clean_repro_manifest.json" in text
