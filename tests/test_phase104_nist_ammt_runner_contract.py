from pathlib import Path


def test_phase104_runner_rebuilds_expanded_registered_table_before_baselines():
    script = Path("scripts/server/run_phase104_nist_ammt_baseline_smoke_a100.sh").read_text(
        encoding="utf-8"
    )

    assert 'REGISTERED_ROWS_PER_TARGET_TYPE="${REGISTERED_ROWS_PER_TARGET_TYPE:-64}"' in script
    assert "phase103_nist_ammt_tiny_registered_table_builder.py" in script
    assert "--rows-per-target-type \"$REGISTERED_ROWS_PER_TARGET_TYPE\"" in script
    assert "phase104_nist_ammt_expanded_registered_table_a100_manifest.json" in script
    assert '--tiny-table "$OUTPUT_DIR/phase103_nist_ammt_tiny_registered_source_target_table.csv"' in script
    assert "phase104_nist_ammt_baseline_smoke.py" in script
