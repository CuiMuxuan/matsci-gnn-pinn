from pathlib import Path


def test_phase148_runner_is_no_training_path_contact_audit():
    script = Path("scripts/server/run_phase148_nist_ammt_path_contact_graph_audit_a800.sh").read_text(
        encoding="utf-8"
    )
    assert "phase148_nist_ammt_path_contact_graph_audit.py" in script
    assert "phase148_nist_ammt_path_contact_graph_audit_manifest.json" in script
    assert "path_contact_graph_audit_phase148_gate" in script
    assert "--data-root" in script
    assert "data/raw/nist_ammt/mds2_2044" in script
    assert "phase148_model_mechanism_allowed" in script
    assert "phase148_model_training_allowed" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
