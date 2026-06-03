from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase102_registered_source_manifest_gate.py")
    spec = importlib.util.spec_from_file_location("phase102_gate", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase102_opens_minimal_nist_intake_without_training(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(Path(".").resolve(), tmp_path / "out")

    gate = manifest["gate"]
    assert manifest["phase"] == 102
    assert gate["status"] == "source_manifest_ready_phase103_intake_allowed"
    assert gate["preferred_candidate"] == "P102-CAND-NIST-AMMT-3D-SCAN"
    assert gate["phase103_intake_allowed"] is True
    assert gate["phase103_large_server_download_allowed"] is True
    assert gate["phase104_baseline_smoke_allowed"] is False
    assert gate["phase105_model_mechanism_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False

    with (tmp_path / "out/phase102_registered_source_candidate_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        source_rows = list(csv.DictReader(handle))
    nist = next(row for row in source_rows if row["candidate_id"] == "P102-CAND-NIST-AMMT-3D-SCAN")
    assert nist["source_manifest_status"] == "ready_official_file_manifest"
    assert nist["phase103_intake_allowed"] == "true"
    assert nist["phase104_baseline_smoke_allowed"] == "false"
    assert nist["a100_training_allowed"] == "false"
    assert nist["status"] == "source_manifest_ready_phase103_intake_allowed"
    assert "10.18434/M32044" in nist["doi"]

    with (tmp_path / "out/phase102_nist_ammt_file_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        file_rows = list(csv.DictReader(handle))
    metadata = next(row for row in file_rows if row["file_name"] == "Metadata.zip")
    assert metadata["required_for_phase103"] == "true"
    assert metadata["download_order"] == "1"
    assert metadata["expected_bytes"] == "2489233"
    build_commands = next(row for row in file_rows if row["file_name"] == "Build Command Data.zip")
    assert build_commands["required_for_phase103"] == "false"
    assert build_commands["download_scope"] == "long_running_server_download_after_metadata_pass"

    with (tmp_path / "out/phase102_phase103_intake_queue.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        queue_rows = list(csv.DictReader(handle))
    assert queue_rows[0]["task"].startswith("download Metadata.zip")
    assert "A100 server" in queue_rows[1]["allowed_location"]
    assert "Build Command Data" in queue_rows[1]["expected_input"]

    card = manifest["server_download_policy"]
    assert card["large_server_download_allowed"] is True
    assert card["local_large_download_allowed"] is False
    assert card["first_file"] == "Metadata.zip"
    assert card["known_total_gib"] > 16.0


def test_phase102_blocks_intake_when_large_server_download_not_allowed(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        Path(".").resolve(),
        tmp_path / "out",
        large_server_download_allowed=False,
    )

    gate = manifest["gate"]
    assert gate["status"] == "blocked_no_source_manifest_ready"
    assert gate["phase103_intake_allowed"] is False
    assert gate["phase103_large_server_download_allowed"] is False
    assert gate["phase104_baseline_smoke_allowed"] is False
    assert gate["a100_training_allowed_now"] is False

    with (tmp_path / "out/phase102_registered_source_candidate_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        source_rows = list(csv.DictReader(handle))
    nist = next(row for row in source_rows if row["candidate_id"] == "P102-CAND-NIST-AMMT-3D-SCAN")
    assert nist["status"] == "blocked_until_user_allows_server_download"
    assert nist["phase103_intake_allowed"] == "false"


def test_phase102_registration_checks_keep_baseline_smoke_locked():
    module = _load_module()

    registration_rows = module.build_registration_rows()
    gate = module.build_gate(
        phase101_gate={"status": "blocked_no_real_registered_target"},
        source_rows=[
            {
                "candidate_id": "P102-CAND-NIST-AMMT-3D-SCAN",
                "phase103_intake_allowed": True,
            }
        ],
        registration_rows=registration_rows,
        queue_rows=[{"queue_id": "P102-INTAKE-001"}],
        large_server_download_allowed=True,
    )

    assert gate["phase103_intake_allowed"] is True
    assert gate["phase104_baseline_smoke_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["critical_registration_checks_pending"] == 2
    assert "verify explicit coordinate transforms" in " ".join(gate["required_before_baseline_smoke"])
