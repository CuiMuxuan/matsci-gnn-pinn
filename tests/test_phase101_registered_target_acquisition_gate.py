from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase101_registered_target_acquisition_gate.py")
    spec = importlib.util.spec_from_file_location("phase101_gate", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase101_blocks_without_real_registered_target(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(Path(".").resolve(), tmp_path / "out")

    gate = manifest["gate"]
    assert manifest["phase"] == 101
    assert gate["status"] == "blocked_no_real_registered_target"
    assert gate["phase102_baseline_smoke_allowed"] is False
    assert gate["ready_real_registered_target_rows"] == 0
    assert gate["analytic_surrogate_closed_for_transfer"] is True
    assert gate["am_bench_transfer_unlocked"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert gate["source_phase100_status"] == "local_mechanism_package_ready_transfer_locked"

    with (tmp_path / "out/phase101_registered_target_acquisition_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        target_rows = list(csv.DictReader(handle))
    assert len(target_rows) >= 6
    assert all(row["a100_training_allowed"] == "false" for row in target_rows)
    surrogate = next(
        row for row in target_rows if row["target_id"] == "phase101_generated_registered_surrogate_closed"
    )
    assert surrogate["status"] == "closed_local_mechanism_not_transfer_target"
    assert surrogate["phase102_baseline_smoke_allowed"] == "false"
    pad = next(
        row for row in target_rows if row["target_id"] == "phase101_ambench_mds2_2716_pad_thermography_xypt"
    )
    assert pad["status"] == "blocked_registration_evidence_required"
    assert pad["physical_source_path_meaning"] == "blocked"
    external = next(
        row
        for row in target_rows
        if row["target_id"] == "phase101_external_public_registered_thermal_process_dataset"
    )
    assert external["physical_source_path_meaning"] == "unverified"

    with (tmp_path / "out/phase101_registered_target_manual_queue.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        queue_rows = list(csv.DictReader(handle))
    assert queue_rows
    assert any(row["missing_component"] == "coordinate_registration" for row in queue_rows)
    assert any(row["missing_component"] == "source_manifest_data_card" for row in queue_rows)


def test_phase101_allows_phase102_for_future_real_registered_target():
    module = _load_module()

    phase81_row = {
        "route_id": "future_registered_pad",
        "route_family": "registered_pad_thermal_source_path",
        "dataset_id": "future-public-pad",
        "target_family": "registered pad thermography",
        "source_manifest": "configs/data/future_sources.yaml",
        "split_status": "train_validation_test_split_available",
        "baseline_entry_status": "baseline_table_available",
        "coordinate_registration_status": "camera_to_galvo_registration_available",
        "status": "open_registered_target",
        "priority": "1",
        "next_action": "enter Phase 102 baseline-first smoke",
        "evidence": "future registration data card",
    }
    target_rows = module.build_target_rows(
        phase81_rows=[phase81_row],
        phase94_rows=[],
        phase100_gate={
            "status": "local_mechanism_package_ready_transfer_locked",
            "appendix_local_mechanism_ready": True,
        },
    )
    manual_queue = module.build_manual_queue(target_rows)
    gate = module.build_gate(
        phase81_gate={
            "status": "registered_target_available",
            "open_registered_target_count": 1,
            "preferred_next_route": "future_registered_pad",
        },
        phase100_gate={
            "status": "local_mechanism_package_ready_transfer_locked",
            "phase101_registered_target_acquisition_allowed": True,
        },
        target_rows=target_rows,
        manual_queue=manual_queue,
    )

    assert gate["status"] == "registered_target_acquired_phase102_allowed"
    assert gate["phase102_baseline_smoke_allowed"] is True
    assert gate["ready_real_registered_target_rows"] == 1
    assert gate["am_bench_transfer_unlocked"] is True
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    ready = [row for row in target_rows if row["phase102_baseline_smoke_allowed"]]
    assert ready[0]["source_family"] == "registered_pad_thermal_source_path"
