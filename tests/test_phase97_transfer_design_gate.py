from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase97_transfer_design_gate.py")
    spec = importlib.util.spec_from_file_location("phase97_gate", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _phase81_rows(*, include_open_registered: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "route_id": "ambench_mds2_2716_pad_thermography_xypt",
            "route_family": "registered_pad_thermal_source_path",
            "dataset_id": "mds2-2716",
            "public_record": "https://doi.org/10.18434/mds2-2716",
            "source_manifest": "configs/data/ambench_mds2_2716_sources.yaml",
            "target_family": "ThermalData/X_pad* or Y_pad* pad thermography",
            "source_family": "XYPT/Xpad or XYPT/Ypad scan strategy",
            "required_files_pinned": 3,
            "required_files_present": 3,
            "local_files_present": "true",
            "process_metadata_status": "process_metadata_available",
            "split_status": "pad_frame_or_region_split_possible",
            "coordinate_registration_status": "pad thermography and pad XYPT exist, but current evidence has only independent-rescale diagnostics",
            "registration_blocker": "missing paper-facing registration",
            "baseline_entry_status": "diagnostic_phase53_baselines_exist",
            "model_gate_status": "diagnostic_only_until_registration",
            "paper_use": "highest_priority_data_followup",
            "status": "blocked_missing_registration",
            "priority": 1,
            "next_action": "search for documented pad camera-to-galvo registration",
            "evidence": "fixture",
        },
        {
            "route_id": "external_public_registered_thermal_process_dataset",
            "route_family": "external_registered_target",
            "dataset_id": "external_tbd",
            "public_record": "",
            "source_manifest": "",
            "target_family": "public registered thermal/process target",
            "source_family": "aligned scan path, source command, or camera-to-galvo calibration",
            "required_files_pinned": 0,
            "required_files_present": 0,
            "local_files_present": "false",
            "process_metadata_status": "must_be_verified",
            "split_status": "must_define_train_val_test_split",
            "coordinate_registration_status": "must_be_verified",
            "registration_blocker": "no source manifest",
            "baseline_entry_status": "baseline_table_required",
            "model_gate_status": "data_card_required_before_model_gate",
            "paper_use": "future_registered_target_or_second_paper_branch",
            "status": "blocked_no_data_card",
            "priority": 2,
            "next_action": "create data card",
            "evidence": "fixture",
        },
    ]
    if include_open_registered:
        rows.append(
            {
                "route_id": "external_registered_fixture",
                "route_family": "external_registered_target",
                "dataset_id": "external_fixture",
                "public_record": "https://example.test/dataset",
                "source_manifest": "configs/data/external_fixture.yaml",
                "target_family": "registered thermal target",
                "source_family": "registered source path",
                "required_files_pinned": 2,
                "required_files_present": 2,
                "local_files_present": "true",
                "process_metadata_status": "process_metadata_available",
                "split_status": "train_validation_test_split_ready",
                "coordinate_registration_status": "coordinate_compatible_registered",
                "registration_blocker": "",
                "baseline_entry_status": "baseline_table_ready",
                "model_gate_status": "ready_for_local_feature_gate",
                "paper_use": "future_transfer_smoke",
                "status": "open_registered_target",
                "priority": 1,
                "next_action": "enter local smoke",
                "evidence": "fixture open route",
            }
        )
    return rows


def _paths(tmp_path: Path, *, include_open_registered: bool = False, phase97_allowed: bool = True) -> dict[str, Path]:
    phase96_gate = {
        "status": "local_smoke_positive_transfer_design_only" if phase97_allowed else "closed_local_smoke_negative",
        "phase97_transfer_design_allowed": phase97_allowed,
        "positive_mechanisms": ["fixed_green_function_features"] if phase97_allowed else [],
        "a100_training_allowed_now": False,
    }
    mechanism_rows = [
        {
            "mechanism_id": "P96-MECH-001",
            "mechanism": "fixed_green_function_features",
            "comparator": "vanilla_deterministic_surrogate",
            "selected_by_validation": "true",
            "validation_gate_pass": "true",
            "test_audit_pass": "true",
            "transfer_design_signal": "true" if phase97_allowed else "false",
            "next_action": "open Phase 97 transfer design gate",
            "reason": "fixture",
        }
    ]
    phase81_gate = {
        "status": "blocked_no_registered_target",
        "open_registered_target_count": 1 if include_open_registered else 0,
        "preferred_next_route": "external_registered_fixture" if include_open_registered else "ambench_mds2_2716_pad_thermography_xypt",
    }
    return {
        "phase96_gate": _write_json(tmp_path / "phase96_gate.json", phase96_gate),
        "phase96_mechanism_table": _write_csv(tmp_path / "phase96_mechanisms.csv", mechanism_rows),
        "phase81_gate": _write_json(tmp_path / "phase81_gate.json", phase81_gate),
        "phase81_intake_table": _write_csv(
            tmp_path / "phase81_rows.csv",
            _phase81_rows(include_open_registered=include_open_registered),
        ),
        "phase81_data_card": _write_json(
            tmp_path / "phase81_data_card.json",
            {"gate_status": phase81_gate["status"]},
        ),
        "phase92_gate": _write_json(
            tmp_path / "phase92_gate.json",
            {"status": "blocked_missing_target_benchmarks", "submission_ready": False},
        ),
    }


def test_phase97_blocks_current_routes_without_registered_transfer_target(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    gate = manifest["gate"]
    assert manifest["phase"] == 97
    assert gate["status"] == "blocked_no_registered_transfer_target"
    assert gate["phase98_local_smoke_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert gate["positive_mechanisms"] == ["fixed_green_function_features"]
    assert gate["preferred_phase98_route"] == "none"

    with (tmp_path / manifest["outputs"]["transfer_route_table"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["route_id"] == "phase97_current_ambench_spot_size_process_kernel" for row in rows)
    assert any(row["status"] == "synthetic_appendix_only" for row in rows)
    assert all(row["a100_training_allowed"] == "false" for row in rows)
    assert all(row["phase98_local_smoke_allowed"] == "false" for row in rows)


def test_phase97_allows_only_phase98_when_registered_target_exists(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        tmp_path,
        tmp_path / "out",
        paths=_paths(tmp_path, include_open_registered=True),
    )

    gate = manifest["gate"]
    assert gate["status"] == "transfer_design_ready_no_a100"
    assert gate["phase98_local_smoke_allowed"] is True
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert gate["preferred_phase98_route"] == "phase97_external_registered_fixture"

    with (tmp_path / manifest["outputs"]["transfer_route_table"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    open_row = next(row for row in rows if row["route_id"] == "phase97_external_registered_fixture")
    assert open_row["phase98_local_smoke_allowed"] == "true"
    assert open_row["a100_training_allowed"] == "false"


def test_phase97_blocks_if_phase96_has_no_positive_mechanism(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        tmp_path,
        tmp_path / "out",
        paths=_paths(tmp_path, phase97_allowed=False),
    )

    gate = manifest["gate"]
    assert gate["status"] == "blocked_by_phase96_local_smoke"
    assert gate["phase98_local_smoke_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
