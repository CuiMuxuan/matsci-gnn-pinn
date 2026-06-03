from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase98_registered_target_unlock_gate.py")
    spec = importlib.util.spec_from_file_location("phase98_gate", script)
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


def _phase96_target() -> dict:
    return {
        "target_id": "phase96_pfhub_style_heat_source_v1",
        "target_type": "manufactured_heat_diffusion_with_moving_source",
        "alpha": 0.04,
        "source_x0": 0.28,
        "source_velocity": 0.42,
        "source_sigma2": 0.018,
        "splits": {
            "train_grid": {"nx": 19, "nt": 15},
            "validation_grid": {"nx": 23, "nt": 17},
            "test_grid": {"nx": 41, "nt": 31},
        },
    }


def _transfer_rows(*, include_open_route: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "route_id": "phase97_ambench_mds2_2716_pad_thermography_xypt",
            "source_mechanism": "fixed_green_function_features",
            "transfer_target": "ThermalData/X_pad* or Y_pad* pad thermography",
            "dataset_id": "mds2-2716",
            "route_family": "registered_pad_thermal_source_path",
            "physical_mapping": "XYPT/Xpad or XYPT/Ypad scan strategy",
            "registration_status": "pad thermography and pad XYPT exist, but current evidence has only independent-rescale diagnostics",
            "split_status": "pad_frame_or_region_split_possible",
            "baseline_status": "diagnostic_phase53_baselines_exist",
            "leakage_control": "blocked route cannot be used for feature selection or A100 training",
            "phase98_local_smoke_allowed": "false",
            "a100_training_allowed": "false",
            "status": "blocked_missing_registration",
            "priority": 1,
            "next_action": "search for documented registration",
            "evidence": "fixture",
        },
        {
            "route_id": "phase97_external_public_registered_thermal_process_dataset",
            "source_mechanism": "fixed_green_function_features",
            "transfer_target": "public registered thermal/process target",
            "dataset_id": "external_tbd",
            "route_family": "external_registered_target",
            "physical_mapping": "aligned scan path, source command, or camera-to-galvo calibration",
            "registration_status": "must_be_verified",
            "split_status": "must_define_train_val_test_split",
            "baseline_status": "baseline_table_required",
            "leakage_control": "blocked route cannot be used for feature selection or A100 training",
            "phase98_local_smoke_allowed": "false",
            "a100_training_allowed": "false",
            "status": "blocked_no_data_card",
            "priority": 2,
            "next_action": "create data card",
            "evidence": "fixture",
        },
        {
            "route_id": "phase97_pfhub_only_appendix_extension",
            "source_mechanism": "fixed_green_function_features",
            "transfer_target": "PFHub-style synthetic benchmark only",
            "dataset_id": "pfhub_style_local",
            "route_family": "synthetic_benchmark_appendix",
            "physical_mapping": "defined inside manufactured benchmark",
            "registration_status": "synthetic_registered_by_definition",
            "split_status": "fixed_train_validation_test_grids",
            "baseline_status": "phase96_local_baselines_available",
            "leakage_control": "appendix only",
            "phase98_local_smoke_allowed": "false",
            "a100_training_allowed": "false",
            "status": "synthetic_appendix_only",
            "priority": 5,
            "next_action": "keep as appendix",
            "evidence": "fixture",
        },
    ]
    if include_open_route:
        rows.append(
            {
                "route_id": "phase97_external_registered_fixture",
                "source_mechanism": "fixed_green_function_features",
                "transfer_target": "registered external thermal target",
                "dataset_id": "external_registered_fixture",
                "route_family": "external_registered_target",
                "physical_mapping": "registered source path",
                "registration_status": "coordinate_compatible_registered",
                "split_status": "train_validation_test_split_ready",
                "baseline_status": "baseline_table_ready",
                "leakage_control": "freeze split and registration before local smoke",
                "phase98_local_smoke_allowed": "true",
                "a100_training_allowed": "false",
                "status": "phase98_local_smoke_ready_no_a100",
                "priority": 1,
                "next_action": "enter local smoke",
                "evidence": "fixture",
            }
        )
    return rows


def _paths(
    tmp_path: Path,
    *,
    phase97_positive: bool = True,
    include_open_route: bool = False,
) -> dict[str, Path]:
    phase97_gate = {
        "status": "blocked_no_registered_transfer_target"
        if phase97_positive
        else "blocked_by_phase96_local_smoke",
        "positive_mechanisms": ["fixed_green_function_features"] if phase97_positive else [],
        "phase98_local_smoke_allowed": False,
        "a100_training_allowed_now": False,
    }
    return {
        "phase96_target_manifest": _write_json(tmp_path / "phase96_target.json", _phase96_target()),
        "phase97_gate": _write_json(tmp_path / "phase97_gate.json", phase97_gate),
        "phase97_transfer_routes": _write_csv(
            tmp_path / "phase97_routes.csv",
            _transfer_rows(include_open_route=include_open_route),
        ),
    }


def test_phase98_unlocks_generated_surrogate_only_for_current_state(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    gate = manifest["gate"]
    assert manifest["phase"] == 98
    assert gate["status"] == "registered_surrogate_unlocked_no_a100"
    assert gate["phase99_local_smoke_allowed"] is True
    assert gate["am_bench_transfer_unlocked"] is False
    assert gate["ambench_phase99_candidate_rows"] == 0
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert gate["preferred_phase99_candidate"] == "phase98_generated_pfhub_registered_surrogate_v1"

    card = json.loads(
        (tmp_path / manifest["outputs"]["registered_surrogate_data_card"]).read_text()
    )
    assert card["candidate_id"] == "phase98_generated_pfhub_registered_surrogate_v1"
    assert "not AM-Bench evidence" in card["not_a_claim"]
    assert card["split_plan"]["train_grid"] == {"nx": 19, "nt": 15}

    with (tmp_path / manifest["outputs"]["unlock_candidate_table"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    surrogate = next(
        row
        for row in rows
        if row["candidate_id"] == "phase98_generated_pfhub_registered_surrogate_v1"
    )
    pad = next(
        row
        for row in rows
        if row["source_route_id"] == "phase97_ambench_mds2_2716_pad_thermography_xypt"
    )
    assert surrogate["phase99_local_smoke_allowed"] == "true"
    assert surrogate["a100_training_allowed"] == "false"
    assert pad["status"] == "blocked_registration_evidence_required"
    assert pad["phase99_local_smoke_allowed"] == "false"


def test_phase98_counts_future_open_external_route_without_a100(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        tmp_path,
        tmp_path / "out",
        paths=_paths(tmp_path, include_open_route=True),
    )

    gate = manifest["gate"]
    assert gate["status"] == "registered_surrogate_unlocked_no_a100"
    assert gate["phase99_local_smoke_allowed"] is True
    assert gate["ambench_phase99_candidate_rows"] == 1
    assert gate["am_bench_transfer_unlocked"] is True
    assert gate["a100_training_allowed_now"] is False

    with (tmp_path / manifest["outputs"]["unlock_candidate_table"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    open_row = next(row for row in rows if row["source_route_id"] == "phase97_external_registered_fixture")
    assert open_row["phase99_local_smoke_allowed"] == "true"
    assert open_row["a100_training_allowed"] == "false"


def test_phase98_blocks_without_phase97_positive_signal(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        tmp_path,
        tmp_path / "out",
        paths=_paths(tmp_path, phase97_positive=False),
    )

    gate = manifest["gate"]
    assert gate["status"] == "blocked_no_phase97_positive_mechanism"
    assert gate["phase99_local_smoke_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
