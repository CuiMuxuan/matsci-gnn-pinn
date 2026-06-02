from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase94_external_registered_target_candidate_gate.py")
    spec = importlib.util.spec_from_file_location("phase94_gate", script)
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


def _phase81_rows() -> list[dict[str, object]]:
    return [
        {
            "route_id": "ambench_mds2_2716_single_track_scan_path",
            "route_family": "registered_thermal_source_path",
            "dataset_id": "mds2-2716",
            "public_record": "https://doi.org/10.18434/mds2-2716",
            "source_manifest": "configs/data/ambench_mds2_2716_sources.yaml",
            "target_family": "ThermalData/Line_* single-track thermography",
            "source_family": "single-track scan path",
            "required_files_pinned": "3",
            "required_files_present": "3",
            "local_files_present": "true",
            "process_metadata_status": "process_metadata_available",
            "split_status": "broad12_broad21_process_splits_available",
            "coordinate_registration_status": "no single-track scan-path group",
            "registration_blocker": "missing registration",
            "baseline_entry_status": "existing_route_guard_baselines_available",
            "model_gate_status": "feature_gate_blocked",
            "paper_use": "appendix_or_future_data_requirement",
            "status": "blocked_missing_registration",
            "priority": "3",
            "next_action": "acquire aligned metadata",
            "evidence": "fixture",
        },
        {
            "route_id": "ambench_mds2_2716_pad_thermography_xypt",
            "route_family": "registered_pad_thermal_source_path",
            "dataset_id": "mds2-2716",
            "public_record": "https://doi.org/10.18434/mds2-2716",
            "source_manifest": "configs/data/ambench_mds2_2716_sources.yaml",
            "target_family": "pad thermography",
            "source_family": "XYPT/Xpad or XYPT/Ypad scan strategy",
            "required_files_pinned": "3",
            "required_files_present": "3",
            "local_files_present": "true",
            "process_metadata_status": "process_metadata_available",
            "split_status": "pad_frame_or_region_split_possible",
            "coordinate_registration_status": "pad thermography and pad XYPT exist, but current evidence has only independent-rescale diagnostics",
            "registration_blocker": "missing paper-facing registration",
            "baseline_entry_status": "diagnostic_phase53_baselines_exist",
            "model_gate_status": "diagnostic_only_until_registration",
            "paper_use": "highest_priority_data_followup",
            "status": "blocked_missing_registration",
            "priority": "1",
            "next_action": "search for documented pad camera-to-galvo registration",
            "evidence": "fixture",
        },
        {
            "route_id": "ambench_mds2_2718_exact_line_microstructure",
            "route_family": "registered_microstructure_context",
            "dataset_id": "mds2-2718",
            "public_record": "https://doi.org/10.18434/mds2-2718",
            "source_manifest": "configs/data/ambench_mds2_2718_sources.yaml",
            "target_family": "optical microscopy",
            "source_family": "exact-line TIFF panel",
            "required_files_pinned": "4",
            "required_files_present": "4",
            "local_files_present": "true",
            "process_metadata_status": "process_metadata_available_for_exact_line_images",
            "split_status": "limited_image_panel_not_broad_thermal_split",
            "coordinate_registration_status": "not_registered_to_thermal_pixels_or_source_path",
            "registration_blocker": "prior features unstable",
            "baseline_entry_status": "prior_microstructure_diagnostic_baselines_exist",
            "model_gate_status": "blocked_for_registered_physics",
            "paper_use": "appendix_diagnostic_or_separate_microstructure_branch",
            "status": "diagnostic_prior_unstable",
            "priority": "4",
            "next_action": "do not open GCN/image-encoder training",
            "evidence": "fixture",
        },
    ]


def _paths(tmp_path: Path, *, phase92_ready: bool = False) -> dict[str, Path]:
    phase81_gate = {
        "status": "blocked_no_registered_target",
        "open_registered_target_count": 0,
        "preferred_next_route": "ambench_mds2_2716_pad_thermography_xypt",
    }
    phase92_gate = {
        "status": "benchmark_review_ready" if phase92_ready else "blocked_missing_target_benchmarks",
        "benchmark_review_ready": phase92_ready,
        "usable_benchmark_inputs": 3 if phase92_ready else 0,
    }
    return {
        "phase81_gate": _write_json(tmp_path / "phase81_gate.json", phase81_gate),
        "phase81_table": _write_csv(tmp_path / "phase81_table.csv", _phase81_rows()),
        "phase92_gate": _write_json(tmp_path / "phase92_gate.json", phase92_gate),
        "phase92_manifest": _write_json(tmp_path / "phase92_manifest.json", {"gate": phase92_gate}),
    }


def test_phase94_opens_only_local_design_gate_for_pfhub_candidate(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    gate = manifest["gate"]
    assert manifest["phase"] == 94
    assert gate["status"] == "opened_local_design_gate_no_a100"
    assert gate["preferred_next_candidate"] == "P94-CAND-PFHUB-PINN"
    assert gate["phase95_local_gate_allowed"] is True
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert gate["open_local_design_candidates"] == 1

    with (tmp_path / manifest["outputs"]["candidate_triage"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["candidate_id"]: row for row in rows}
    assert by_id["P94-CAND-PFHUB-PINN"]["status"] == "open_for_local_design_gate"
    assert by_id["P94-CAND-AMBNCH-PAD-REG"]["status"] == "blocked_until_pad_registration_evidence"
    assert by_id["P94-CAND-EXACA-SIM"]["status"] == "blocked_until_simulation_data_card"
    assert all(row["a100_training_allowed"] == "false" for row in rows)

    with (tmp_path / manifest["outputs"]["local_design_queue"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        design_rows = list(csv.DictReader(handle))
    assert design_rows[0]["candidate_id"] == "P94-CAND-PFHUB-PINN"
    assert "no A100 training" in design_rows[0]["allowed_compute"]


def test_phase94_keeps_submission_dependency_separate_from_model_gate(tmp_path: Path):
    module = _load_module()

    blocked_manifest = module.build_package(tmp_path, tmp_path / "blocked", paths=_paths(tmp_path))
    ready_manifest = module.build_package(
        tmp_path, tmp_path / "ready", paths=_paths(tmp_path, phase92_ready=True)
    )

    blocked_rows_path = tmp_path / blocked_manifest["outputs"]["candidate_triage"]
    ready_rows_path = tmp_path / ready_manifest["outputs"]["candidate_triage"]
    with blocked_rows_path.open(encoding="utf-8", newline="") as handle:
        blocked = {row["candidate_id"]: row for row in csv.DictReader(handle)}
    with ready_rows_path.open(encoding="utf-8", newline="") as handle:
        ready = {row["candidate_id"]: row for row in csv.DictReader(handle)}

    assert blocked["P94-CAND-MANUSCRIPT"]["status"] == "blocked_missing_target_benchmarks"
    assert ready["P94-CAND-MANUSCRIPT"]["status"] == "review_input_available"
    assert blocked_manifest["gate"]["a100_training_allowed_now"] is False
    assert ready_manifest["gate"]["a100_training_allowed_now"] is False
