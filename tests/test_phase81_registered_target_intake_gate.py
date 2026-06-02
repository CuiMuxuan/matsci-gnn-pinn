from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import yaml


def _load_module():
    script = Path("scripts/server/build_phase81_registered_target_intake_gate.py")
    spec = importlib.util.spec_from_file_location("phase81_gate", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_yaml(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _manifest(dataset_id: str, local_root: str, required_paths: list[str]) -> dict:
    return {
        "dataset_id": dataset_id,
        "dataset_name": f"{dataset_id} test",
        "local_root": local_root,
        "record": {
            "doi": f"https://doi.org/10.18434/{dataset_id}",
            "pdr_landing_page": f"https://data.nist.gov/od/id/{dataset_id}",
        },
        "required_files": [
            {
                "id": f"file_{idx}",
                "relative_path": rel,
                "size_bytes": 1,
                "sha256": "0" * 64,
                "download_url": f"https://example.test/{dataset_id}/{idx}",
                "purpose": "test fixture",
            }
            for idx, rel in enumerate(required_paths)
        ],
    }


def _phase71_rows(*, open_target: bool = False) -> list[dict[str, object]]:
    if open_target:
        return [
            {
                "evidence_source": "phase52_registered_source_path_gate",
                "target_family": "single_track_thermography",
                "source_path_family": "single_track_scan_path",
                "coordinate_status": "coordinate.compatible=true",
                "unit_status": "registered_physical_units",
                "coverage_status": "aligned_target_available",
                "registration_status": "registered_compatible",
                "feature_route_status": "ready_for_fixed_feature_gate",
                "paper_use": "candidate_c_reopen_evidence",
                "status": "aligned_single_track_source_path",
                "blocker": "",
                "evidence": "test aligned route",
            },
            {
                "evidence_source": "phase53_pad_inventory",
                "target_family": "pad_thermography",
                "source_path_family": "pad_xypt",
                "coordinate_status": "documented_pad_registration",
                "unit_status": "registered_physical_units",
                "coverage_status": "pad_target_and_source_overlap",
                "registration_status": "registered_compatible",
                "feature_route_status": "ready_for_fixed_feature_gate",
                "paper_use": "candidate_c_reopen_evidence",
                "status": "aligned_pad_target_available",
                "blocker": "",
                "evidence": "test aligned pad route",
            },
        ]
    return [
        {
            "evidence_source": "phase52_registered_source_path_gate",
            "target_family": "single_track_thermography",
            "source_path_family": "pad_xypt_only",
            "coordinate_status": "camera_pixels_vs_galvo_mm",
            "unit_status": "unit_mismatch_without_registration",
            "coverage_status": "single_track_target_not_covered_by_pad_xypt",
            "registration_status": "not_registered",
            "feature_route_status": "do_not_build_source_path_features",
            "paper_use": "appendix_data_blocker",
            "status": "blocked_single_track_source_path",
            "blocker": "pad XYPT cannot be mapped",
            "evidence": "test blocked route",
        },
        {
            "evidence_source": "phase53_pad_inventory",
            "target_family": "pad_thermography",
            "source_path_family": "pad_xypt",
            "coordinate_status": "independent_rescale_only",
            "unit_status": "galvo_mm_vs_camera_pixels",
            "coverage_status": "pad_tables_exist_but_unregistered",
            "registration_status": "not_paper_registered",
            "feature_route_status": "diagnostic_only",
            "paper_use": "appendix_data_blocker",
            "status": "blocked_pad_registration",
            "blocker": "no HDF5 registration metadata",
            "evidence": "test blocked pad route",
        },
    ]


def _paths(tmp_path: Path, *, open_target: bool = False) -> dict[str, Path]:
    raw_root = tmp_path / "raw"
    (raw_root / "mds2-2716" / "readme.txt").parent.mkdir(parents=True)
    (raw_root / "mds2-2716" / "readme.txt").write_text("x", encoding="utf-8")
    (raw_root / "mds2-2718" / "readme.txt").parent.mkdir(parents=True)
    (raw_root / "mds2-2718" / "readme.txt").write_text("x", encoding="utf-8")
    return {
        "mds2_2716_manifest": _write_yaml(
            tmp_path / "mds2_2716.yaml",
            _manifest("mds2-2716", "raw/mds2-2716", ["readme.txt", "missing.h5"]),
        ),
        "mds2_2718_manifest": _write_yaml(
            tmp_path / "mds2_2718.yaml",
            _manifest("mds2-2718", "raw/mds2-2718", ["readme.txt"]),
        ),
        "phase71_manifest": _write_json(
            tmp_path / "phase71_manifest.json",
            {
                "candidate_c_gate": {
                    "status": "opened_for_aligned_feature_gate"
                    if open_target
                    else "blocked_by_registration_data",
                    "open_aligned_feature_gate": open_target,
                }
            },
        ),
        "phase71_table": _write_csv(tmp_path / "phase71_table.csv", _phase71_rows(open_target=open_target)),
        "phase80_manifest": _write_json(
            tmp_path / "phase80_manifest.json",
            {"gate": {"status": "blocked_by_local_surrogate_gate"}},
        ),
    }


def test_phase81_blocks_training_without_registered_target(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    gate = manifest["gate"]
    assert manifest["phase"] == 81
    assert gate["status"] == "blocked_no_registered_target"
    assert gate["phase82_baseline_smoke_allowed"] is False
    assert gate["phase83_registered_feature_gate_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert gate["preferred_next_route"] == "ambench_mds2_2716_pad_thermography_xypt"

    table_path = tmp_path / manifest["outputs"]["intake_table"]
    with table_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert any(row["route_id"] == "ambench_mds2_2716_pad_thermography_xypt" for row in rows)
    assert any(row["status"] == "diagnostic_prior_unstable" for row in rows)
    assert any(row["status"] == "blocked_no_data_card" for row in rows)


def test_phase81_opens_only_baseline_smoke_for_registered_target(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        tmp_path,
        tmp_path / "out",
        paths=_paths(tmp_path, open_target=True),
    )

    gate = manifest["gate"]
    assert gate["status"] == "opened_for_phase82_baseline_smoke"
    assert gate["phase82_baseline_smoke_allowed"] is True
    assert gate["phase83_registered_feature_gate_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["open_registered_target_count"] == 2
    assert gate["preferred_next_route"] == "ambench_mds2_2716_pad_thermography_xypt"

    card_path = tmp_path / manifest["outputs"]["data_card"]
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["gate_status"] == "opened_for_phase82_baseline_smoke"
    assert "No architecture training opens" in card["selection_rule"]
