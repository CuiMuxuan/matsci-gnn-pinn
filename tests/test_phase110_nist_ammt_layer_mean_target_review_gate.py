from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    script = Path("scripts/server/phase110_nist_ammt_layer_mean_target_review_gate.py")
    spec = importlib.util.spec_from_file_location("phase110_layer_mean_review_gate", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_phase108_table(path: Path) -> Path:
    rows = []
    split_by_layer = {1: "train", 2: "train", 3: "train", 4: "val", 5: "val", 6: "test", 7: "test"}
    for layer in range(1, 8):
        value = 10.0 + layer * 0.75
        for camera in (0, 1):
            rows.append(
                {
                    "x": str(layer),
                    "y": "0",
                    "t": str(layer),
                    "source_layer_index": str(layer),
                    "target_camera_code": str(camera),
                    "source_p_mean": "10",
                    "source_p_nonzero_fraction": "1",
                    "source_x_range": "4",
                    "source_y_range": "3",
                    "target_cp_layer_mean": str(value),
                    "split_name": split_by_layer[layer],
                    "row_id": f"{camera}::{layer:04d}",
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_split(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "splits": {
                    "train": [0, 1, 2, 3, 4, 5],
                    "val": [6, 7, 8, 9],
                    "test": [10, 11, 12, 13],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_phase108_gate(path: Path, *, ready: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "phase108_sequence_target_gap_ready_focused_review"
                if ready
                else "phase108_sequence_target_gate_closed_no_baseline_gap",
                "candidate_targets": ["target_cp_layer_mean"],
                "phase108_model_training_allowed": False,
                "a100_training_allowed_now": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_phase109_gate(path: Path, *, closed: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "phase109_sequence_target_focused_review_closed_camera_shortcut"
                if closed
                else "phase109_sequence_target_focused_review_ready_mechanism_design",
                "phase109_model_training_allowed": False,
                "a100_training_allowed_now": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_layer_mean_review_closes_layer_time_shortcut_and_training(tmp_path: Path):
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed in this environment")
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase108_field_table=_write_phase108_table(tmp_path / "phase108.csv"),
        split_manifest=_write_split(tmp_path / "split.json"),
        phase108_gate_path=_write_phase108_gate(tmp_path / "phase108_gate.json", ready=True),
        phase109_gate_path=_write_phase109_gate(tmp_path / "phase109_gate.json", closed=True),
        output_dir=tmp_path / "out",
        n_neighbors=1,
        n_estimators=5,
        min_independent_validation_gain=0.05,
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase110_layer_mean_target_review_closed_layer_time_shortcut"
    assert gate["layer_time_shortcut_detected"] is True
    assert gate["phase110_model_mechanism_allowed"] is False
    assert gate["phase110_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert gate["layer_diagnostics"]["all_two_camera_layers_duplicate_target"] is True

    with (tmp_path / "out/phase110_nist_ammt_layer_mean_target_review_profile_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        profile_rows = {row["profile"]: row for row in csv.DictReader(handle)}
    assert (
        profile_rows["layer_time_only"]["status"]
        == "layer_time_shortcut_matches_or_beats_full_validation"
    )
    assert profile_rows["source_only"]["status"] == "no_independent_source_validation_gain"


def test_layer_mean_review_blocks_when_phase109_not_closed(tmp_path: Path):
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed in this environment")
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase108_field_table=_write_phase108_table(tmp_path / "phase108.csv"),
        split_manifest=_write_split(tmp_path / "split.json"),
        phase108_gate_path=_write_phase108_gate(tmp_path / "phase108_gate.json", ready=True),
        phase109_gate_path=_write_phase109_gate(tmp_path / "phase109_gate.json", closed=False),
        output_dir=tmp_path / "out",
        n_neighbors=1,
        n_estimators=5,
        min_independent_validation_gain=0.05,
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase110_layer_mean_review_blocked_by_phase109"
    assert gate["phase110_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
