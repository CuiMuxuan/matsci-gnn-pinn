from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    script = Path("scripts/server/phase104_nist_ammt_baseline_smoke.py")
    spec = importlib.util.spec_from_file_location("phase104_baseline", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_field_table(path: Path) -> Path:
    rows = [
        {
            "x": "0",
            "y": "0",
            "t": "0",
            "target_intensity_mean": "10",
            "source_p_mean": "1",
            "source_p_nonzero_fraction": "0.5",
            "source_x_range": "1",
            "source_y_range": "1",
            "target_camera_code": "0",
        },
        {
            "x": "1",
            "y": "0",
            "t": "1",
            "target_intensity_mean": "12",
            "source_p_mean": "2",
            "source_p_nonzero_fraction": "0.6",
            "source_x_range": "1",
            "source_y_range": "1",
            "target_camera_code": "0",
        },
        {
            "x": "2",
            "y": "0",
            "t": "2",
            "target_intensity_mean": "14",
            "source_p_mean": "3",
            "source_p_nonzero_fraction": "0.7",
            "source_x_range": "1",
            "source_y_range": "1",
            "target_camera_code": "1",
        },
        {
            "x": "3",
            "y": "0",
            "t": "3",
            "target_intensity_mean": "16",
            "source_p_mean": "4",
            "source_p_nonzero_fraction": "0.8",
            "source_x_range": "1",
            "source_y_range": "1",
            "target_camera_code": "1",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_phase104_baseline_smoke_runs_strong_baselines_but_keeps_mechanisms_locked(
    tmp_path: Path,
):
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed in this environment")
    module = _load_module()
    field_table = _write_field_table(tmp_path / "field.csv")
    split = tmp_path / "split.json"
    split.write_text('{"splits":{"train":[0,1],"val":[2],"test":[3]}}', encoding="utf-8")
    numeric_gate = tmp_path / "numeric_gate.json"
    numeric_gate.write_text(
        json.dumps({"numeric_field_table_ready": True, "row_count": 4}) + "\n",
        encoding="utf-8",
    )

    manifest = module.build_package(
        root=Path(".").resolve(),
        field_table=field_table,
        split_manifest=split,
        numeric_gate_path=numeric_gate,
        output_dir=tmp_path / "out",
        target="target_intensity_mean",
        min_rows_for_mechanism=100,
        n_neighbors=1,
        n_estimators=5,
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase104_baseline_smoke_boundary_tiny_sample_mechanisms_locked"
    assert gate["baseline_smoke_completed"] is True
    assert gate["methods"] == ["extra_trees", "knn", "mean"]
    assert gate["phase105_model_mechanism_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    with (tmp_path / "out/phase104_nist_ammt_baseline_metric_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
