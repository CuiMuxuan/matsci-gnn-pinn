from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_probe_module():
    module_path = Path("scripts/server/phase50_moving_source_inversion_probe.py")
    module_spec = importlib.util.spec_from_file_location("phase50_probe", module_path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def test_phase50_synthetic_moving_source_inversion_reports_parameters(tmp_path: Path):
    probe = _load_probe_module()
    output = tmp_path / "moving_source_summary.json"

    status = probe.main(
        [
            "--mode",
            "synthetic",
            "--grid-mode",
            "tiny",
            "--synthetic-grid",
            "8",
            "--synthetic-frames",
            "4",
            "--synthetic-noise-std",
            "4.0",
            "--initial-size",
            "24",
            "--acquisition-size",
            "32",
            "--repeats",
            "1",
            "--json-output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["mode"] == "moving_source_inversion"
    assert "test_rmse_mean" in payload["summary"]
    assert "parameter_recovery_pass_rate" in payload["summary"]
    assert payload["runs"][0]["params"]["core_width"] > 0.0
    assert payload["runs"][0]["calibration"]["scale"] >= 1.0


def test_phase50_table_moving_source_inversion_reports_split_metrics(tmp_path: Path):
    probe = _load_probe_module()
    table = tmp_path / "thermal.csv"
    rows = [
        "x,y,t,temperature_C,frame_index,row_index,col_index",
    ]
    for index in range(36):
        frame = index // 12
        row = (index // 6) % 2
        col = index % 6
        x = float(col)
        y = float(row)
        t = float(frame)
        temp = 1000.0 + 25.0 * x - 5.0 * y + 10.0 * t
        rows.append(f"{x},{y},{t},{temp},{frame},{row},{col}")
    table.write_text("\n".join(rows) + "\n", encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "splits": {
                    "train": list(range(0, 22)),
                    "val": list(range(22, 29)),
                    "test": list(range(29, 36)),
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "table_summary.json"

    status = probe.main(
        [
            "--mode",
            "table",
            "--grid-mode",
            "tiny",
            "--table",
            str(table),
            "--target",
            "temperature_C",
            "--split-manifest",
            str(split),
            "--initial-size",
            "8",
            "--acquisition-size",
            "8",
            "--repeats",
            "1",
            "--json-output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["split_sizes"] == {"train": 22, "val": 7, "test": 7}
    assert payload["runs"][0]["splits"]["test"]["n_points"] == 7
    assert payload["decision"]["status"] in {"positive", "negative"}
