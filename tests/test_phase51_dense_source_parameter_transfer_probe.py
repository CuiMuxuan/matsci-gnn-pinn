from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_probe_module():
    module_path = Path("scripts/server/phase51_dense_source_parameter_transfer_probe.py")
    module_spec = importlib.util.spec_from_file_location("phase51_probe", module_path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def test_phase51_synthetic_dense_to_sparse_probe_reports_transfer(tmp_path: Path):
    probe = _load_probe_module()
    output = tmp_path / "synthetic_dense_transfer.json"

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
            "--sparse-fit-size",
            "48",
            "--dense-fit-size",
            "96",
            "--repeats",
            "1",
            "--json-output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["mode"] == "dense_to_sparse_moving_source_transfer"
    assert set(payload["summary"]) >= {
        "sparse_search",
        "dense_params_sparse_theta",
        "dense_upper_bound",
        "gains_vs_sparse_search",
    }
    assert "parameter_recovery_pass_rate" in payload["summary"]["dense_params_sparse_theta"]
    assert payload["decision"]["status"] in {"positive", "negative"}


def test_phase51_table_dense_to_sparse_probe_reports_three_paths(tmp_path: Path):
    probe = _load_probe_module()
    table = tmp_path / "thermal.csv"
    rows = [
        "x,y,t,temperature_C,frame_index,row_index,col_index",
    ]
    for index in range(60):
        frame = index // 20
        row = (index // 5) % 4
        col = index % 5
        x = float(col)
        y = float(row)
        t = float(frame)
        temp = 1000.0 + 30.0 * x - 7.0 * y + 12.0 * t
        rows.append(f"{x},{y},{t},{temp},{frame},{row},{col}")
    table.write_text("\n".join(rows) + "\n", encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "splits": {
                    "train": list(range(0, 40)),
                    "val": list(range(40, 50)),
                    "test": list(range(50, 60)),
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "table_dense_transfer.json"

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
            "--sparse-fit-size",
            "20",
            "--dense-fit-size",
            "36",
            "--repeats",
            "1",
            "--json-output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["split_sizes"] == {"train": 40, "val": 10, "test": 10}
    run = payload["runs"][0]
    assert set(run["methods"]) == {
        "sparse_search",
        "dense_params_sparse_theta",
        "dense_upper_bound",
    }
    assert "test_rmse_gain" in payload["summary"]["gains_vs_sparse_search"]["dense_params_sparse_theta"]
    assert "start_x" in payload["summary"]["parameter_delta_dense_vs_sparse_mean"]
