from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_probe_module():
    module_path = Path("scripts/server/phase52_registered_source_path_probe.py")
    module_spec = importlib.util.spec_from_file_location("phase52_probe", module_path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def _write_xypt_file(path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        for group_name in ["Xpad", "Ypad"]:
            group = handle.create_group(f"XYPT/{group_name}")
            group.create_dataset("X", data=[[0.0, 0.5, 1.0, 1.5, 2.0]])
            group.create_dataset("Y", data=[[0.0, 0.0, 0.0, 0.0, 0.0]])
            group.create_dataset("P", data=[[0.0, 285.0, 285.0, 285.0, 0.0]])
            group.create_dataset("T", data=[[0, 0, 4, 0, 0]])


def _write_table(path: Path, *, dataset_path: str, line_id: str) -> None:
    rows = [
        "x,y,z,t,temperature_C,frame_index,row_index,col_index,dataset_path,line_id",
    ]
    for index in range(48):
        frame = index // 16
        row = (index // 4) % 4
        col = index % 4
        x_pixel = col * 100
        y_pixel = row * 200
        temp = 1000.0 + 20.0 * col - 3.0 * row + 10.0 * frame
        rows.append(f"{x_pixel},{y_pixel},0,{frame},{temp},{frame},{y_pixel},{x_pixel},{dataset_path},{line_id}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_split(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "splits": {
                    "train": list(range(0, 32)),
                    "val": list(range(32, 40)),
                    "test": list(range(40, 48)),
                }
            }
        ),
        encoding="utf-8",
    )


def test_phase52_rejects_line_table_when_xypt_is_pad_only(tmp_path: Path):
    probe = _load_probe_module()
    xypt = tmp_path / "scan.h5"
    table = tmp_path / "line.csv"
    split = tmp_path / "split.json"
    output = tmp_path / "summary.json"
    _write_xypt_file(xypt)
    _write_table(table, dataset_path="ThermalData/Line_0_1/Signal", line_id="Line_0_1")
    _write_split(split)

    status = probe.main(
        [
            "--scan-strategy",
            str(xypt),
            "--table",
            str(table),
            "--target",
            "temperature_C",
            "--split-manifest",
            str(split),
            "--json-output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["decision"]["status"] == "negative"
    assert payload["compatibility"]["selected_xypt_path"] is None
    assert "single-track" in payload["decision"]["reason"]


def test_phase52_registered_path_diagnostic_runs_for_xpad_table(tmp_path: Path):
    probe = _load_probe_module()
    xypt = tmp_path / "scan.h5"
    table = tmp_path / "xpad.csv"
    split = tmp_path / "split.json"
    output = tmp_path / "summary.json"
    _write_xypt_file(xypt)
    _write_table(table, dataset_path="ThermalData/X_pad1/Signal", line_id="X_pad1")
    _write_split(split)

    status = probe.main(
        [
            "--scan-strategy",
            str(xypt),
            "--table",
            str(table),
            "--target",
            "temperature_C",
            "--split-manifest",
            str(split),
            "--allow-independent-rescale",
            "--json-output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["compatibility"]["selected_xypt_path"] == "XYPT/Xpad"
    assert payload["compatibility"]["coordinate"]["compatible"] is False
    assert payload["compatibility"]["coordinate"]["span_ratio_compatible"] is False
    assert "registered_source_path" in payload["summary"]
    assert "registered_source_inverse_distance" in payload["feature_names"]
    assert payload["decision"]["status"] in {"positive", "negative"}
