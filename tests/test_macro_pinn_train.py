from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _torch_available() -> bool:
    result = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not _torch_available(),
    reason="torch is not importable in the current environment",
)


def test_macro_pinn_training_cli_writes_artifacts(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "toy_temperature.csv"
    table.write_text(
        "x,y,t,T\n"
        "0,0,0,0\n"
        "1,0,0,1\n"
        "0,1,0,1\n"
        "1,1,0,2\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"

    status = main(
        [
            "--table",
            str(table),
            "--target",
            "T",
            "--output-dir",
            str(output_dir),
            "--steps",
            "5",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--log-every",
            "1",
        ]
    )

    metrics_path = output_dir / "metrics.json"
    assert status == 0
    assert metrics_path.exists()
    assert (output_dir / "checkpoint.pt").exists()
    assert (output_dir / "artifact_manifest.json").exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["target"] == "T"
    assert payload["n_points"] == 4
    assert "rmse" in payload["metrics"]
    assert payload["target_normalization"]["enabled"] is True


def test_macro_pinn_training_cli_with_split_manifest(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "toy_temperature.csv"
    table.write_text(
        "x,y,t,T\n"
        "0,0,0,0\n"
        "1,0,0,1\n"
        "0,1,0,1\n"
        "1,1,0,2\n",
        encoding="utf-8",
    )
    split = tmp_path / "split.json"
    split.write_text(
        '{"splits":{"train":[0,1],"val":[2],"test":[3]}}',
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"

    status = main(
        [
            "--table",
            str(table),
            "--target",
            "T",
            "--output-dir",
            str(output_dir),
            "--steps",
            "3",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--split-manifest",
            str(split),
            "--input-normalization",
            "standard",
        ]
    )

    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert status == 0
    assert payload["train_points"] == 2
    assert payload["split_metrics"]["train"]["n_points"] == 2
    assert payload["split_metrics"]["val"]["n_points"] == 1
    assert payload["input_normalization"]["mode"] == "standard"
    assert payload["input_normalization"]["coordinates"]["applied"] is True
    assert payload["input_normalization"]["time"]["applied"] is True
