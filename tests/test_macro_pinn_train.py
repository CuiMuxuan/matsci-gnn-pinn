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
            "--hot-quantile",
            "0.5",
            "--gradient-quantile",
            "0.5",
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
    assert "hot_q50" in payload["split_metrics"]["train"]["region_metrics"]
    assert "gradient_q50" in payload["split_metrics"]["train"]["region_metrics"]


def test_macro_pinn_training_cli_records_process_conditioning_columns(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "toy_process_temperature.csv"
    table.write_text(
        "x,y,t,T,line_id,laser_power_W,scan_speed_mm_s,spot_size_um\n"
        "0,0,0,10,Line_0_1,285,960,67\n"
        "1,0,0,11,Line_0_1,285,960,67\n"
        "0,1,1,20,Line_3_1,325,960,67\n"
        "1,1,1,21,Line_3_1,325,960,67\n",
        encoding="utf-8",
    )
    split = tmp_path / "split.json"
    split.write_text(
        '{"splits":{"train":[0,1],"val":[2],"test":[3]}}',
        encoding="utf-8",
    )
    output_dir = tmp_path / "conditioned_run"

    status = main(
        [
            "--table",
            str(table),
            "--target",
            "T",
            "--output-dir",
            str(output_dir),
            "--steps",
            "2",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--split-manifest",
            str(split),
            "--input-normalization",
            "standard",
            "--input-feature-column",
            "laser_power_W",
            "--input-feature-column",
            "scan_speed_mm_s",
            "--input-feature-column",
            "spot_size_um",
            "--log-every",
            "1",
        ]
    )

    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = __import__("torch").load(output_dir / "checkpoint.pt", map_location="cpu")

    assert status == 0
    assert payload["input_features"]["enabled"] is True
    assert payload["input_features"]["columns"] == [
        "laser_power_W",
        "scan_speed_mm_s",
        "spot_size_um",
    ]
    assert payload["input_features"]["count"] == 3
    assert payload["input_features"]["normalization"]["mode"] == "standard"
    assert payload["input_features"]["normalization"]["applied"] is True
    assert payload["input_features"]["conditioning_mode"] == "concat"
    assert checkpoint["metadata"]["param_dim"] == 3
    assert checkpoint["metadata"]["input_features"]["conditioning_mode"] == "concat"
    assert checkpoint["metadata"]["input_features"]["columns"] == [
        "laser_power_W",
        "scan_speed_mm_s",
        "spot_size_um",
    ]


def test_macro_pinn_training_cli_supports_film_process_conditioning(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "toy_process_temperature.csv"
    table.write_text(
        "x,y,t,T,line_id,laser_power_W,scan_speed_mm_s,spot_size_um\n"
        "0,0,0,10,Line_0_1,285,960,67\n"
        "1,0,0,11,Line_0_1,285,960,67\n"
        "0,1,1,20,Line_3_1,325,960,67\n"
        "1,1,1,21,Line_3_1,325,960,67\n",
        encoding="utf-8",
    )
    split = tmp_path / "split.json"
    split.write_text(
        '{"splits":{"train":[0,1],"val":[2],"test":[3]}}',
        encoding="utf-8",
    )
    output_dir = tmp_path / "film_conditioned_run"

    status = main(
        [
            "--table",
            str(table),
            "--target",
            "T",
            "--output-dir",
            str(output_dir),
            "--steps",
            "2",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--split-manifest",
            str(split),
            "--input-normalization",
            "standard",
            "--input-conditioning-mode",
            "film",
            "--input-feature-normalization",
            "global_standard",
            "--input-feature-column",
            "laser_power_W",
            "--input-feature-column",
            "scan_speed_mm_s",
            "--input-feature-column",
            "spot_size_um",
            "--log-every",
            "1",
        ]
    )

    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = __import__("torch").load(output_dir / "checkpoint.pt", map_location="cpu")

    assert status == 0
    assert payload["config"]["input_conditioning_mode"] == "film"
    assert payload["config"]["input_feature_normalization"] == "global_standard"
    assert payload["input_features"]["enabled"] is True
    assert payload["input_features"]["conditioning_mode"] == "film"
    assert payload["input_features"]["normalization"]["mode"] == "global_standard"
    assert payload["input_features"]["normalization"]["fit_scope"] == "global"
    assert checkpoint["metadata"]["param_dim"] == 3
    assert checkpoint["metadata"]["input_features"]["conditioning_mode"] == "film"


def test_macro_pinn_training_cli_supports_concat_film_process_conditioning(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "toy_process_temperature.csv"
    table.write_text(
        "x,y,t,T,line_id,laser_power_W,scan_speed_mm_s,spot_size_um\n"
        "0,0,0,10,Line_0_1,285,960,67\n"
        "1,0,0,11,Line_0_1,285,960,67\n"
        "0,1,1,20,Line_3_1,325,960,67\n"
        "1,1,1,21,Line_3_1,325,960,67\n",
        encoding="utf-8",
    )
    split = tmp_path / "split.json"
    split.write_text(
        '{"splits":{"train":[0,1],"val":[2],"test":[3]}}',
        encoding="utf-8",
    )
    output_dir = tmp_path / "concat_film_conditioned_run"

    status = main(
        [
            "--table",
            str(table),
            "--target",
            "T",
            "--output-dir",
            str(output_dir),
            "--steps",
            "2",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--split-manifest",
            str(split),
            "--input-normalization",
            "standard",
            "--input-conditioning-mode",
            "concat_film",
            "--input-film-strength",
            "0.25",
            "--input-feature-normalization",
            "global_standard",
            "--input-feature-column",
            "laser_power_W",
            "--input-feature-column",
            "scan_speed_mm_s",
            "--input-feature-column",
            "spot_size_um",
            "--log-every",
            "1",
        ]
    )

    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = __import__("torch").load(output_dir / "checkpoint.pt", map_location="cpu")

    assert status == 0
    assert payload["config"]["input_conditioning_mode"] == "concat_film"
    assert payload["config"]["input_film_strength"] == 0.25
    assert payload["input_features"]["conditioning_mode"] == "concat_film"
    assert payload["input_features"]["film_strength"] == 0.25
    assert payload["input_features"]["normalization"]["mode"] == "global_standard"
    assert payload["input_features"]["normalization"]["fit_scope"] == "global"
    assert checkpoint["metadata"]["param_dim"] == 3
    assert checkpoint["metadata"]["input_features"]["conditioning_mode"] == "concat_film"
    assert checkpoint["metadata"]["input_features"]["film_strength"] == 0.25


def test_macro_pinn_training_cli_supports_routed_process_conditioning(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "toy_process_temperature.csv"
    table.write_text(
        "x,y,t,T,line_id,laser_power_W,scan_speed_mm_s,spot_size_um\n"
        "0,0,0,10,Line_0_1,285,960,67\n"
        "1,0,0,11,Line_0_1,285,960,67\n"
        "0,1,1,20,Line_3_1,325,960,67\n"
        "1,1,1,21,Line_3_1,325,960,67\n",
        encoding="utf-8",
    )
    split = tmp_path / "split.json"
    split.write_text(
        '{"splits":{"train":[0,1],"val":[2],"test":[3]}}',
        encoding="utf-8",
    )
    output_dir = tmp_path / "routed_conditioned_run"

    status = main(
        [
            "--table",
            str(table),
            "--target",
            "T",
            "--output-dir",
            str(output_dir),
            "--steps",
            "2",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--split-manifest",
            str(split),
            "--input-normalization",
            "standard",
            "--input-conditioning-mode",
            "routed",
            "--input-route-film-prior",
            "0.8",
            "--freeze-input-route",
            "--input-feature-normalization",
            "global_standard",
            "--input-feature-column",
            "laser_power_W",
            "--input-feature-column",
            "scan_speed_mm_s",
            "--input-feature-column",
            "spot_size_um",
            "--log-every",
            "1",
        ]
    )

    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = __import__("torch").load(output_dir / "checkpoint.pt", map_location="cpu")

    assert status == 0
    assert payload["config"]["input_conditioning_mode"] == "routed"
    assert payload["config"]["input_route_film_prior"] == 0.8
    assert payload["config"]["input_route_trainable"] is False
    assert payload["input_features"]["conditioning_mode"] == "routed"
    assert payload["input_features"]["route"]["enabled"] is True
    assert payload["input_features"]["route"]["film_prior"] == 0.8
    assert payload["input_features"]["route"]["trainable"] is False
    assert payload["input_features"]["route"]["summary"]["film_gate_mean"] == pytest.approx(0.8)
    assert checkpoint["metadata"]["param_dim"] == 3
    assert checkpoint["metadata"]["input_features"]["conditioning_mode"] == "routed"
    assert checkpoint["metadata"]["input_features"]["route"]["summary"]["film_gate_mean"] == pytest.approx(0.8)


def test_macro_pinn_training_cli_supports_process_axis_conditioning_profile(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "toy_process_temperature.csv"
    table.write_text(
        "x,y,t,T,line_id,laser_power_W,scan_speed_mm_s,spot_size_um\n"
        "0,0,0,10,Line_0_1,285,960,67\n"
        "1,0,0,11,Line_0_1,285,960,67\n"
        "0,1,1,20,Line_3_1,325,960,82\n"
        "1,1,1,21,Line_3_1,325,960,82\n",
        encoding="utf-8",
    )
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "group_key": "spot_size_um",
                "splits": {"train": [0, 1], "val": [2], "test": [3]},
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "profiled_conditioned_run"

    status = main(
        [
            "--table",
            str(table),
            "--target",
            "T",
            "--output-dir",
            str(output_dir),
            "--steps",
            "2",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--split-manifest",
            str(split),
            "--input-normalization",
            "standard",
            "--input-conditioning-mode",
            "concat",
            "--input-feature-normalization",
            "same",
            "--input-conditioning-profile",
            "process_axis_v1",
            "--input-feature-column",
            "laser_power_W",
            "--input-feature-column",
            "scan_speed_mm_s",
            "--input-feature-column",
            "spot_size_um",
            "--log-every",
            "1",
        ]
    )

    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    profile = payload["input_features"]["conditioning_profile"]

    assert status == 0
    assert payload["config"]["input_conditioning_mode"] == "film"
    assert payload["config"]["input_feature_normalization"] == "global_standard"
    assert payload["input_features"]["conditioning_mode"] == "film"
    assert payload["input_features"]["normalization"]["mode"] == "global_standard"
    assert profile["enabled"] is True
    assert profile["profile"] == "process_axis_v1"
    assert profile["group_key"] == "spot_size_um"
    assert profile["requested"]["conditioning_mode"] == "concat"
    assert profile["requested"]["feature_normalization"] == "same"
    assert profile["selected"]["conditioning_mode"] == "film"
    assert profile["selected"]["feature_normalization"] == "global_standard"


def test_macro_pinn_training_cli_process_profile_uses_line_like_route(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "toy_process_temperature.csv"
    table.write_text(
        "x,y,t,T,line_id,laser_power_W,scan_speed_mm_s,spot_size_um\n"
        "0,0,0,10,Line_0_1,285,960,67\n"
        "1,0,0,11,Line_0_1,285,960,67\n"
        "0,1,1,20,Line_3_1,325,960,82\n"
        "1,1,1,21,Line_3_1,325,960,82\n",
        encoding="utf-8",
    )
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "group_key": "process_condition",
                "splits": {"train": [0, 1], "val": [2], "test": [3]},
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "process_profiled_conditioned_run"

    status = main(
        [
            "--table",
            str(table),
            "--target",
            "T",
            "--output-dir",
            str(output_dir),
            "--steps",
            "2",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--split-manifest",
            str(split),
            "--input-normalization",
            "minmax",
            "--input-conditioning-mode",
            "film",
            "--input-feature-normalization",
            "global_standard",
            "--input-conditioning-profile",
            "process_axis_v1",
            "--input-feature-column",
            "laser_power_W",
            "--input-feature-column",
            "scan_speed_mm_s",
            "--input-feature-column",
            "spot_size_um",
            "--log-every",
            "1",
        ]
    )

    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    profile = payload["input_features"]["conditioning_profile"]

    assert status == 0
    assert payload["config"]["input_conditioning_mode"] == "concat"
    assert payload["config"]["input_feature_normalization"] == "same"
    assert payload["input_features"]["conditioning_mode"] == "concat"
    assert payload["input_features"]["normalization"]["mode"] == "minmax"
    assert profile["enabled"] is True
    assert profile["group_key"] == "process_condition"
    assert profile["requested"]["conditioning_mode"] == "film"
    assert profile["requested"]["feature_normalization"] == "global_standard"
    assert profile["selected"]["conditioning_mode"] == "concat"
    assert profile["selected"]["feature_normalization"] == "same"


def test_macro_pinn_training_cli_broad_profile_can_fall_back_to_no_process(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "toy_process_temperature.csv"
    table.write_text(
        "x,y,t,T,line_id,laser_power_W,scan_speed_mm_s,spot_size_um\n"
        "0,0,0,10,Line_0_1,285,960,67\n"
        "1,0,0,11,Line_0_1,285,960,67\n"
        "0,1,1,20,Line_3_1,325,960,82\n"
        "1,1,1,21,Line_3_1,325,960,82\n",
        encoding="utf-8",
    )
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "sample_id": "toy",
                "n_rows": 4,
                "group_key": "scan_speed_mm_s",
                "splits": {"train": [0, 1], "val": [2], "test": [3]},
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "broad_profiled_no_process_run"

    status = main(
        [
            "--table",
            str(table),
            "--target",
            "T",
            "--output-dir",
            str(output_dir),
            "--steps",
            "2",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--split-manifest",
            str(split),
            "--input-normalization",
            "minmax",
            "--input-conditioning-mode",
            "film",
            "--input-feature-normalization",
            "global_standard",
            "--input-conditioning-profile",
            "broad_process_v1",
            "--input-feature-column",
            "laser_power_W",
            "--input-feature-column",
            "scan_speed_mm_s",
            "--input-feature-column",
            "spot_size_um",
            "--log-every",
            "1",
        ]
    )

    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = __import__("torch").load(output_dir / "checkpoint.pt", map_location="cpu")
    profile = payload["input_features"]["conditioning_profile"]

    assert status == 0
    assert payload["config"]["input_conditioning_mode"] == "concat"
    assert payload["config"]["input_feature_normalization"] == "none"
    assert payload["config"]["input_feature_columns"] == []
    assert payload["input_features"]["enabled"] is False
    assert payload["input_features"]["columns"] == []
    assert payload["input_features"]["count"] == 0
    assert payload["input_features"]["normalization"] is None
    assert checkpoint["metadata"]["param_dim"] == 0
    assert profile["enabled"] is True
    assert profile["profile"] == "broad_process_v1"
    assert profile["group_key"] == "scan_speed_mm_s"
    assert profile["requested"]["conditioning_mode"] == "film"
    assert profile["requested"]["feature_normalization"] == "global_standard"
    assert profile["requested"]["feature_columns"] == [
        "laser_power_W",
        "scan_speed_mm_s",
        "spot_size_um",
    ]
    assert profile["selected"]["conditioning_mode"] == "none"
    assert profile["effective"]["conditioning_mode"] == "concat"
    assert profile["effective"]["feature_columns"] == []


def test_macro_pinn_training_cli_with_sparse_closure_writes_expression(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "toy_temperature.csv"
    table.write_text(
        "x,y,t,T\n"
        "0,0,0,0\n"
        "1,0,0,1\n"
        "0,1,0,1\n"
        "1,1,1,3\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "closure_run"

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
            "--pde-weight",
            "1e-4",
            "--pde-field",
            "normalized",
            "--closure-mode",
            "sparse_linear",
            "--closure-feature",
            "T",
            "--closure-feature",
            "x",
            "--closure-feature",
            "t",
            "--closure-polynomial-order",
            "1",
            "--closure-l1-weight",
            "1e-5",
        ]
    )

    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = __import__("torch").load(output_dir / "checkpoint.pt", map_location="cpu")

    assert status == 0
    assert payload["pde"]["field"] == "normalized"
    assert payload["closure"]["enabled"] is True
    assert payload["closure"]["term_names"] == ["1", "T", "x", "t"]
    assert "coefficients" in payload["closure"]
    assert "expression" in payload["closure"]
    assert checkpoint["metadata"]["closure"]["mode"] == "sparse_linear"


def test_macro_pinn_sparse_closure_uses_default_features(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "toy_temperature.csv"
    table.write_text(
        "x,y,t,T\n"
        "0,0,0,0\n"
        "1,0,0,1\n"
        "0,1,0,1\n"
        "1,1,1,3\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "default_closure_run"

    status = main(
        [
            "--table",
            str(table),
            "--target",
            "T",
            "--output-dir",
            str(output_dir),
            "--steps",
            "1",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--pde-weight",
            "1e-4",
            "--pde-field",
            "normalized",
            "--closure-mode",
            "sparse_linear",
        ]
    )

    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

    assert status == 0
    assert payload["closure"]["term_names"] == ["1", "T", "x", "y", "t"]


def test_macro_pinn_sparse_closure_supports_residual_sampling(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "toy_temperature.csv"
    table.write_text(
        "x,y,t,T\n"
        "0,0,0,0\n"
        "1,0,0,1\n"
        "0,1,0,1\n"
        "1,1,1,3\n"
        "2,1,1,4\n"
        "1,2,1,4\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "sampled_residual_run"

    status = main(
        [
            "--table",
            str(table),
            "--target",
            "T",
            "--output-dir",
            str(output_dir),
            "--steps",
            "2",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--pde-weight",
            "1e-4",
            "--pde-field",
            "normalized",
            "--closure-mode",
            "sparse_linear",
            "--residual-sample-size",
            "2",
            "--residual-sampling-seed",
            "11",
            "--log-every",
            "1",
        ]
    )

    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

    assert status == 0
    assert payload["pde"]["residual_sample_size"] == 2
    assert payload["pde"]["residual_sampling_seed"] == 11
    assert payload["history"][0]["residual_points"] == 2.0
    assert payload["history"][-1]["residual_points"] == 2.0


def test_macro_pinn_sparse_closure_supports_hot_residual_sampling(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "toy_temperature.csv"
    table.write_text(
        "x,y,t,T\n"
        "0,0,0,0\n"
        "1,0,0,1\n"
        "0,1,0,1\n"
        "1,1,1,3\n"
        "2,1,1,10\n"
        "1,2,1,11\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "hot_residual_run"

    status = main(
        [
            "--table",
            str(table),
            "--target",
            "T",
            "--output-dir",
            str(output_dir),
            "--steps",
            "2",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--pde-weight",
            "1e-4",
            "--pde-field",
            "normalized",
            "--closure-mode",
            "sparse_linear",
            "--residual-sampling-mode",
            "hot",
            "--residual-hot-quantile",
            "0.8",
            "--residual-sample-size",
            "1",
            "--log-every",
            "1",
        ]
    )

    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

    assert status == 0
    assert payload["pde"]["residual_sampling_mode"] == "hot"
    assert payload["pde"]["residual_candidate_points"] == 2
    assert payload["history"][0]["residual_candidates"] == 2.0
    assert payload["history"][0]["residual_points"] == 1.0


def test_macro_pinn_sparse_closure_can_start_after_warmup(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "toy_temperature.csv"
    table.write_text(
        "x,y,t,T\n"
        "0,0,0,0\n"
        "1,0,0,1\n"
        "0,1,0,1\n"
        "1,1,1,3\n"
        "2,1,1,10\n"
        "1,2,1,11\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "warmup_closure_run"

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
            "--pde-weight",
            "1e-4",
            "--pde-field",
            "normalized",
            "--closure-mode",
            "sparse_linear",
            "--residual-sample-size",
            "2",
            "--closure-start-step",
            "2",
            "--log-every",
            "1",
        ]
    )

    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

    assert status == 0
    assert payload["pde"]["closure_start_step"] == 2
    assert payload["history"][0]["closure_stage_active"] is False
    assert payload["history"][0]["residual_points"] == 0.0
    assert payload["history"][-1]["closure_stage_active"] is True
    assert payload["history"][-1]["residual_points"] == 2.0


def test_macro_pinn_sparse_closure_supports_separate_closure_lr(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "toy_temperature.csv"
    table.write_text(
        "x,y,t,T\n"
        "0,0,0,0\n"
        "1,0,0,1\n"
        "0,1,0,1\n"
        "1,1,1,3\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "closure_lr_run"

    status = main(
        [
            "--table",
            str(table),
            "--target",
            "T",
            "--output-dir",
            str(output_dir),
            "--steps",
            "2",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--lr",
            "1e-3",
            "--closure-lr",
            "1e-5",
            "--pde-weight",
            "1e-4",
            "--pde-field",
            "normalized",
            "--closure-mode",
            "sparse_linear",
        ]
    )

    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = __import__("torch").load(output_dir / "checkpoint.pt", map_location="cpu")

    assert status == 0
    assert payload["config"]["closure_lr"] == 1e-5
    assert payload["optimizer"]["backbone_lr"] == 1e-3
    assert payload["optimizer"]["closure_lr"] == 1e-5
    assert payload["optimizer"]["closure_lr_overridden"] is True
    assert checkpoint["metadata"]["optimizer"]["closure_lr"] == 1e-5


def test_macro_pinn_sparse_closure_can_freeze_backbone_after_warmup(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "toy_temperature.csv"
    table.write_text(
        "x,y,t,T\n"
        "0,0,0,0\n"
        "1,0,0,1\n"
        "0,1,0,1\n"
        "1,1,1,3\n"
        "2,1,1,10\n"
        "1,2,1,11\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "frozen_backbone_run"

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
            "--pde-weight",
            "1e-4",
            "--pde-field",
            "normalized",
            "--closure-mode",
            "sparse_linear",
            "--residual-sample-size",
            "2",
            "--closure-start-step",
            "2",
            "--freeze-backbone-after-closure-start",
            "--log-every",
            "1",
        ]
    )

    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

    assert status == 0
    assert payload["optimizer"]["freeze_backbone_after_closure_start"] is True
    assert payload["history"][0]["backbone_frozen"] is False
    assert payload["history"][-1]["closure_stage_active"] is True
    assert payload["history"][-1]["backbone_frozen"] is True


def test_macro_pinn_sparse_closure_supports_toy_graph_conditioning(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "toy_temperature.csv"
    table.write_text(
        "x,y,t,T\n"
        "0,0,0,0\n"
        "1,0,0,1\n"
        "0,1,0,1\n"
        "1,1,1,3\n"
        "2,1,1,10\n"
        "1,2,1,11\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "graph_closure_run"

    status = main(
        [
            "--table",
            str(table),
            "--target",
            "T",
            "--output-dir",
            str(output_dir),
            "--steps",
            "2",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--pde-weight",
            "1e-4",
            "--pde-field",
            "normalized",
            "--closure-mode",
            "sparse_linear",
            "--closure-graph-mode",
            "toy_static",
            "--closure-graph-embedding-dim",
            "2",
            "--closure-graph-hidden-dim",
            "8",
            "--residual-sample-size",
            "2",
            "--log-every",
            "1",
        ]
    )

    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

    assert status == 0
    assert payload["closure"]["graph_conditioning"]["enabled"] is True
    assert payload["closure"]["graph_conditioning"]["mode"] == "toy_static"
    assert payload["closure"]["graph_conditioning"]["metadata"]["feature_names"] == ["g0", "g1"]
    assert payload["closure"]["term_names"] == ["1", "T", "x", "y", "t", "g0", "g1"]
    assert payload["config"]["closure_features"] == ["T", "x", "y", "t", "g0", "g1"]


def test_macro_pinn_sparse_closure_supports_coordinate_rbf_graph_features(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "toy_temperature.csv"
    table.write_text(
        "x,y,t,T\n"
        "0,0,0,0\n"
        "1,0,0,1\n"
        "0,1,0,1\n"
        "1,1,1,3\n"
        "2,1,1,10\n"
        "1,2,1,11\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "coordinate_graph_closure_run"

    status = main(
        [
            "--table",
            str(table),
            "--target",
            "T",
            "--output-dir",
            str(output_dir),
            "--steps",
            "2",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--pde-weight",
            "1e-4",
            "--pde-field",
            "normalized",
            "--closure-mode",
            "sparse_linear",
            "--closure-graph-mode",
            "coordinate_rbf",
            "--closure-graph-embedding-dim",
            "3",
            "--closure-graph-length-scale",
            "0.5",
            "--residual-sample-size",
            "2",
            "--log-every",
            "1",
        ]
    )

    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

    assert status == 0
    assert payload["closure"]["graph_conditioning"]["mode"] == "coordinate_rbf"
    assert payload["closure"]["graph_conditioning"]["metadata"]["feature_names"] == ["g0", "g1", "g2"]
    assert payload["closure"]["graph_conditioning"]["metadata"]["state_dim"] == 3
    assert payload["closure"]["graph_conditioning"]["metadata"]["length_scale"] == 0.5
    assert payload["closure"]["term_names"] == ["1", "T", "x", "y", "t", "g0", "g1", "g2"]
    assert payload["config"]["closure_features"] == ["T", "x", "y", "t", "g0", "g1", "g2"]


def test_macro_pinn_graph_conditioned_closure_records_gate_and_graph_l1(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "toy_temperature.csv"
    table.write_text(
        "x,y,t,T\n"
        "0,0,0,0\n"
        "1,0,0,1\n"
        "0,1,0,1\n"
        "1,1,1,3\n"
        "2,1,1,10\n"
        "1,2,1,11\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "gated_graph_closure_run"

    status = main(
        [
            "--table",
            str(table),
            "--target",
            "T",
            "--output-dir",
            str(output_dir),
            "--steps",
            "2",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--pde-weight",
            "1e-4",
            "--pde-field",
            "normalized",
            "--closure-mode",
            "sparse_linear",
            "--closure-l1-weight",
            "1e-5",
            "--closure-graph-mode",
            "coordinate_rbf",
            "--closure-graph-embedding-dim",
            "2",
            "--closure-graph-gate",
            "0.1",
            "--closure-graph-l1-weight",
            "1e-4",
            "--residual-sample-size",
            "2",
            "--log-every",
            "1",
        ]
    )

    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

    assert status == 0
    assert payload["closure"]["graph_gate"] == 0.1
    assert payload["closure"]["graph_l1_weight"] == 1e-4
    assert payload["config"]["closure_graph_gate"] == 0.1
    assert payload["config"]["closure_graph_l1_weight"] == 1e-4


def test_macro_pinn_sparse_closure_supports_real_micro_graph_conditioning(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "field.csv"
    table.write_text(
        "x,y,t,T\n"
        "0,0,0,1\n"
        "1,0,0,2\n"
        "0,1,1,3\n"
        "1,1,1,4\n",
        encoding="utf-8",
    )
    graph_features = tmp_path / "micro_features.jsonl"
    graph_features.write_text(
        json.dumps(
            {
                "sample_id": "sample_a",
                "sample_metadata": {"process": "P2"},
                "feature_names": [
                    "image_mask_fraction",
                    "node_mask_fraction_mean",
                    "node_mask_fraction_std",
                ],
                "features": {
                    "image_mask_fraction": 0.1,
                    "node_mask_fraction_mean": 0.2,
                    "node_mask_fraction_std": 0.05,
                },
                "graph_summary": {"num_nodes": 64},
            }
        )
        + "\n",
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
            "2",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--pde-weight",
            "1e-4",
            "--closure-mode",
            "sparse_linear",
            "--closure-feature",
            "T",
            "--closure-graph-mode",
            "real_micro",
            "--closure-graph-features",
            str(graph_features),
            "--closure-graph-sample-id",
            "sample_a",
            "--closure-graph-embedding-dim",
            "3",
            "--log-every",
            "1",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    graph_payload = payload["closure"]["graph_conditioning"]
    assert graph_payload["mode"] == "real_micro"
    assert graph_payload["metadata"]["sample_id"] == "sample_a"
    assert graph_payload["metadata"]["source_feature_names"][:2] == [
        "image_mask_fraction",
        "node_mask_fraction_mean",
    ]
    assert payload["closure"]["term_names"] == ["1", "T", "g0", "g1", "g2"]
    assert any(name.startswith("g") for name in payload["closure"]["term_names"])


def test_macro_pinn_real_micro_graph_conditioning_supports_sample_id_column(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "field.csv"
    table.write_text(
        "x,y,t,T,micro_sample_id\n"
        "0,0,0,1,sample_a\n"
        "1,0,0,2,sample_b\n"
        "0,1,1,3,sample_a\n"
        "1,1,1,4,sample_b\n",
        encoding="utf-8",
    )
    graph_features = tmp_path / "micro_features_panel.jsonl"
    records = [
        {
            "sample_id": "sample_a",
            "sample_metadata": {"process": "P1"},
            "feature_names": [
                "image_mask_fraction",
                "node_mask_fraction_mean",
            ],
            "features": {
                "image_mask_fraction": 0.1,
                "node_mask_fraction_mean": 0.2,
            },
            "graph_summary": {"num_nodes": 64},
        },
        {
            "sample_id": "sample_b",
            "sample_metadata": {"process": "P2"},
            "feature_names": [
                "image_mask_fraction",
                "node_mask_fraction_mean",
            ],
            "features": {
                "image_mask_fraction": 0.7,
                "node_mask_fraction_mean": 0.4,
            },
            "graph_summary": {"num_nodes": 64},
        },
    ]
    graph_features.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
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
            "2",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--pde-weight",
            "1e-4",
            "--closure-mode",
            "sparse_linear",
            "--closure-feature",
            "T",
            "--closure-graph-mode",
            "real_micro",
            "--closure-graph-features",
            str(graph_features),
            "--closure-graph-sample-id-column",
            "micro_sample_id",
            "--closure-graph-embedding-dim",
            "2",
            "--no-closure-graph-normalize",
            "--log-every",
            "1",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    graph_payload = payload["closure"]["graph_conditioning"]
    assert graph_payload["mode"] == "real_micro"
    assert graph_payload["selection"]["sample_id_column"] == "micro_sample_id"
    assert graph_payload["metadata"]["available_sample_ids"] == ["sample_a", "sample_b"]
    assert payload["config"]["closure_graph_sample_id_column"] == "micro_sample_id"


def test_macro_pinn_sparse_closure_supports_real_micro_region_graph_conditioning(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "field.csv"
    table.write_text(
        "x,y,t,T\n"
        "0,0,0,1\n"
        "1,0,0,2\n"
        "0,1,1,3\n"
        "1,1,1,4\n",
        encoding="utf-8",
    )
    graph_features = tmp_path / "micro_region_features.jsonl"
    graph_features.write_text(
        json.dumps(
            {
                "sample_id": "sample_a",
                "sample_metadata": {"process": "P3"},
                "feature_names": ["image_mask_fraction"],
                "features": {"image_mask_fraction": 0.1},
                "region_feature_names": [
                    "center_row_norm",
                    "center_col_norm",
                    "mean_intensity_norm",
                    "std_intensity_norm",
                    "mask_fraction",
                ],
                "region_features": [
                    [0.25, 0.25, 0.1, 0.01, 0.0],
                    [0.25, 0.75, 0.2, 0.02, 0.2],
                    [0.75, 0.25, 0.3, 0.03, 0.4],
                    [0.75, 0.75, 0.4, 0.04, 0.8],
                ],
                "graph_summary": {"num_nodes": 4},
            }
        )
        + "\n",
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
            "2",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--pde-weight",
            "1e-4",
            "--closure-mode",
            "sparse_linear",
            "--closure-feature",
            "T",
            "--closure-graph-mode",
            "real_micro_region",
            "--closure-graph-features",
            str(graph_features),
            "--closure-graph-sample-id",
            "sample_a",
            "--closure-graph-embedding-dim",
            "3",
            "--no-closure-graph-normalize",
            "--closure-graph-region-row-source",
            "x",
            "--closure-graph-region-col-source",
            "y",
            "--closure-graph-region-flip-row",
            "--closure-graph-region-selection",
            "inverse_distance",
            "--log-every",
            "1",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    graph_payload = payload["closure"]["graph_conditioning"]
    assert graph_payload["mode"] == "real_micro_region"
    assert graph_payload["metadata"]["sample_id"] == "sample_a"
    assert graph_payload["metadata"]["source_feature_names"] == [
        "mask_fraction",
        "mean_intensity_norm",
        "std_intensity_norm",
    ]
    assert graph_payload["metadata"]["region_counts_by_sample_id"] == {"sample_a": 4}
    assert graph_payload["metadata"]["coordinate_mapping"]["row_source"] == "x"
    assert graph_payload["metadata"]["coordinate_mapping"]["col_source"] == "y"
    assert graph_payload["metadata"]["coordinate_mapping"]["flip_row"] is True
    assert graph_payload["metadata"]["coordinate_mapping"]["selection"] == "inverse_distance"
    assert payload["config"]["closure_graph_region_row_source"] == "x"
    assert payload["config"]["closure_graph_region_selection"] == "inverse_distance"
    assert payload["closure"]["term_names"] == ["1", "T", "g0", "g1", "g2"]


def test_macro_pinn_sparse_closure_supports_real_micro_region_embedding_graph_conditioning(tmp_path: Path):
    from gnnpinn.train.macro_pinn import main

    table = tmp_path / "field.csv"
    table.write_text(
        "x,y,t,T\n"
        "0,0,0,1\n"
        "1,0,0,2\n"
        "0,1,1,3\n"
        "1,1,1,4\n",
        encoding="utf-8",
    )
    graph_features = tmp_path / "micro_region_embedding_features.jsonl"
    graph_features.write_text(
        json.dumps(
            {
                "sample_id": "sample_a",
                "sample_metadata": {"process": "P4"},
                "feature_names": ["image_mask_fraction"],
                "features": {"image_mask_fraction": 0.1},
                "region_feature_names": [
                    "center_row_norm",
                    "center_col_norm",
                    "mean_intensity_norm",
                    "std_intensity_norm",
                    "mask_fraction",
                ],
                "region_features": [
                    [0.25, 0.25, 0.1, 0.01, 0.0],
                    [0.25, 0.75, 0.2, 0.02, 0.2],
                    [0.75, 0.25, 0.3, 0.03, 0.4],
                    [0.75, 0.75, 0.4, 0.04, 0.8],
                ],
                "region_embedding_feature_names": [
                    "patch_embedding_0",
                    "patch_embedding_1",
                    "patch_embedding_2",
                ],
                "region_embedding_features": [
                    [1.0, 0.1, 0.0],
                    [2.0, 0.2, 0.0],
                    [3.0, 0.3, 0.0],
                    [4.0, 0.4, 0.0],
                ],
                "region_embedding_metadata": {
                    "method": "pca_lifted_region_descriptors",
                    "embedding_dim": 3,
                },
                "graph_summary": {"num_nodes": 4},
            }
        )
        + "\n",
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
            "2",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--pde-weight",
            "1e-4",
            "--closure-mode",
            "sparse_linear",
            "--closure-feature",
            "T",
            "--closure-graph-mode",
            "real_micro_region_embedding",
            "--closure-graph-features",
            str(graph_features),
            "--closure-graph-sample-id",
            "sample_a",
            "--closure-graph-embedding-dim",
            "3",
            "--no-closure-graph-normalize",
            "--closure-graph-region-flip-col",
            "--log-every",
            "1",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    graph_payload = payload["closure"]["graph_conditioning"]
    assert graph_payload["mode"] == "real_micro_region_embedding"
    assert graph_payload["metadata"]["source_feature_names"] == [
        "patch_embedding_0",
        "patch_embedding_1",
        "patch_embedding_2",
    ]
    assert graph_payload["metadata"]["region_embedding_metadata_by_sample_id"]["sample_a"]["method"] == (
        "pca_lifted_region_descriptors"
    )
    assert graph_payload["metadata"]["coordinate_mapping"]["flip_col"] is True
    assert payload["config"]["closure_graph_mode"] == "real_micro_region_embedding"
    assert payload["closure"]["term_names"] == ["1", "T", "g0", "g1", "g2"]
