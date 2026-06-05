from __future__ import annotations

import csv
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


def _load_module():
    script = Path("scripts/server/phase114_nist_ammt_gcode_strategy_source_gate.py")
    spec = importlib.util.spec_from_file_location("phase114_gcode_gate", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gcode(strategy_id: int, *, layers: int = 8, base_power: int = 100) -> str:
    lines = []
    line_number = 1
    for layer in range(1, layers + 1):
        z = layer * 0.02
        power = base_power + strategy_id * 10 + layer
        for step in range(4):
            x = -1.0 + step
            y = -1.0 + ((step + layer + strategy_id) % 3)
            feed = 800 if step % 2 == 0 else 500
            laser = power if step % 2 == 0 else 0
            lines.append(f"N{line_number:04d} G01 X{x:.4f} Y{y:.4f} Z{z:.4f} F{feed} L{laser}")
            line_number += 1
    return "\n".join(lines) + "\n"


def _generator_csv(mode_island: int, hatch_space: float = 0.1) -> str:
    return (
        "Variable,Value,Remark\n"
        f"mode_island,{mode_island},synthetic island mode\n"
        f"hatch_space,{hatch_space},synthetic hatch space\n"
    )


def _interpreter_csv(mode_laser_power: int, mode_laser_path: int) -> str:
    return (
        "Variable,Value,Remark\n"
        f"mode_laser_power,{mode_laser_power},synthetic laser power mode\n"
        f"mode_laser_path,{mode_laser_path},synthetic laser path mode\n"
        "laser_density,0.00125,synthetic density\n"
    )


def _write_build_zip(root: Path, *, strategy_count: int = 4, layers: int = 8) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "Build Command Data.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for strategy_id in range(1, strategy_count + 1):
            name = f"{strategy_id}_synthetic_strategy"
            prefix = f"Build Command Data/AM Gcode/{name}"
            archive.writestr(f"{prefix}/{name}.gcode", _gcode(strategy_id, layers=layers))
            archive.writestr(
                f"{prefix}/Gcode_Generator_par.csv",
                _generator_csv(mode_island=strategy_id % 2, hatch_space=0.1 * strategy_id),
            )
            archive.writestr(
                f"{prefix}/Gcode_interpreter_par.csv",
                _interpreter_csv(mode_laser_power=1 + strategy_id % 2, mode_laser_path=1 + strategy_id % 3),
            )
    return path


def _write_join_gate(path: Path, *, ready: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "source_target_layer_join_ready_timing_not_absolute",
                "layer_camera_join_ready": ready,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_registered_table(path: Path, *, rows_per_split: int = 4) -> Path:
    fieldnames = [
        "x",
        "y",
        "t",
        "source_layer_index",
        "target_layer_index",
        "target_camera_code",
        "source_p_mean",
        "source_p_nonzero_fraction",
        "source_p_range",
        "source_x_range",
        "source_y_range",
        "source_rows_read",
        "target_intensity_std",
        "target_center_periphery_contrast",
        "target_grid_mean_range",
        "target_quadrant_contrast",
        "split_name",
        "row_id",
        "source_member_name",
        "target_member_name",
    ]
    rows = []
    split_names = ["train"] * rows_per_split + ["val"] * rows_per_split + ["test"] * rows_per_split
    for index, split in enumerate(split_names, start=1):
        layer = index
        camera = index % 2
        rows.append(
            {
                "x": index * 0.1,
                "y": -index * 0.1,
                "t": layer,
                "source_layer_index": layer,
                "target_layer_index": layer,
                "target_camera_code": camera,
                "source_p_mean": 120 + layer,
                "source_p_nonzero_fraction": 0.5,
                "source_p_range": 100,
                "source_x_range": 10,
                "source_y_range": 10,
                "source_rows_read": 1000,
                "target_intensity_std": 10 + layer * 0.5 + camera,
                "target_center_periphery_contrast": 20 + layer + camera * 3,
                "target_grid_mean_range": 30 + layer * 0.25,
                "target_quadrant_contrast": 40 + camera,
                "split_name": split,
                "row_id": f"row-{index}",
                "source_member_name": (
                    "Build Command Data/XYPT Commands/"
                    f"T500_3D_Scan_Strategies_fused_layer{layer:04d}.csv"
                ),
                "target_member_name": f"In-situ Meas Data/Layer Camera/A{layer:04d}.bmp",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_gcode_strategy_gate_writes_artifacts_and_keeps_training_locked(tmp_path: Path):
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed in this environment")
    module = _load_module()
    data_root = tmp_path / "data"
    _write_build_zip(data_root, strategy_count=4, layers=12)

    manifest = module.build_package(
        root=Path(".").resolve(),
        data_root=data_root,
        registered_field_table=_write_registered_table(tmp_path / "registered.csv"),
        join_gate_path=_write_join_gate(tmp_path / "join_gate.json"),
        output_dir=tmp_path / "out",
        target_columns=("target_intensity_std", "target_center_periphery_contrast"),
        target_priority=("target_center_periphery_contrast", "target_intensity_std"),
        min_rows_for_review=10,
        min_strategy_count=4,
        min_validation_relative_improvement=0.0,
        min_xypt_guard_relative_improvement=0.0,
        min_shortcut_val_rmse_delta=1e9,
        n_neighbors=1,
        n_estimators=5,
    )

    gate = manifest["gate"]
    assert gate["status"] in {
        "phase114_gcode_strategy_source_gate_closed_no_guarded_baseline_gap",
        "phase114_gcode_strategy_source_gap_ready_focused_review",
    }
    assert gate["row_count"] == 12
    assert gate["strategy_summary"]["strategy_count"] == 4
    assert gate["phase114_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False

    with (tmp_path / "out/phase114_nist_ammt_gcode_strategy_field_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert "gcode_layer_motion_length" in rows[0]
    assert "gcode_mode_laser_power" in rows[0]

    with (tmp_path / "out/phase114_nist_ammt_gcode_strategy_metric_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        metric_rows = list(csv.DictReader(handle))
    assert len(metric_rows) == 2 * (1 + 5 * 3) * 3


def test_gcode_strategy_gate_closes_when_strategy_bank_is_too_small(tmp_path: Path):
    module = _load_module()
    data_root = tmp_path / "data"
    _write_build_zip(data_root, strategy_count=1, layers=4)

    manifest = module.build_package(
        root=Path(".").resolve(),
        data_root=data_root,
        registered_field_table=_write_registered_table(tmp_path / "registered.csv", rows_per_split=2),
        join_gate_path=_write_join_gate(tmp_path / "join_gate.json"),
        output_dir=tmp_path / "out",
        target_columns=("target_intensity_std",),
        target_priority=("target_intensity_std",),
        min_rows_for_review=1,
        min_strategy_count=4,
        min_validation_relative_improvement=0.0,
        min_xypt_guard_relative_improvement=0.0,
        min_shortcut_val_rmse_delta=1e9,
        n_neighbors=1,
        n_estimators=5,
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase114_gcode_strategy_source_gate_closed_insufficient_strategy_bank"
    assert gate["strategy_summary"]["strategy_count"] == 1
    assert gate["phase114_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False


def test_gcode_strategy_gate_blocks_without_layer_join(tmp_path: Path):
    module = _load_module()
    data_root = tmp_path / "data"
    _write_build_zip(data_root, strategy_count=4, layers=4)

    manifest = module.build_package(
        root=Path(".").resolve(),
        data_root=data_root,
        registered_field_table=_write_registered_table(tmp_path / "registered.csv", rows_per_split=2),
        join_gate_path=_write_join_gate(tmp_path / "join_gate.json", ready=False),
        output_dir=tmp_path / "out",
        target_columns=("target_intensity_std",),
        target_priority=("target_intensity_std",),
        min_rows_for_review=1,
        min_strategy_count=1,
        min_validation_relative_improvement=0.0,
        min_xypt_guard_relative_improvement=0.0,
        min_shortcut_val_rmse_delta=1e9,
        n_neighbors=1,
        n_estimators=5,
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase114_gcode_strategy_source_gate_blocked_no_layer_join"
    assert gate["phase114_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
