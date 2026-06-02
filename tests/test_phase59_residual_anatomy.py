from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/summarize_phase59_residual_anatomy.py")
    spec = importlib.util.spec_from_file_location("phase59_residual", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_predictions(path: Path, *, method: str, predictions: list[float]) -> None:
    truth = [0.0, 1.0, 10.0, 12.0, 20.0, 22.0]
    splits = ["train", "train", "test", "test", "test", "test"]
    laser = [245, 245, 245, 245, 325, 325]
    spot = [67, 67, 49, 49, 82, 82]
    line = ["Line_0", "Line_0", "Line_1", "Line_1", "Line_2", "Line_2"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_index",
                "split",
                "sample_id",
                "method",
                "x",
                "y",
                "t",
                "temperature_C",
                "prediction",
                "error",
                "abs_error",
                "laser_power_W",
                "scan_speed_mm_s",
                "spot_size_um",
                "line_id",
                "frame_index",
                "metadata_row_index",
                "col_index",
            ],
        )
        writer.writeheader()
        for index, prediction in enumerate(predictions):
            error = prediction - truth[index]
            writer.writerow(
                {
                    "row_index": index,
                    "split": splits[index],
                    "sample_id": "toy",
                    "method": method,
                    "x": index,
                    "y": 0,
                    "t": index,
                    "temperature_C": truth[index],
                    "prediction": prediction,
                    "error": error,
                    "abs_error": abs(error),
                    "laser_power_W": laser[index],
                    "scan_speed_mm_s": 960,
                    "spot_size_um": spot[index],
                    "line_id": line[index],
                    "frame_index": index,
                    "metadata_row_index": 0,
                    "col_index": index,
                }
            )


def test_phase59_residual_anatomy_surfaces_worst_group(tmp_path: Path):
    module = _load_module()
    mean = tmp_path / "mean.csv"
    no_process = tmp_path / "no_process.csv"
    broad = tmp_path / "broad.csv"
    _write_predictions(mean, method="mean", predictions=[0.0, 1.0, 12.0, 12.0, 20.0, 22.0])
    _write_predictions(no_process, method="no_process", predictions=[0.0, 1.0, 30.0, 32.0, 18.0, 20.0])
    _write_predictions(broad, method="broad_process_v1", predictions=[0.0, 1.0, 30.0, 32.0, 20.0, 22.0])
    output = tmp_path / "summary.json"

    status = module.main(
        [
            "--prediction",
            str(mean),
            "--prediction",
            str(no_process),
            "--prediction",
            str(broad),
            "--label",
            "mean",
            "--label",
            "no_process",
            "--label",
            "broad_process_v1",
            "--target",
            "temperature_C",
            "--candidate",
            "broad_process_v1",
            "--reference",
            "mean",
            "--secondary-reference",
            "no_process",
            "--analysis-split",
            "test",
            "--group-field",
            "laser_power_W",
            "--min-group-n",
            "1",
            "--json-output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["decision"]["candidate_beats_reference_rmse"] is False
    worst = payload["worst_candidate_vs_reference"][0]
    assert worst["field"] == "laser_power_W"
    assert worst["value"] == "245"
    assert worst["delta_candidate_minus_reference_rmse"] > 0


def test_phase59_runner_exports_isolated_predictions():
    text = Path("scripts/server/run_phase59_broad21_density_residual_anatomy_a100.sh").read_text(encoding="utf-8")

    assert "outputs/predictions/phase59" in text
    assert "phase59_density_anatomy" in text
    assert "PREDICTION_OUTPUT_DIR" in text
    assert "phase59_broad21_density_residual_anatomy.json" in text
    assert "phase59_broad21_density_residual_upper_bound.json" in text
    assert "--label mean" in text
    assert "--label no_process" in text
    assert "--label broad_process_v1" in text
    assert "--fit-split train" in text
    assert "--selection-split val" in text


def _write_upper_bound_predictions(path: Path, *, method: str, predictions: list[float]) -> None:
    truth = [0.0, 1.0, 10.0, 11.0, 20.0, 21.0]
    splits = ["train", "train", "val", "val", "test", "test"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_index",
                "split",
                "sample_id",
                "method",
                "x",
                "y",
                "t",
                "temperature_C",
                "prediction",
                "error",
                "abs_error",
                "laser_power_W",
                "scan_speed_mm_s",
                "spot_size_um",
                "line_id",
                "frame_index",
                "metadata_row_index",
                "col_index",
            ],
        )
        writer.writeheader()
        for index, prediction in enumerate(predictions):
            error = prediction - truth[index]
            writer.writerow(
                {
                    "row_index": index,
                    "split": splits[index],
                    "sample_id": "toy_upper",
                    "method": method,
                    "x": index,
                    "y": 0,
                    "t": index,
                    "temperature_C": truth[index],
                    "prediction": prediction,
                    "error": error,
                    "abs_error": abs(error),
                    "laser_power_W": 285,
                    "scan_speed_mm_s": 960,
                    "spot_size_um": 49,
                    "line_id": "Line_1_1",
                    "frame_index": index,
                    "metadata_row_index": 0,
                    "col_index": index,
                }
            )


def test_phase59_upper_bound_selects_from_validation_without_test_leakage(tmp_path: Path):
    script = Path("scripts/server/summarize_phase59_residual_upper_bound.py")
    spec = importlib.util.spec_from_file_location("phase59_upper", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    mean = tmp_path / "mean.csv"
    no_process = tmp_path / "no_process.csv"
    broad = tmp_path / "broad.csv"
    _write_upper_bound_predictions(mean, method="mean", predictions=[0, 1, 30, 31, 40, 41])
    _write_upper_bound_predictions(no_process, method="no_process", predictions=[0, 1, 20, 21, 30, 31])
    _write_upper_bound_predictions(broad, method="broad_process_v1", predictions=[10, 11, 20, 21, 30, 31])
    output = tmp_path / "upper.json"

    status = module.main(
        [
            "--prediction",
            str(mean),
            "--prediction",
            str(no_process),
            "--prediction",
            str(broad),
            "--label",
            "mean",
            "--label",
            "no_process",
            "--label",
            "broad_process_v1",
            "--target",
            "temperature_C",
            "--candidate",
            "broad_process_v1",
            "--reference",
            "mean",
            "--secondary-reference",
            "no_process",
            "--fit-split",
            "train",
            "--selection-split",
            "val",
            "--analysis-split",
            "test",
            "--min-fit-n",
            "1",
            "--json-output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["uses_test_for_selection"] is False
    assert payload["selected_variant"]["name"] == "broad_process_v1:train_global_bias"
    assert payload["decision"]["selected_beats_reference_rmse"] is True
    assert payload["selected_variant"]["metrics"]["test"]["rmse"] == 0.0
