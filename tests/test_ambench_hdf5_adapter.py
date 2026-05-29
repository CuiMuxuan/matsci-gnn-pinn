from argparse import Namespace
from pathlib import Path

import pytest

from gnnpinn.data.loaders.ambench_hdf5 import (
    build_frame_split_manifest,
    calibrate_signal_to_temperature_c,
    convert_thermography_hdf5,
)
from gnnpinn.data.loaders.field_table import load_field_table

h5py = pytest.importorskip("h5py")


def test_convert_thermography_hdf5_writes_field_table(tmp_path: Path):
    source = tmp_path / "thermal.h5"
    with h5py.File(source, "w") as handle:
        thermal = handle.create_group("ThermalData")
        thermal.attrs["frame_rate"] = [10.0]
        cal = handle.create_group("Calibration").create_group("ThermalCal")
        cal.attrs["Coeff_a"] = [0.9655]
        cal.attrs["Coeff_b"] = [197.2]
        cal.attrs["Coeff_c"] = [43920000.0]
        cal.attrs["Model"] = "T(x) = 14388/a/log((c*e/x+1)-b/a;"
        cal.attrs["Model_input"] = "Signal [DL]"
        cal.attrs["Model_output"] = "Emissivity-Corrected Temperature [^oC]"
        line = thermal.create_group("Line_0_1")
        line.attrs["laser_power"] = [285.0]
        line.attrs["scan_speed"] = [960.0]
        line.attrs["spot_size"] = [67.0]
        data = line.create_dataset("Signal", shape=(3, 4, 5), dtype="uint16")
        data[...] = 1
        data[2, 3, 4] = 99
        data.attrs["units"] = "digital levels"

    output = tmp_path / "field.csv"
    split = tmp_path / "split.json"
    args = Namespace(
        thermal_hdf5=source,
        dataset="ThermalData/Line_0_1/Signal",
        sample_id="toy_hdf5",
        output=output,
        manifest=None,
        split_manifest=split,
        frame_start=0,
        frame_step=2,
        max_frames=2,
        row_start=1,
        row_step=2,
        max_rows=2,
        col_start=0,
        col_step=4,
        max_cols=2,
        calibrate_temperature=False,
        min_signal=None,
        sampling_mode="uniform",
        hot_quantile=0.9,
        gradient_quantile=0.9,
        background_fraction=0.1,
        max_points_per_frame=None,
        split_strategy="random_row",
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
        seed=3,
    )

    manifest = convert_thermography_hdf5(args)
    sample = load_field_table(output, observation_columns=["signal"])

    assert manifest["n_rows"] == 8
    assert manifest["frame_indices"] == [0, 2]
    assert manifest["row_indices"] == [1, 3]
    assert manifest["col_indices"] == [0, 4]
    assert manifest["frame_rate_hz"] == 10.0
    assert sample.n_points == 8
    assert max(sample.observations["signal"]) == 99.0
    assert split.exists()


def test_convert_thermography_hdf5_can_calibrate_temperature_and_frame_split(tmp_path: Path):
    source = tmp_path / "thermal.h5"
    with h5py.File(source, "w") as handle:
        thermal = handle.create_group("ThermalData")
        thermal.attrs["frame_rate"] = [20.0]
        cal = handle.create_group("Calibration").create_group("ThermalCal")
        cal.attrs["Coeff_a"] = [0.9655]
        cal.attrs["Coeff_b"] = [197.2]
        cal.attrs["Coeff_c"] = [43920000.0]
        line = thermal.create_group("Line_0_1")
        line.attrs["laser_power"] = [285.0]
        line.attrs["scan_speed"] = [960.0]
        line.attrs["spot_size"] = [67.0]
        data = line.create_dataset("Signal", shape=(4, 2, 2), dtype="uint16")
        data[...] = 1000

    output = tmp_path / "field.csv"
    split = tmp_path / "split.json"
    args = Namespace(
        thermal_hdf5=source,
        dataset="ThermalData/Line_0_1/Signal",
        sample_id="toy_temp",
        output=output,
        manifest=None,
        split_manifest=split,
        frame_start=0,
        frame_step=1,
        max_frames=4,
        row_start=0,
        row_step=1,
        max_rows=2,
        col_start=0,
        col_step=1,
        max_cols=2,
        calibrate_temperature=True,
        min_signal=None,
        sampling_mode="uniform",
        hot_quantile=0.9,
        gradient_quantile=0.9,
        background_fraction=0.1,
        max_points_per_frame=None,
        split_strategy="frame",
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
        seed=3,
    )

    manifest = convert_thermography_hdf5(args)
    sample = load_field_table(output, observation_columns=["temperature_C"])

    assert manifest["target"] == "temperature_C"
    assert sample.n_points == 16
    assert sample.observations["temperature_C"][0] > 0
    assert '"strategy": "frame_order"' in split.read_text(encoding="utf-8")


def test_convert_thermography_hdf5_hot_gradient_sampling_records_strategy(tmp_path: Path):
    source = tmp_path / "thermal.h5"
    with h5py.File(source, "w") as handle:
        thermal = handle.create_group("ThermalData")
        thermal.attrs["frame_rate"] = [10.0]
        line = thermal.create_group("Line_0_1")
        line.attrs["laser_power"] = [285.0]
        line.attrs["scan_speed"] = [960.0]
        line.attrs["spot_size"] = [67.0]
        data = line.create_dataset("Signal", shape=(1, 4, 4), dtype="uint16")
        data[...] = [
            [1, 1, 1, 1],
            [1, 50, 90, 1],
            [1, 10, 200, 1],
            [1, 1, 1, 1],
        ]

    output = tmp_path / "active.csv"
    args = Namespace(
        thermal_hdf5=source,
        dataset="ThermalData/Line_0_1/Signal",
        sample_id="toy_active",
        output=output,
        manifest=None,
        split_manifest=None,
        frame_start=0,
        frame_step=1,
        max_frames=1,
        row_start=0,
        row_step=1,
        max_rows=4,
        col_start=0,
        col_step=1,
        max_cols=4,
        calibrate_temperature=False,
        min_signal=None,
        sampling_mode="hot_gradient",
        hot_quantile=0.75,
        gradient_quantile=0.75,
        background_fraction=0.0,
        max_points_per_frame=None,
        split_strategy="random_row",
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
        seed=3,
    )

    manifest = convert_thermography_hdf5(args)
    sample = load_field_table(output, observation_columns=["signal"])
    sampling = manifest["metadata"]["sampling"]["selection"]

    assert 0 < manifest["n_rows"] < 16
    assert 200.0 in sample.observations["signal"]
    assert sampling["mode"] == "hot_gradient"
    assert sampling["frames"][0]["hot_points"] > 0
    assert sampling["frames"][0]["gradient_points"] > 0


def test_convert_thermography_hdf5_active_sampling_can_keep_background_anchors(tmp_path: Path):
    source = tmp_path / "thermal.h5"
    with h5py.File(source, "w") as handle:
        thermal = handle.create_group("ThermalData")
        thermal.attrs["frame_rate"] = [10.0]
        line = thermal.create_group("Line_0_1")
        data = line.create_dataset("Signal", shape=(1, 3, 3), dtype="uint16")
        data[...] = [
            [1, 1, 1],
            [1, 100, 1],
            [1, 1, 1],
        ]

    output = tmp_path / "background.csv"
    args = Namespace(
        thermal_hdf5=source,
        dataset="ThermalData/Line_0_1/Signal",
        sample_id="toy_background",
        output=output,
        manifest=None,
        split_manifest=None,
        frame_start=0,
        frame_step=1,
        max_frames=1,
        row_start=0,
        row_step=1,
        max_rows=3,
        col_start=0,
        col_step=1,
        max_cols=3,
        calibrate_temperature=False,
        min_signal=None,
        sampling_mode="balanced_hot_gradient",
        hot_quantile=0.9,
        gradient_quantile=1.0,
        background_fraction=0.25,
        max_points_per_frame=None,
        split_strategy="random_row",
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
        seed=3,
    )

    manifest = convert_thermography_hdf5(args)
    sampling = manifest["metadata"]["sampling"]["selection"]["frames"][0]

    assert sampling["background_points"] > 0
    assert manifest["n_rows"] > sampling["hot_points"]


def test_calibrate_signal_to_temperature_c_returns_positive_value():
    value = calibrate_signal_to_temperature_c(1000.0, coeff_a=0.9655, coeff_b=197.2, coeff_c=43920000.0)

    assert float(value) > 0


def test_build_frame_split_manifest_groups_rows_by_frame():
    manifest = build_frame_split_manifest(
        rows_by_frame={0: [0, 1], 10: [2, 3], 20: [4, 5], 30: [6, 7]},
        sample_id="toy",
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
    )

    assert manifest["splits"]["train"] == [0, 1, 2, 3]
    assert manifest["splits"]["val"] == [4, 5]
    assert manifest["splits"]["test"] == [6, 7]


def test_build_frame_split_manifest_uses_only_written_rows():
    manifest = build_frame_split_manifest(
        rows_by_frame={0: [], 10: [0, 1], 20: [], 30: [2]},
        sample_id="filtered",
        train_fraction=0.5,
        val_fraction=0.0,
        test_fraction=0.5,
    )

    assert manifest["n_rows"] == 3
    assert manifest["frame_splits"]["train"] == [10]
    assert manifest["splits"]["train"] == [0, 1]
    assert manifest["splits"]["test"] == [2]
