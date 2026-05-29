from pathlib import Path

import yaml

from gnnpinn.data.loaders.ambench import convert_mapped_table
from gnnpinn.data.loaders.field_table import load_field_table


def test_ambench_mapping_converter_writes_field_table_and_split(tmp_path: Path):
    raw = tmp_path / "raw_temperature_points.csv"
    raw.write_text(
        "X_mm,Y_mm,time_s,temperature_K\n"
        "0,0,0,300\n"
        "1,0,0,310\n"
        "0,1,0,320\n"
        "1,1,0,330\n"
        "2,1,0,340\n",
        encoding="utf-8",
    )
    output = tmp_path / "processed.csv"
    split = tmp_path / "split.json"
    mapping = {
        "dataset_id": "ambench_test",
        "sample_id": "toy",
        "source": str(raw),
        "output": str(output),
        "split_manifest": str(split),
        "columns": {"x": "X_mm", "y": "Y_mm", "t": "time_s"},
        "observations": {"T": "temperature_K"},
        "process_parameters": {"laser_power_W": 195},
        "splits": {"train_fraction": 0.6, "val_fraction": 0.2, "test_fraction": 0.2, "seed": 3},
    }

    manifest = convert_mapped_table(mapping)
    sample = load_field_table(output, observation_columns=["T"])

    assert manifest["n_rows"] == 5
    assert output.exists()
    assert split.exists()
    assert sample.coordinates[0] == [0.0, 0.0]
    assert sample.time[0] == 0.0
    assert sample.observations["T"][-1] == 340.0

