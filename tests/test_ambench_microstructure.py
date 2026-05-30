from __future__ import annotations

from pathlib import Path

import pytest

from gnnpinn.data.loaders.ambench_microstructure import (
    build_graph_feature_table,
    graph_record_from_inspection,
    inspect_microstructure_image,
    parse_ambench_microstructure_filename,
)


def test_inspect_microstructure_image_builds_grid_graph(tmp_path: Path):
    import numpy as np

    skimage_io = pytest.importorskip("skimage.io")
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    image_path = tmp_path / "toy.tif"
    skimage_io.imsave(image_path, image)

    report = inspect_microstructure_image(
        image_path,
        sample_id="toy_micro",
        threshold_quantile=0.75,
        grid_rows=2,
        grid_cols=2,
        graph_k=1,
    )

    assert report["sample_id"] == "toy_micro"
    assert report["image"]["gray_shape"] == [8, 8]
    assert report["graph"]["num_nodes"] == 4
    assert report["graph"]["num_edges"] > 0
    assert report["graph"]["node_feature_names"] == [
        "center_row_norm",
        "center_col_norm",
        "mean_intensity_norm",
        "std_intensity_norm",
        "mask_fraction",
    ]
    assert len(report["graph"]["node_features"][0]) == 5
    assert 0.0 < report["image"]["statistics"]["mask_fraction"] < 1.0


def test_inspect_microstructure_image_returns_payload_with_mocked_reader(monkeypatch, tmp_path: Path):
    import numpy as np
    import gnnpinn.data.loaders.ambench_microstructure as module

    monkeypatch.setattr(module, "_read_image", lambda path: np.arange(16, dtype=np.uint8).reshape(4, 4))
    image_path = tmp_path / "AMB2022-718-SH1-BP1-P2-L2.1-3_m.tif"

    report = inspect_microstructure_image(
        image_path,
        sample_id="mocked",
        threshold_quantile=0.75,
        grid_rows=2,
        grid_cols=2,
        graph_k=1,
    )

    assert report["sample_id"] == "mocked"
    assert report["sample_metadata"]["process"] == "P2"
    assert report["image"]["gray_shape"] == [4, 4]
    assert report["graph"]["num_nodes"] == 4


def test_parse_ambench_microstructure_filename_single_track():
    metadata = parse_ambench_microstructure_filename("AMB2022-718-SH1-BP1-P2-L2.1-3_m.tif")

    assert metadata["parsed"] is True
    assert metadata["masked"] is True
    assert metadata["alloy"] == "718"
    assert metadata["build_plate"] == "BP1"
    assert metadata["process"] == "P2"
    assert metadata["process_index"] == 2
    assert metadata["line"] == "L2.1"
    assert metadata["line_value"] == 2.1
    assert metadata["replicate_index"] == 3


def test_parse_ambench_microstructure_filename_top_view():
    metadata = parse_ambench_microstructure_filename("AMB2022-718-SH1-BP1-BEFORE.tif")

    assert metadata["parsed"] is True
    assert metadata["masked"] is False
    assert metadata["build_plate_index"] == 1
    assert metadata["view"] == "BEFORE"


def test_build_graph_feature_table_writes_jsonl_and_csv(tmp_path: Path):
    inspection = {
        "dataset_id": "mds2-2718",
        "sample_id": "toy_micro",
        "source": "toy.tif",
        "sample_metadata": {"process": "P2"},
        "image": {
            "shape": [4, 4],
            "gray_shape": [4, 4],
            "statistics": {
                "mean": 10.0,
                "std": 2.0,
                "mask_fraction": 0.25,
            },
        },
        "graph": {
            "num_nodes": 2,
            "num_edges": 2,
            "node_feature_names": ["center_row_norm", "mask_fraction"],
            "node_features": [[0.25, 0.0], [0.75, 0.5]],
        },
    }
    inspection_path = tmp_path / "inspection.json"
    inspection_path.write_text(__import__("json").dumps(inspection), encoding="utf-8")
    jsonl_output = tmp_path / "features.jsonl"
    csv_output = tmp_path / "features.csv"

    report = build_graph_feature_table(
        [inspection_path],
        jsonl_output=jsonl_output,
        csv_output=csv_output,
    )
    record = graph_record_from_inspection(inspection)

    assert report["n_records"] == 1
    assert "image_mask_fraction" in report["feature_names"]
    assert record["features"]["node_mask_fraction_mean"] == 0.25
    assert jsonl_output.read_text(encoding="utf-8").count("\n") == 1
    assert "image_mask_fraction" in csv_output.read_text(encoding="utf-8")
