from __future__ import annotations

from pathlib import Path

import pytest

from gnnpinn.data.loaders.ambench_microstructure import (
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
