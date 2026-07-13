from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase182_amb2022_01_layer_space_dataset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase182_layer_space", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_block_nanmean_preserves_missingness_fraction():
    module = _load_module()
    values = np.asarray(
        [
            [1.0, 3.0, np.nan, 8.0],
            [5.0, 7.0, np.nan, 4.0],
            [2.0, 2.0, 6.0, 6.0],
            [2.0, 2.0, 6.0, 6.0],
        ],
        dtype=np.float32,
    )
    means, fractions = module._block_nanmean(values, 2)
    assert np.allclose(means[[0, 2, 3]], [4.0, 2.0, 6.0])
    assert np.isclose(means[1], 6.0)
    assert np.allclose(fractions, [1.0, 0.5, 1.0, 1.0])


def test_scan_block_features_use_only_active_in_fov_commands():
    module = _load_module()
    features = module.scan_block_features(
        x_coords=np.asarray([0.0, 1.0, 2.0, 3.0]),
        y_coords=np.asarray([0.0, 1.0, 2.0, 3.0]),
        x_commands=np.asarray([0.0, 0.9, 2.1, 20.0]),
        y_commands=np.asarray([0.0, 0.9, 2.1, 20.0]),
        power_commands=np.asarray([100.0, 200.0, 0.0, 1000.0]),
        trigger_commands=np.asarray([4, 4, 4, 4]),
        digital_rate_hz=100000.0,
        block_size=2,
    )
    assert features["laser_active"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert np.isclose(features["laser_dwell_s"][0], 2e-5)
    assert np.isclose(features["laser_energy_J"][0], 3e-3)
    assert np.isclose(features["mean_power_W"][0], 150.0)
    assert np.isclose(features["staring_trigger_count"][0], 2.0)
    assert np.allclose(features["laser_energy_J"][1:], 0.0)


def test_feature_matrix_has_no_build_identity_column():
    module = _load_module()
    features = module.scan_block_features(
        x_coords=np.asarray([0.0, 1.0, 2.0, 3.0]),
        y_coords=np.asarray([0.0, 1.0, 2.0, 3.0]),
        x_commands=np.asarray([0.0]),
        y_commands=np.asarray([0.0]),
        power_commands=np.asarray([100.0]),
        trigger_commands=np.asarray([4]),
        digital_rate_hz=100000.0,
        block_size=2,
    )
    matrix = module._feature_matrix(features, z_mm=1.2, layer_index=10, layer_count=312)
    assert matrix.shape == (4, len(module.FEATURE_NAMES))
    assert "build_id" not in module.FEATURE_NAMES
    assert np.isfinite(matrix).all()
