from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase186_amb2022_01_feature_ablation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase186_feature_ablation", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_causal_heat_kernel_excludes_future_source():
    module = _load_module()
    values = module.causal_heat_kernel_features(
        x_mm=np.asarray([0.0, 0.1]),
        y_mm=np.asarray([0.0, 0.0]),
        energy_j=np.asarray([1.0, 1.0]),
        local_time_s=np.asarray([0.1, 0.2]),
        alphas_mm2_s=(1.0,),
    )
    assert values.shape == (2, 1)
    assert values[0, 0] == 0.0
    assert values[1, 0] > 0.0


def test_shuffle_preserves_history_value_multiset_per_layer():
    module = _load_module()
    features = np.asarray([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]], dtype=np.float32)
    shuffled = module.shuffle_history_within_layer(
        features,
        build_index=np.asarray([0, 0, 0, 0]),
        layer_index=np.asarray([1, 1, 2, 2]),
        history_columns=[1],
        seed=1841,
    )
    assert np.allclose(shuffled[:, 0], features[:, 0])
    assert sorted(shuffled[:2, 1].tolist()) == [1.0, 2.0]
    assert sorted(shuffled[2:, 1].tolist()) == [3.0, 4.0]
