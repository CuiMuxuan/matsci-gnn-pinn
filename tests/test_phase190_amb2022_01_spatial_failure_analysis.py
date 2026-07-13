from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase190_amb2022_01_spatial_failure_analysis.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase190_spatial_failure", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase189() -> dict[str, object]:
    return {
        "gate": {
            "status": "phase189_replicate_stability_review_ready_phase190_spatial_failure_analysis",
            "phase190_spatial_failure_analysis_allowed": True,
            "model_training_allowed": False,
            "post_b8_model_reselection_allowed": False,
        }
    }


def _verification_rows(*, matches: bool = True) -> list[dict[str, object]]:
    return [
        {
            "variant_id": variant,
            "seed": seed,
            "target": target,
            "matches_phase188_rmse": matches,
        }
        for variant in ("small_data_only_mlp", "physics_regularized_history_mlp")
        for seed in (1871, 1872, 1873)
        for target in ("tam_s", "scr_C_per_s")
    ]


def test_spatial_rows_keep_registered_layer_cell_and_laser_strata():
    module = _load_module()
    targets = np.asarray([[1.0, 10.0], [2.0, 11.0], [3.0, 12.0], [4.0, 13.0]], dtype=np.float32)
    valid = np.ones_like(targets, dtype=bool)
    data_only = targets + 1.0
    candidate = targets + 0.25
    strata, layers, cells = module.build_spatial_rows(
        targets=targets,
        valid_mask=valid,
        data_only_prediction=data_only,
        candidate_prediction=candidate,
        layer_index=np.asarray([0, 1, 2, 3]),
        block_row=np.asarray([0, 0, 1, 1]),
        block_col=np.asarray([0, 1, 0, 1]),
        laser_active=np.asarray([0.0, 1.0, 0.0, 1.0]),
    )
    assert {row["stratum_family"] for row in strata} == {"layer_quartile", "laser_state", "spatial_region"}
    assert len(layers) == 8
    assert len(cells) == 8
    assert all(row["candidate_better"] is True for row in strata + layers + cells)


def test_phase190_gate_requires_checkpoint_metric_reconstruction_match():
    module = _load_module()
    strata = [
        {"target": target, "stratum_family": family}
        for target in ("tam_s", "scr_C_per_s")
        for family in ("layer_quartile", "laser_state", "spatial_region")
    ]
    layers = [{"target": target, "candidate_better": True} for target in ("tam_s", "scr_C_per_s")]
    cells = [{"target": target, "candidate_better": True} for target in ("tam_s", "scr_C_per_s")]
    ready = module.build_gate(
        phase189=_phase189(),
        verification_rows=_verification_rows(matches=True),
        stratum_rows=strata,
        layer_rows=layers,
        cell_rows=cells,
    )
    blocked = module.build_gate(
        phase189=_phase189(),
        verification_rows=_verification_rows(matches=False),
        stratum_rows=strata,
        layer_rows=layers,
        cell_rows=cells,
    )
    assert ready["phase191_external_confirmation_design_allowed"] is True
    assert ready["model_training_allowed"] is False
    assert ready["spatial_error_stratification_descriptive_only"] is True
    assert blocked["phase191_external_confirmation_design_allowed"] is False
    assert "checkpoint_prediction_rmse_mismatch" in blocked["blocking_audits"]
