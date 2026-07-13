from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase189_amb2022_01_replicate_stability_review.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase189_replicate_stability", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase188() -> dict[str, object]:
    return {
        "gate": {
            "status": "phase188_bounded_gpu_training_complete_phase189_replicate_review",
            "training_complete": True,
            "phase189_replicate_review_allowed": True,
            "model_training_allowed": False,
        }
    }


def _metric_rows() -> list[dict[str, object]]:
    rows = []
    for target_index, target in enumerate(("tam_s", "scr_C_per_s")):
        for split_index, split in enumerate(("train", "val", "test")):
            for variant, offset in (("small_data_only_mlp", 1.0), ("physics_regularized_history_mlp", 0.0)):
                for seed_index, seed in enumerate((1871, 1872, 1873)):
                    rows.append(
                        {
                            "target": target,
                            "split": split,
                            "variant_id": variant,
                            "seed": seed,
                            "rmse": 10.0 + target_index + split_index + offset + 0.01 * seed_index,
                        }
                    )
    return rows


def _audits() -> list[dict[str, object]]:
    rows = []
    for seed in (1871, 1872, 1873):
        rows.append(
            {
                "variant_id": "physics_regularized_history_mlp",
                "seed": seed,
                "monotonic_violation_fraction_test": 0.1,
            }
        )
    return rows


def _phase186_rows() -> list[dict[str, object]]:
    rows = []
    for target_index, target in enumerate(("tam_s", "scr_C_per_s")):
        for split_index, split in enumerate(("val", "test")):
            rows.append(
                {
                    "target": target,
                    "split": split,
                    "variant_id": "heat_kernel_history_ridge",
                    "rmse": 15.0 + target_index + split_index,
                }
            )
    return rows


def test_phase189_admits_complete_directional_replicates_without_component_overclaim():
    module = _load_module()
    review = module.build_review(_phase188(), _metric_rows(), _audits(), _phase186_rows())
    gate = review["gate"]
    assert gate["phase190_spatial_failure_analysis_allowed"] is True
    assert gate["model_training_allowed"] is False
    assert gate["post_b8_model_reselection_allowed"] is False
    assert gate["compound_candidate_effect_claim_allowed"] is True
    assert gate["isolated_heat_kernel_effect_claim_allowed"] is False
    assert gate["isolated_monotonic_prior_effect_claim_allowed"] is False
    assert len(review["summary"]) == 4
    assert all(gate["fixed_seed_directional_passes"].values())


def test_phase189_blocks_a_test_seed_that_reverses_candidate_directionality():
    module = _load_module()
    rows = _metric_rows()
    for row in rows:
        if (
            row["target"] == "tam_s"
            and row["split"] == "test"
            and row["variant_id"] == "physics_regularized_history_mlp"
            and row["seed"] == 1873
        ):
            row["rmse"] = 99.0
            break
    gate = module.build_gate(_phase188(), rows, _audits(), _phase186_rows())
    assert gate["phase190_spatial_failure_analysis_allowed"] is False
    assert "tam_s_test_candidate_not_directionally_replicated" in gate["blocking_audits"]


def test_phase189_blocks_duplicate_metric_seed_even_when_row_count_is_three():
    module = _load_module()
    rows = _metric_rows()
    for row in rows:
        if (
            row["target"] == "scr_C_per_s"
            and row["split"] == "val"
            and row["variant_id"] == "small_data_only_mlp"
            and row["seed"] == 1873
        ):
            row["seed"] = 1871
            break
    gate = module.build_gate(_phase188(), rows, _audits(), _phase186_rows())
    assert gate["phase190_spatial_failure_analysis_allowed"] is False
    assert "scr_C_per_s_val_small_data_only_mlp_seed_contract" in gate["blocking_audits"]
