from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase198_amb2022_03_baseline_contract.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase198_baseline_contract", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase197(*, ready: bool = True) -> dict[str, object]:
    return {
        "gate": {
            "status": (
                "phase197_calibration_table_design_ready_phase198_baseline_contract_design"
                if ready
                else "phase197_calibration_table_design_incomplete_or_leaky"
            ),
            "phase198_baseline_contract_design_allowed": ready,
            "calibration_fitting_allowed": False,
            "model_training_allowed": False,
        }
    }


def test_phase198_freezes_all_variants_and_repeatability_boundary():
    module = _load_module()
    payload = module.build_payload(_phase197())
    by_id = {row["variant_id"]: row for row in payload["baseline_contract"]}

    assert payload["gate"]["phase199_fixed_baseline_execution_allowed"] is True
    assert payload["gate"]["calibration_fitting_allowed"] is False
    assert set(by_id) == {
        "train_mean_control",
        "process_ridge_control",
        "raw_thermal_ridge",
        "process_plus_raw_thermal_ridge",
        "shuffled_raw_thermal_negative_control",
    }
    assert by_id["process_plus_raw_thermal_ridge"]["ridge_alpha"] == 1.0
    assert "training rows only" in by_id["shuffled_raw_thermal_negative_control"]["training_descriptor_permutation"]
    assert "not a predictive interval" in payload["repeatability_boundary"]


def test_phase198_blocks_reselection_or_changed_ridge_alpha():
    module = _load_module()
    rows = module.build_variant_rows()
    rows[1] = deepcopy(rows[1])
    rows[1]["ridge_alpha"] = 0.5
    rows[2] = deepcopy(rows[2])
    rows[2]["selection_policy"] = "Choose the best held-out model."
    gate = module.build_gate(_phase197(), rows)

    assert gate["phase199_fixed_baseline_execution_allowed"] is False
    assert "ridge_alpha_not_fixed" in gate["blocking_audits"]
    assert "post_holdout_selection_not_blocked" in gate["blocking_audits"]
