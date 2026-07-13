from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase187_amb2022_01_candidate_model_design.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase187_candidate_model_design", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_design_admits_only_bounded_runs_after_phase186():
    module = _load_module()
    phase186 = {
        "gate": {
            "status": "phase186_feature_ablation_ready_phase187_candidate_model_design",
            "phase187_candidate_model_design_allowed": True,
        }
    }
    gate = module.build_gate(phase186)
    assert gate["phase188_bounded_gpu_training_allowed"] is True
    assert gate["compute_budget"]["max_total_runs"] == 6
    assert gate["compute_budget"]["hyperparameter_search_allowed"] is False


def test_candidate_contract_has_data_only_and_physics_variants():
    module = _load_module()
    rows = module.build_model_rows()
    by_id = {row["variant_id"]: row for row in rows}
    assert by_id["small_data_only_mlp"]["uses_heat_kernel"] is False
    assert by_id["physics_regularized_history_mlp"]["uses_tam_monotonic_prior"] is True
