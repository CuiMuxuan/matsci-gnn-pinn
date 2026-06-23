from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase168_hidden_source_closure_redesign_gate.py")
    spec = importlib.util.spec_from_file_location("phase168_redesign_gate", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _paths(tmp_path: Path, *, phase167_status: str | None = None) -> dict[str, Path]:
    phase167_gate = {
        "status": phase167_status or "phase167_low_budget_pinn_smoke_closed_no_stable_model_gain",
        "selected_variant": "uniform_grid_pinn",
        "best_control_variant": "uniform_grid_pinn",
        "blocking_audits": [
            "validation_selected_control_variant",
            "validation_gain_vs_best_control",
        ],
        "phase168_focused_review_allowed": False,
        "phase167_model_claim_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }
    summary_rows = [
        {
            "variant_id": "uniform_grid_pinn",
            "split": "val",
            "selection_score_mean": 0.044,
        },
        {
            "variant_id": "failure_informed_hot_gradient_pinn",
            "split": "val",
            "selection_score_mean": 0.098,
        },
        {
            "variant_id": "wrong_prior_failure_sampler_control",
            "split": "val",
            "selection_score_mean": 0.225,
        },
        {
            "variant_id": "data_only_tiny_mlp_no_residual",
            "split": "val",
            "selection_score_mean": 0.096,
        },
    ]
    reference_rows = [
        {
            "reference_id": f"P166-REF-{idx:03d}",
            "doi": f"10.example/{idx}",
        }
        for idx in range(8)
    ]
    return {
        "phase167_gate": _write_json(tmp_path / "p167/gate.json", phase167_gate),
        "phase167_summary_table": _write_csv(tmp_path / "p167/summary.csv", summary_rows),
        "phase166_reference_table": _write_csv(tmp_path / "p166/references.csv", reference_rows),
    }


def test_phase168_builds_redesign_gate_without_training(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert gate["status"] == (
        "phase168_hidden_source_closure_redesign_ready_phase169_identifiability_gate"
    )
    assert gate["selected_next_route"] == "hidden_source_closure_identifiability_gate"
    assert gate["phase169_hidden_source_closure_identifiability_gate_allowed"] is True
    assert gate["phase168_retrain_same_sampler_route_allowed"] is False
    assert gate["phase168_model_mechanism_allowed"] is False
    assert gate["phase168_model_training_allowed"] is False
    assert gate["phase169_training_allowed_now"] is False
    assert gate["bayesian_pinn_training_allowed_now"] is False
    assert gate["adaptive_sampling_training_allowed_now"] is False
    assert gate["gcn_pinn_training_allowed_now"] is False
    assert gate["cnn_operator_training_allowed_now"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["evidence_rows"] >= 4
    assert manifest["counts"]["route_rows"] >= 5
    assert manifest["counts"]["design_rows"] >= 5

    route_table = (tmp_path / "out/phase168_route_redesign_table.csv").read_text(
        encoding="utf-8"
    )
    assert "hidden_source_closure_identifiability_gate" in route_table
    assert "blocked_by_prior_path_graph_guard" in route_table
    assert "blocked_by_prior_dense_baseline_guard" in route_table

    markdown = (
        tmp_path / "out/phase168_hidden_source_closure_redesign_gate.md"
    ).read_text(encoding="utf-8")
    assert "away from sampler retuning" in markdown
    assert "|  |  |" not in markdown


def test_phase168_closes_if_phase167_not_closed_by_uniform(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path, phase167_status="phase167_ready_phase168"),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase168_hidden_source_closure_redesign_incomplete"
    assert "phase167_not_closed_by_uniform_control" in gate["blocking_audits"]
    assert gate["phase169_hidden_source_closure_identifiability_gate_allowed"] is False
    assert gate["phase168_model_training_allowed"] is False
    assert gate["a100_80gb_request_now"] is False


def test_phase168_blocks_any_route_that_opens_training():
    module = _load_module()
    phase167_gate = {
        "status": "phase167_low_budget_pinn_smoke_closed_no_stable_model_gain",
        "selected_variant": "uniform_grid_pinn",
        "phase168_focused_review_allowed": False,
    }
    evidence_rows = [
        {
            "finding": "adaptive_sampler_model_score_worse_than_uniform",
        },
        {
            "finding": "wrong_source_prior_control_failed_strongly",
        },
    ]
    route_rows = module.build_route_rows()
    route_rows[0]["opens_training_now"] = True
    gate = module.build_gate(
        phase167_gate=phase167_gate,
        phase166_reference_rows=[{"reference_id": str(idx)} for idx in range(8)],
        evidence_rows=evidence_rows,
        route_rows=route_rows,
        design_rows=module.build_design_rows(),
    )

    assert gate["status"] == "phase168_hidden_source_closure_redesign_incomplete"
    assert "route_attempted_training_now" in gate["blocking_audits"]
    assert gate["phase169_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
