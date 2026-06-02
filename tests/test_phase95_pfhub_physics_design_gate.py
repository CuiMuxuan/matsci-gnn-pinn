from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase95_pfhub_physics_design_gate.py")
    spec = importlib.util.spec_from_file_location("phase95_gate", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _candidate_rows() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": "P94-CAND-PFHUB-PINN",
            "candidate_family": "external_physics_benchmark",
            "source_name": "PFHub phase-field benchmark problems",
            "source_url": "https://pages.nist.gov/pfhub/benchmarks/",
            "source_type": "official_benchmark_collection",
            "public_reproducibility": "public_benchmark_specs",
            "registration_status": "synthetic_or_benchmark_registered_by_problem_definition",
            "target_relevance": "supports mechanism testing",
            "model_innovation_fit": "Bayesian/adaptive PINN, Green's-function features",
            "allowed_next_gate": "phase95_local_synthetic_benchmark_design",
            "a100_training_allowed": "false",
            "a100_80gb_request_now": "false",
            "status": "open_for_local_design_gate",
            "priority": "2",
            "stop_condition": "cannot preserve global/hot/gradient metrics on AM-Bench after synthetic success",
            "next_action": "build a local/no-training Phase 95 design gate",
            "evidence": "PFHub public physics benchmark definitions",
        },
        {
            "candidate_id": "P94-CAND-AMBNCH-PAD-REG",
            "candidate_family": "current_ambench_registration_followup",
            "source_name": "AM-Bench pad thermography plus XYPT",
            "source_url": "https://data.nist.gov/od/id/mds2-2716",
            "source_type": "official_dataset_record",
            "public_reproducibility": "strong_existing_manifest",
            "registration_status": "blocked",
            "target_relevance": "current manuscript",
            "model_innovation_fit": "source-path features",
            "allowed_next_gate": "data_registration_evidence_update_only",
            "a100_training_allowed": "false",
            "a100_80gb_request_now": "false",
            "status": "blocked_until_pad_registration_evidence",
            "priority": "1",
            "stop_condition": "no registration",
            "next_action": "find registration",
            "evidence": "fixture",
        },
    ]


def _design_queue_rows() -> list[dict[str, object]]:
    return [
        {
            "queue_id": "P94-DESIGN-001",
            "priority": "P1",
            "candidate_id": "P94-CAND-PFHUB-PINN",
            "design_task": "write candidate_design.json and local/no-training benchmark protocol",
            "minimum_artifact": "docs/results/phase95_* candidate design package",
            "pass_condition": "mechanism has measurable target",
            "stop_condition": "synthetic-only success",
            "allowed_compute": "local CPU/GPU smoke only; no A100 training",
        }
    ]


def _paths(tmp_path: Path, *, phase95_allowed: bool = True) -> dict[str, Path]:
    phase94_gate = {
        "status": "opened_local_design_gate_no_a100",
        "preferred_next_candidate": "P94-CAND-PFHUB-PINN",
        "phase95_local_gate_allowed": phase95_allowed,
        "a100_training_allowed_now": False,
    }
    return {
        "phase94_gate": _write_json(tmp_path / "phase94_gate.json", phase94_gate),
        "phase94_candidate_triage": _write_csv(tmp_path / "candidate_triage.csv", _candidate_rows()),
        "phase94_design_queue": _write_csv(tmp_path / "design_queue.csv", _design_queue_rows()),
    }


def test_phase95_design_gate_allows_only_local_smoke(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    gate = manifest["gate"]
    assert manifest["phase"] == 95
    assert gate["status"] == "local_design_ready_no_a100"
    assert gate["source_candidate"] == "P94-CAND-PFHUB-PINN"
    assert gate["phase96_local_smoke_allowed"] is True
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert gate["eligible_smoke_mechanisms"] == 2

    design = json.loads((tmp_path / manifest["outputs"]["candidate_design"]).read_text(encoding="utf-8"))
    assert design["candidate_id"] == "phase95_pfhub_local_physics_v1"
    assert "not AM-Bench performance evidence" in design["not_a_claim"]
    assert "A100 training is not allowed" in design["a100_policy"]

    with (tmp_path / manifest["outputs"]["metric_contract"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        metric_rows = list(csv.DictReader(handle))
    assert any(row["metric"] == "global_rmse" for row in metric_rows)
    assert any("global collapse" in row["regression_guard"] for row in metric_rows)


def test_phase95_blocks_if_phase94_does_not_allow_local_gate(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        tmp_path, tmp_path / "out", paths=_paths(tmp_path, phase95_allowed=False)
    )

    gate = manifest["gate"]
    assert gate["status"] == "blocked_design_incomplete"
    assert gate["phase96_local_smoke_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
