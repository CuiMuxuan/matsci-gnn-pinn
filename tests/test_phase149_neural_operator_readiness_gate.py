from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase149_neural_operator_readiness_gate.py")
    spec = importlib.util.spec_from_file_location("phase149_neural_operator", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_positive_floor(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "floor_id",
                "dataset",
                "split",
                "route",
                "metric",
                "broad_process_v1_mean",
                "broad_process_v1_std",
                "best_strong_baseline",
                "delta_vs_best_strong",
                "n_seeds",
                "claim_anchor",
                "manuscript_use",
                "evidence_source",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "floor_id": "P116-FLOOR-001",
                "dataset": "broad12",
                "split": "spot_size",
                "route": "film/global_standard",
                "metric": "Test RMSE",
                "broad_process_v1_mean": "136.3",
                "broad_process_v1_std": "0.4",
                "best_strong_baseline": "151.8",
                "delta_vs_best_strong": "-15.5",
                "n_seeds": "3",
                "claim_anchor": "C61",
                "manuscript_use": "current_main_text_floor",
                "evidence_source": "phase91.csv",
            }
        )
    return path


def _phase_inputs(tmp_path: Path) -> dict[str, Path]:
    return {
        "phase116_gate": _write_json(
            tmp_path / "phase116_gate.json",
            {
                "status": "phase116_paper_evidence_consolidation_ready",
                "positive_floor_ready": True,
                "phase116_model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase116_positive_floor": _write_positive_floor(tmp_path / "phase116_floor.csv"),
        "phase148_gate": _write_json(
            tmp_path / "phase148_gate.json",
            {
                "status": "phase148_path_contact_graph_audit_closed_no_guarded_graph_gap",
                "phase148_model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase33_fourier_diagnostic": _write_text(
            tmp_path / "phase33.md",
            "Phase 33 Fourier spacetime representation was negative and worsened broad_process_v1.",
        ),
        "phase55_spot_size_seed_validation": _write_text(
            tmp_path / "phase55.md",
            "The spot_size branch is seed-validated across broad12 and broad21.",
        ),
    }


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def test_phase149_closes_operator_training_but_allows_dense_inventory(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        phase_inputs=_phase_inputs(tmp_path),
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 149
    assert gate["status"] == "phase149_neural_operator_readiness_closed_not_ready_for_operator_training"
    assert gate["blocker_rows"] == 5
    assert gate["phase150_dense_tensorization_inventory_allowed"] is True
    assert gate["operator_training_allowed_now"] is False
    assert gate["phase149_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False

    with (tmp_path / "out/phase149_neural_operator_readiness_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert {row["criterion"] for row in rows} >= {
        "dense_operator_tensor_dataset",
        "spectral_representation_prior",
        "training_and_compute_locks",
    }
    assert all(row["blocks_operator_training"] == "true" for row in rows)

    markdown = (tmp_path / "out/phase149_neural_operator_readiness_gate.md").read_text(
        encoding="utf-8"
    )
    assert "Operator training allowed now: `false`" in markdown
    assert "dense tensorization inventory" in markdown
