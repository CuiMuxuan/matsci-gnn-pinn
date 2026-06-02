from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase60_manuscript_evidence_package.py")
    spec = importlib.util.spec_from_file_location("phase60_package", script)
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _metric(candidate: float, baseline: float, no_process: float, beats: bool = True) -> dict:
    return {
        "candidate": candidate,
        "best_strong_baseline": baseline,
        "best_strong_baseline_method": "mean",
        "beats_best_strong_baseline": beats,
        "no_process": no_process,
        "beats_no_process": candidate < no_process,
        "delta_vs_best_strong": candidate - baseline,
        "delta_vs_no_process": candidate - no_process,
    }


def _seed_summary(label: str, pass_gate: bool) -> dict:
    return {
        "phase": 55,
        "split": "spot_size",
        "required_metrics": ["rmse", "hot_q90_rmse", "gradient_q90_rmse"],
        "datasets": [
            {
                "label": label,
                "aggregate_gate": {
                    "pass": pass_gate,
                    "metrics": {
                        "rmse": _metric(140.0, 150.0, 240.0, pass_gate),
                        "hot_q90_rmse": _metric(160.0, 250.0, 420.0, pass_gate),
                        "gradient_q90_rmse": _metric(170.0, 230.0, 380.0, pass_gate),
                    },
                },
            }
        ],
    }


def _paths(tmp_path: Path) -> dict[str, Path]:
    main = _write_csv(
        tmp_path / "phase56/main.csv",
        [
            {
                "dataset": "broad12",
                "split": "spot_size",
                "route": "film/global_standard",
                "metric": "Test RMSE",
                "broad_process_v1_mean": 136.0,
                "broad_process_v1_std": 1.0,
                "no_process_mean": 238.0,
                "no_process_std": 2.0,
                "best_strong_baseline": 151.0,
                "best_strong_baseline_method": "mean",
                "delta_vs_best_strong": -15.0,
                "delta_vs_no_process": -102.0,
                "n_seeds": 3,
                "gate": "seed_robust_transfer_positive",
            }
        ],
    )
    route = _write_csv(
        tmp_path / "phase56/route.csv",
        [
            {
                "dataset": "broad12",
                "split": "laser_power",
                "classification": "route_guard_positive",
                "route": "concat/global_standard",
                "claim_use": "route-guard-only boundary evidence",
                "metrics_summary": "candidate trails mean",
                "notes": "strong baseline remains better",
            }
        ],
    )
    appendix = _write_csv(
        tmp_path / "phase56/appendix.csv",
        [
            {
                "phase": "33",
                "branch": "Fourier",
                "target": "broad12",
                "result": "Negative",
                "paper_use": "Appendix diagnostic",
                "evidence": "docs/results/x.md",
            }
        ],
    )
    contract = _write_json(
        tmp_path / "phase57/contract.json",
        {
            "required_metrics": ["rmse", "hot_q90_rmse", "gradient_q90_rmse"],
            "current_transfer_gate": {"paper_claim_status": "seed_robust_transfer_positive"},
            "frozen_floor": {
                "required_datasets": ["broad12", "broad21"],
                "required_metrics": ["rmse", "hot_q90_rmse", "gradient_q90_rmse"],
            },
        },
    )
    ledger = _write_csv(
        tmp_path / "phase57/ledger.csv",
        [
            {
                "kind": "process_axis",
                "dataset": "broad12",
                "split_or_branch": "spot_size",
                "status": "paper_positive_seed_robust",
                "process_conditioning_claim": "yes",
                "route": "film/global_standard",
                "metrics_gate": "pass",
                "seed_gate": "pass",
                "paper_placement": "main_table",
                "current_evidence": "phase55",
                "next_action": "draft",
                "notes": "",
            }
        ],
    )
    stronger = _write_json(
        tmp_path / "phase58/stronger.json",
        {
            "phase": 58,
            "required_metrics": ["rmse", "hot_q90_rmse", "gradient_q90_rmse"],
            "datasets": [
                {
                    "label": "broad12",
                    "metrics": {
                        "rmse": {
                            "frozen_broad_process_v1": 136.0,
                            "best_baseline_after_stress": {"method": "mean", "value": 151.0},
                            "delta_vs_best_after_stress": -15.0,
                            "frozen_beats_best_after_stress": True,
                        },
                        "hot_q90_rmse": {
                            "frozen_broad_process_v1": 162.0,
                            "best_baseline_after_stress": {"method": "mean", "value": 252.0},
                            "delta_vs_best_after_stress": -90.0,
                            "frozen_beats_best_after_stress": True,
                        },
                        "gradient_q90_rmse": {
                            "frozen_broad_process_v1": 165.0,
                            "best_baseline_after_stress": {"method": "mean", "value": 233.0},
                            "delta_vs_best_after_stress": -68.0,
                            "frozen_beats_best_after_stress": True,
                        },
                    },
                }
            ],
        },
    )
    anatomy = _write_json(
        tmp_path / "phase59/anatomy.json",
        {
            "phase": 59,
            "analysis_split": "test",
            "candidate": "broad_process_v1",
            "reference": "mean",
            "worst_candidate_vs_reference": [
                {
                    "field": "line_id",
                    "value": "Line_1_1_1",
                    "n": 10,
                    "delta_candidate_minus_reference_rmse": 20.0,
                    "metrics": {
                        "broad_process_v1": {"rmse": 160.0},
                        "mean": {"rmse": 140.0},
                    },
                }
            ],
        },
    )
    upper = _write_json(
        tmp_path / "phase59/upper.json",
        {
            "phase": 59,
            "reference": "mean",
            "uses_test_for_selection": False,
            "selected_variant": {"name": "blend:broad_process_v1->mean:alpha=1"},
            "decision": {
                "candidate_rmse": 153.0,
                "reference_rmse": 139.0,
                "selected_variant": "blend:broad_process_v1->mean:alpha=1",
                "selected_beats_reference_rmse": False,
                "interpretation": "validation-visible correction does not beat the reference",
            },
        },
    )
    return {
        "phase56_main": main,
        "phase56_route": route,
        "phase56_appendix": appendix,
        "phase57_contract": contract,
        "phase57_ledger": ledger,
        "phase58_stronger": stronger,
        "phase58_density": _write_json(tmp_path / "phase58/density.json", _seed_summary("broad21", False)),
        "phase58_panel": _write_json(tmp_path / "phase58/panel.json", _seed_summary("broad15", True)),
        "phase59_anatomy": anatomy,
        "phase59_upper": upper,
    }


def test_phase60_builds_manuscript_package_and_blocks_density_driven_expansion(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    assert manifest["counts"]["main_rows"] == 1
    assert manifest["counts"]["appendix_rows"] == 3
    assert manifest["counts"]["stress_status_counts"]["blocks_model_expansion"] == 1
    assert manifest["model_expansion_gate"]["decision"] == "block_density_failure_driven_model_expansion"
    assert manifest["model_expansion_gate"]["uses_test_for_selection"] is False
    outputs = manifest["outputs"]
    assert (tmp_path / outputs["main_table"]).exists()
    assert (tmp_path / outputs["route_guard_table"]).exists()
    assert (tmp_path / outputs["stress_boundary_table"]).exists()
    assert (tmp_path / outputs["appendix_table"]).exists()
    assert (tmp_path / outputs["next_branch_gate_table"]).exists()
    markdown = (tmp_path / outputs["markdown"]).read_text(encoding="utf-8")
    assert "Main Claim Floor" in markdown
    assert "density-sensitive boundary" in markdown


def test_phase60_next_gate_pauses_candidate_a_when_phase59_upper_bound_is_negative(tmp_path: Path):
    module = _load_module()
    paths = _paths(tmp_path)
    contract = json.loads(paths["phase57_contract"].read_text(encoding="utf-8"))
    upper = json.loads(paths["phase59_upper"].read_text(encoding="utf-8"))

    rows = module.build_next_gate_rows(contract, upper)

    by_branch = {row["branch"]: row for row in rows}
    assert by_branch["Candidate A: physically constrained spot-size conditioning"]["status"] == "paused"
    assert by_branch["Candidate B: validation-auditable route policy"]["status"] == (
        "blocked by Phase 59 density gate"
    )
