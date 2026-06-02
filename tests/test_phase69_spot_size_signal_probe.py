from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase69_spot_size_signal_probe.py")
    spec = importlib.util.spec_from_file_location("phase69_spot_signal", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _metric(candidate: float, baseline: float, no_process: float, beats_best: bool = True) -> dict:
    return {
        "candidate": candidate,
        "best_strong_baseline": baseline,
        "best_strong_baseline_method": "mean",
        "no_process": no_process,
        "delta_vs_best_strong": candidate - baseline,
        "delta_vs_no_process": candidate - no_process,
        "beats_best_strong_baseline": beats_best,
        "beats_no_process": candidate < no_process,
    }


def _summary(labels: list[str], *, fail_broad21: bool = False) -> dict:
    datasets = []
    for label in labels:
        fail = fail_broad21 and label == "broad21"
        datasets.append(
            {
                "label": label,
                "split": "spot_size",
                "status": "seed_unstable_or_negative" if fail else "seed_robust_transfer_positive",
                "aggregate_gate": {
                    "pass": not fail,
                    "metrics": {
                        "rmse": _metric(153.0 if fail else 136.0, 139.0 if fail else 151.0, 226.0, not fail),
                        "hot_q90_rmse": _metric(270.0 if fail else 162.0, 253.0, 421.0, not fail),
                        "gradient_q90_rmse": _metric(250.0 if fail else 165.0, 231.0, 374.0, not fail),
                    },
                },
                "aggregates": {
                    "broad_process_v1": {
                        "n": 1 if fail else 3,
                        "rmse": {"mean": 153.0 if fail else 136.0, "pstdev": 0.0},
                        "hot_q90_rmse": {"mean": 270.0 if fail else 162.0, "pstdev": 0.0},
                        "gradient_q90_rmse": {"mean": 250.0 if fail else 165.0, "pstdev": 0.0},
                    }
                },
            }
        )
    return {"phase": 55, "required_metrics": ["rmse", "hot_q90_rmse", "gradient_q90_rmse"], "datasets": datasets}


def _paths(tmp_path: Path, *, fail_density_broad21: bool = True, upper_blocks: bool = True) -> dict[str, Path]:
    fixed = _write_json(tmp_path / "phase55_fixed.json", _summary(["broad12", "broad21"]))
    density = _write_json(
        tmp_path / "phase58_density.json",
        _summary(["broad12", "broad21"], fail_broad21=fail_density_broad21),
    )
    panel = _write_json(tmp_path / "phase58_panel.json", _summary(["broad15"]))
    upper = _write_json(
        tmp_path / "phase59_upper.json",
        {
            "uses_test_for_selection": False,
            "selected_variant": {"name": "blend:broad_process_v1->mean:alpha=1" if upper_blocks else "identity"},
            "decision": {
                "selected_variant": "blend:broad_process_v1->mean:alpha=1" if upper_blocks else "identity",
                "selected_beats_reference_rmse": not upper_blocks,
            },
        },
    )
    phase68 = _write_json(
        tmp_path / "phase68_manifest.json",
        {
            "phase": 68,
            "current_decision": {
                "trainable_model_opened": False,
                "next_step": "non_training_signal_probe_or_manuscript_v0_claim_audit",
            },
        },
    )
    return {
        "phase55_fixed": fixed,
        "phase58_density": density,
        "phase58_panel": panel,
        "phase59_upper": upper,
        "phase68_manifest": phase68,
    }


def test_phase69_keeps_candidate_a_paused_when_density_boundary_and_upper_bound_block(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    gate = manifest["candidate_a_gate"]
    assert manifest["phase"] == 69
    assert manifest["counts"]["signal_rows"] == 15
    assert gate["status"] == "paused_no_training_signal"
    assert gate["open_for_seed7_a100_gate"] is False
    assert gate["alternate_density_broad21_pass"] is False
    assert gate["broad21_density_boundary"] is True
    assert gate["phase59_upper_blocks_density_driven_expansion"] is True
    assert gate["a100_80gb_request_now"] is False
    with (tmp_path / manifest["outputs"]["signal_table"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert any(
        row["evidence_source"] == "alternate_density_phase58"
        and row["dataset"] == "broad21"
        and row["status"] == "boundary"
        for row in rows
    )


def test_phase69_opens_seed7_gate_only_when_density_and_upper_bound_pass(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        tmp_path,
        tmp_path / "out",
        paths=_paths(tmp_path, fail_density_broad21=False, upper_blocks=False),
    )

    gate = manifest["candidate_a_gate"]
    assert gate["status"] == "opened_for_seed7_a100_gate"
    assert gate["open_for_seed7_a100_gate"] is True
    assert gate["alternate_density_broad21_pass"] is True
    assert gate["phase59_upper_blocks_density_driven_expansion"] is False
