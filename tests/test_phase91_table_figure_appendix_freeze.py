from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase91_table_figure_appendix_freeze.py")
    spec = importlib.util.spec_from_file_location("phase91_freeze", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


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


def _main_rows() -> list[dict[str, object]]:
    metrics = ["Test RMSE", "Hot q90 RMSE", "Gradient q90 RMSE"]
    rows: list[dict[str, object]] = []
    for dataset in ["broad12", "broad21"]:
        for metric in metrics:
            rows.append(
                {
                    "dataset": dataset,
                    "split": "spot_size",
                    "route": "film/global_standard",
                    "metric": metric,
                    "broad_process_v1_mean": "1.0",
                    "broad_process_v1_std": "0.1",
                    "no_process_mean": "2.0",
                    "no_process_std": "0.2",
                    "best_strong_baseline": "1.5",
                    "best_strong_baseline_method": "mean",
                    "delta_vs_best_strong": "-0.5",
                    "delta_vs_no_process": "-1.0",
                    "n_seeds": "3",
                    "gate": "seed_robust_transfer_positive",
                }
            )
    return rows


def _route_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    splits = ["laser_power", "line", "process", "scan_speed"]
    for dataset in ["broad12", "broad21"]:
        for split in splits:
            fallback = split == "line"
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "classification": "paper_claim_positive" if fallback else "route_guard_positive",
                    "route": "none/none" if fallback else "concat/global_standard",
                    "claim_use": "route guard / no-process fallback evidence"
                    if fallback
                    else "route-guard-only boundary evidence",
                    "metrics_summary": "metric summary",
                    "notes": "guarded wording",
                }
            )
    return rows


def _stress_rows() -> list[dict[str, object]]:
    return [
        {
            "scenario": "stronger_baseline_stress",
            "dataset": "broad12",
            "split": "spot_size",
            "metric": "Test RMSE",
            "status": "pass",
            "candidate": "1.0",
            "comparator": "mean: 1.5",
            "delta_vs_comparator": "-0.5",
            "selected_variant": "",
            "manuscript_use": "supports fixed-sampling Phase 55 floor",
            "evidence": "phase58.json",
        },
        {
            "scenario": "alternate_density_stress",
            "dataset": "broad21",
            "split": "spot_size",
            "metric": "Test RMSE",
            "status": "boundary",
            "candidate": "2.0",
            "comparator": "mean: 1.5",
            "delta_vs_comparator": "0.5",
            "selected_variant": "",
            "manuscript_use": "density-sensitive boundary",
            "evidence": "phase58_density.json",
        },
        {
            "scenario": "residual_upper_bound_gate",
            "dataset": "broad21_density",
            "split": "test",
            "metric": "Test RMSE",
            "status": "blocks_model_expansion",
            "candidate": "2.0",
            "comparator": "mean: 1.5",
            "delta_vs_comparator": "0.5",
            "selected_variant": "mean",
            "manuscript_use": "route-boundary evidence",
            "evidence": "phase59.json",
        },
    ]


def _appendix_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(1, 19):
        phase = "75" if index == 15 else str(32 + index)
        rows.append(
            {
                "appendix_id": f"P88-APPX-{index:03d}",
                "phase": phase,
                "branch": f"branch-{index}",
                "status": "negative" if index != 15 else "blocked_by_local_ambench_gate",
                "artifact": f"artifact-{index}",
                "manuscript_use": "appendix diagnostic",
                "reason": "explicit gate result",
            }
        )
    return rows


def _captions_text() -> str:
    return "\n".join(
        [
            "**Table 1. Fixed-sampling `spot_size` transfer under the route-guarded Macro PINN.**",
            "**Table 2. Route-guard boundary classification across process axes.**",
            "**Table 3. Stress tests and residual-boundary checks for the fixed `spot_size` floor.**",
            "**Table S1. Negative diagnostic and route-boundary ledger.**",
            "**Table S2. Gates for future model branches.**",
            "**Figure 1. Seed-stable `spot_size` transfer across broad12 and broad21.**",
        ]
    )


def _paths(tmp_path: Path) -> dict[str, Path]:
    phase60_manifest = {
        "claim_boundary": {
            "main_claim": "fixed-sampling broad12/broad21 spot_size",
            "excluded_claims": ["density-invariant robustness"],
        }
    }
    phase88_gate = {
        "status": "fallback_experimental_claim_complete",
        "experimental_claim_complete": True,
    }
    phase90_gate = {
        "status": "manuscript_v1_core_claims_integrated_venue_unresolved",
        "core_claims_integrated": True,
    }
    phase90_manifest = {"gate": phase90_gate}
    return {
        "phase56_figure_svg": _write_text(tmp_path / "figure.svg", "<svg></svg>"),
        "phase56_figure_png": _write_text(tmp_path / "figure.png", "png"),
        "phase60_main": _write_csv(tmp_path / "main.csv", _main_rows()),
        "phase60_route": _write_csv(tmp_path / "route.csv", _route_rows()),
        "phase60_stress": _write_csv(tmp_path / "stress.csv", _stress_rows()),
        "phase60_manifest": _write_json(tmp_path / "phase60_manifest.json", phase60_manifest),
        "phase61_captions": _write_text(tmp_path / "captions.md", _captions_text()),
        "phase88_appendix": _write_csv(tmp_path / "appendix.csv", _appendix_rows()),
        "phase88_gate": _write_json(tmp_path / "phase88_gate.json", phase88_gate),
        "phase90_manifest": _write_json(tmp_path / "phase90_manifest.json", phase90_manifest),
        "phase90_gate": _write_json(tmp_path / "phase90_gate.json", phase90_gate),
        "phase90_audit": _write_csv(
            tmp_path / "phase90_audit.csv",
            [{"claim_id": "C61-MAIN-001", "integrated_in_v1": "true"}],
        ),
        "phase90_manuscript": _write_text(tmp_path / "phase90_manuscript.md", "# Manuscript"),
    }


def test_phase91_freezes_tables_figures_and_appendix(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    gate = manifest["gate"]
    assert manifest["phase"] == 91
    assert gate["status"] == "table_figure_appendix_frozen_venue_unresolved"
    assert gate["table_figure_appendix_frozen"] is True
    assert gate["venue_alignment_ready"] is False
    assert gate["submission_ready"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert gate["main_rows"] == 6
    assert gate["route_rows"] == 8
    assert gate["appendix_rows"] == 18
    assert gate["figure_caption_rows"] == 6
    assert gate["figure_assets_exist"] is True

    with (tmp_path / manifest["outputs"]["main_table_freeze"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        main_rows = list(csv.DictReader(handle))
    assert {row["freeze_status"] for row in main_rows} == {"frozen_main_claim_row"}
    assert {row["split"] for row in main_rows} == {"spot_size"}


def test_phase91_preserves_route_guards_and_figure_manifest(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    with (tmp_path / manifest["outputs"]["route_guard_table_freeze"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        route_rows = list(csv.DictReader(handle))
    assert any(row["manuscript_role"] == "route_guard_no_process_fallback" for row in route_rows)
    assert any(row["manuscript_role"] == "route_guard_boundary" for row in route_rows)
    assert all(row["freeze_status"] == "frozen_route_guard_row" for row in route_rows)

    with (tmp_path / manifest["outputs"]["table_figure_caption_manifest"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        figure_rows = list(csv.DictReader(handle))
    by_id = {row["item_id"]: row for row in figure_rows}
    assert by_id["P91-FIG-001"]["freeze_status"] == "frozen_existing_asset"
    assert "figure.svg" in by_id["P91-FIG-001"]["source_artifact"]
    assert by_id["P91-TABLE-S001"]["item_type"] == "appendix_table"
    assert "venue" in by_id["P91-FIG-001"]["venue_dependency"]
