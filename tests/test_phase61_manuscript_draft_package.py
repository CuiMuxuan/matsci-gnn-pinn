from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase61_manuscript_draft_package.py")
    spec = importlib.util.spec_from_file_location("phase61_package", script)
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


def _main_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, values in {
        "broad12": (136.384782, 162.125337, 165.282182, 151.850578, 252.554440, 233.119660),
        "broad21": (146.002303, 164.313888, 174.735839, 149.185412, 251.976794, 231.072566),
    }.items():
        for metric, candidate, baseline in [
            ("Test RMSE", values[0], values[3]),
            ("Hot q90 RMSE", values[1], values[4]),
            ("Gradient q90 RMSE", values[2], values[5]),
        ]:
            rows.append(
                {
                    "dataset": dataset,
                    "split": "spot_size",
                    "route": "film/global_standard",
                    "metric": metric,
                    "broad_process_v1_mean": candidate,
                    "broad_process_v1_std": 1.0,
                    "no_process_mean": 220.0,
                    "no_process_std": 1.0,
                    "best_strong_baseline": baseline,
                    "best_strong_baseline_method": "mean",
                    "delta_vs_best_strong": candidate - baseline,
                    "delta_vs_no_process": candidate - 220.0,
                    "n_seeds": 3,
                    "gate": "seed_robust_transfer_positive",
                }
            )
    return rows


def _paths(tmp_path: Path) -> dict[str, Path]:
    package = tmp_path / "phase60"
    main = _write_csv(package / "main.csv", _main_rows())
    route = _write_csv(
        package / "route.csv",
        [
            {
                "dataset": "broad12",
                "split": "laser_power",
                "classification": "route_guard_positive",
                "route": "concat/global_standard",
                "claim_use": "route-guard-only boundary evidence",
                "metrics_summary": "candidate trails mean",
                "notes": "strong baseline remains better",
            },
            {
                "dataset": "broad21",
                "split": "line",
                "classification": "paper_claim_positive",
                "route": "none/none",
                "claim_use": "route guard / no-process fallback evidence",
                "metrics_summary": "fallback positive",
                "notes": "do not claim process conditioning",
            },
        ],
    )
    stress = _write_csv(
        package / "stress.csv",
        [
            {
                "scenario": "stronger_baseline_stress",
                "dataset": "broad12",
                "split": "spot_size",
                "metric": "Test RMSE",
                "status": "pass",
                "candidate": 136.0,
                "comparator": "mean: 151.0",
                "delta_vs_comparator": -15.0,
                "selected_variant": "",
                "manuscript_use": "supports fixed-sampling Phase 55 floor",
                "evidence": "phase58_stronger.json",
            },
            {
                "scenario": "auxiliary_process_panel",
                "dataset": "broad15",
                "split": "spot_size",
                "metric": "Test RMSE",
                "status": "pass",
                "candidate": 138.0,
                "comparator": "mean: 151.0",
                "delta_vs_comparator": -13.0,
                "selected_variant": "",
                "manuscript_use": "auxiliary process-panel support",
                "evidence": "phase58_panel.json",
            },
            {
                "scenario": "alternate_density_stress",
                "dataset": "broad21",
                "split": "spot_size",
                "metric": "Test RMSE",
                "status": "boundary",
                "candidate": 153.0,
                "comparator": "mean: 139.0",
                "delta_vs_comparator": 14.0,
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
                "candidate": 153.0,
                "comparator": "mean: 139.0",
                "delta_vs_comparator": 14.0,
                "selected_variant": "blend:broad_process_v1->mean:alpha=1",
                "manuscript_use": "route-boundary evidence",
                "evidence": "phase59_upper.json",
            },
        ],
    )
    appendix = _write_csv(
        package / "appendix.csv",
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
    next_gate = _write_csv(
        package / "next_gate.csv",
        [
            {
                "branch": "Candidate A: physically constrained spot-size conditioning",
                "status": "paused",
                "entry_condition": "Requires a new validation-visible signal.",
                "focused_validation": "broad12/broad21",
                "seed_expansion_gate": "non-worse than floor",
                "manuscript_rule": "promote only if positive",
            },
            {
                "branch": "Candidate B: validation-auditable route policy",
                "status": "blocked by Phase 59 density gate",
                "entry_condition": "Needs validation-visible selection.",
                "focused_validation": "policy",
                "seed_expansion_gate": "non-worse than floor",
                "manuscript_rule": "boundary otherwise",
            },
        ],
    )
    phase60_manifest = _write_json(
        package / "manifest.json",
        {
            "phase": 60,
            "claim_boundary": {
                "main_claim": "fixed-sampling broad12/broad21 spot_size",
                "excluded_claims": ["density-invariant robustness"],
            },
            "model_expansion_gate": {
                "decision": "block_density_failure_driven_model_expansion",
                "selected_variant": "blend:broad_process_v1->mean:alpha=1",
                "uses_test_for_selection": False,
            },
        },
    )
    phase60_markdown = tmp_path / "phase60.md"
    phase60_markdown.write_text("phase60", encoding="utf-8")
    return {
        "main": main,
        "route": route,
        "stress": stress,
        "appendix": appendix,
        "next_gate": next_gate,
        "phase60_markdown": phase60_markdown,
        "phase60_manifest": phase60_manifest,
    }


def test_phase61_builds_draft_package_with_claim_crosswalk_and_gaps(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    assert manifest["phase"] == 61
    assert manifest["counts"]["claim_anchor_rows"] == 11
    assert manifest["counts"]["literature_gap_rows"] == 3
    assert manifest["model_expansion_gate"]["decision"] == "block_density_failure_driven_model_expansion"
    outputs = manifest["outputs"]
    for path in outputs.values():
        assert (tmp_path / path).exists()
    results = (tmp_path / outputs["results_draft"]).read_text(encoding="utf-8")
    methods = (tmp_path / outputs["methods_draft"]).read_text(encoding="utf-8")
    captions = (tmp_path / outputs["captions"]).read_text(encoding="utf-8")
    assert "C61-MAIN-001" in results
    assert "density-invariant robustness" in results
    assert "no-test-leakage gate" in methods
    assert "Table 1" in captions
    with (tmp_path / outputs["claim_evidence_crosswalk"]).open(encoding="utf-8", newline="") as handle:
        crosswalk = list(csv.DictReader(handle))
    assert {row["claim_anchor_id"] for row in crosswalk} >= {
        "C61-MAIN-001",
        "C61-BOUNDARY-002",
        "C61-METHOD-001",
    }
    with (tmp_path / outputs["literature_gap_register"]).open(encoding="utf-8", newline="") as handle:
        gaps = list(csv.DictReader(handle))
    assert all(row["blocks_current_phase61_draft"] == "no" for row in gaps)


def test_phase61_crosswalk_keeps_boundary_claim_strengths(tmp_path: Path):
    module = _load_module()
    paths = _paths(tmp_path)
    manifest = json.loads(paths["phase60_manifest"].read_text(encoding="utf-8"))

    rows = module.build_claim_crosswalk(
        paths,
        manifest,
        module._read_csv(paths["main"]),
        module._read_csv(paths["route"]),
        module._read_csv(paths["stress"]),
        module._read_csv(paths["appendix"]),
        module._read_csv(paths["next_gate"]),
        tmp_path,
    )

    by_id = {row["claim_anchor_id"]: row for row in rows}
    assert by_id["C61-BOUNDARY-001"]["allowed_claim_strength"] == "strong"
    assert "must not claim density-invariant robustness" in by_id["C61-BOUNDARY-001"]["draft_sentence"]
    assert by_id["C61-STRESS-002"]["allowed_claim_strength"] == "cautious"
