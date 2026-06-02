from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_probe_module():
    module_path = Path("scripts/server/phase46_bayesian_inverse_closure_probe.py")
    module_spec = importlib.util.spec_from_file_location("phase46_probe", module_path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def test_phase46_synthetic_probe_recovers_source_parameters(tmp_path: Path):
    probe = _load_probe_module()
    output = tmp_path / "synthetic_summary.json"

    status = probe.main(
        [
            "--mode",
            "synthetic",
            "--synthetic-grid",
            "12",
            "--synthetic-frames",
            "5",
            "--synthetic-noise-std",
            "4.0",
            "--initial-size",
            "48",
            "--acquisition-size",
            "72",
            "--repeats",
            "2",
            "--json-output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["decision"]["source_recovery_ok"] is True
    assert "random" in payload["summary"]
    assert "uncertainty_source" in payload["summary"]
    assert payload["summary"]["uncertainty_source"]["source_recovery_pass_rate"] >= 0.5
    assert any(
        run["parameter_recovery"]["source_parameter_ci90_coverage"] == 1.0
        for run in payload["runs"]
        if run["strategy"] == "uncertainty_source"
    )


def test_phase47_synthetic_physics_attention_feature_mode_runs(tmp_path: Path):
    probe = _load_probe_module()
    output = tmp_path / "synthetic_attention_summary.json"

    status = probe.main(
        [
            "--mode",
            "synthetic",
            "--feature-mode",
            "physics_attention",
            "--synthetic-grid",
            "10",
            "--synthetic-frames",
            "4",
            "--synthetic-noise-std",
            "4.0",
            "--initial-size",
            "36",
            "--acquisition-size",
            "48",
            "--repeats",
            "1",
            "--json-output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["label"] == "synthetic_heat_source_physics_guided_attention"
    assert "attn_source_core_amp" in payload["feature_names"]
    assert "attn_source_tail_amp" in payload["feature_names"]
    assert "random" in payload["summary"]
    assert "uncertainty_source" in payload["summary"]


def test_phase48_region_policy_and_conformal_calibration_runs(tmp_path: Path):
    probe = _load_probe_module()
    output = tmp_path / "synthetic_region_summary.json"

    status = probe.main(
        [
            "--mode",
            "synthetic",
            "--synthetic-grid",
            "10",
            "--synthetic-frames",
            "4",
            "--synthetic-noise-std",
            "4.0",
            "--strategy",
            "region_quota_uncertainty",
            "--strategy",
            "validation_selected_region_policy",
            "--active-strategy",
            "validation_selected_region_policy",
            "--calibration-mode",
            "conformal90",
            "--require-region-preservation",
            "--initial-size",
            "36",
            "--acquisition-size",
            "48",
            "--repeats",
            "1",
            "--json-output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["active_strategy"] == "validation_selected_region_policy"
    assert payload["calibration_mode"] == "conformal90"
    assert "region_quota_uncertainty" in payload["summary"]
    selected_runs = [
        run
        for run in payload["runs"]
        if run["strategy"] == "validation_selected_region_policy"
    ]
    assert selected_runs
    assert selected_runs[0]["selected_policy"] in {
        "uncertainty_source",
        "region_quota_uncertainty",
        "pareto_source_gradient",
    }
    assert selected_runs[0]["calibration"]["scale"] >= 1.0
    assert payload["decision"]["active_strategy"] == "validation_selected_region_policy"


def test_phase49_synthetic_heat_kernel_feature_mode_runs(tmp_path: Path):
    probe = _load_probe_module()
    output = tmp_path / "synthetic_heat_kernel_summary.json"

    status = probe.main(
        [
            "--mode",
            "synthetic",
            "--feature-mode",
            "heat_kernel",
            "--synthetic-grid",
            "10",
            "--synthetic-frames",
            "4",
            "--synthetic-noise-std",
            "4.0",
            "--initial-size",
            "36",
            "--acquisition-size",
            "48",
            "--repeats",
            "1",
            "--json-output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["label"] == "synthetic_heat_source_heat_kernel"
    assert "heat_kernel_d0.06_tau0.15" in payload["feature_names"]
    assert "source_hot_x_gradient" in payload["feature_names"]
    assert "random" in payload["summary"]
    assert "uncertainty_source" in payload["summary"]


def test_phase46_table_probe_reports_sparse_sampling_metrics(tmp_path: Path):
    probe = _load_probe_module()
    table = tmp_path / "thermal.csv"
    rows = [
        "x,y,t,temperature_C,frame_index,row_index,col_index",
    ]
    for index in range(30):
        frame = index // 10
        row = (index // 5) % 2
        col = index % 5
        x = float(col)
        y = float(row)
        t = float(frame)
        temp = 1000.0 + 20.0 * x - 8.0 * y + 15.0 * t
        rows.append(f"{x},{y},{t},{temp},{frame},{row},{col}")
    table.write_text("\n".join(rows) + "\n", encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "splits": {
                    "train": list(range(0, 18)),
                    "val": list(range(18, 24)),
                    "test": list(range(24, 30)),
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "table_summary.json"

    status = probe.main(
        [
            "--mode",
            "table",
            "--table",
            str(table),
            "--target",
            "temperature_C",
            "--split-manifest",
            str(split),
            "--initial-size",
            "6",
            "--acquisition-size",
            "6",
            "--repeats",
            "2",
            "--json-output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["label"] == "thermal"
    assert payload["split_sizes"] == {"train": 18, "val": 6, "test": 6}
    assert "test_rmse_mean" in payload["summary"]["random"]
    assert "test_coverage90_mean" in payload["summary"]["uncertainty_source"]
    assert payload["decision"]["status"] in {"positive", "negative"}
