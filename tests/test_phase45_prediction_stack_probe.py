from __future__ import annotations

import csv
import json
from pathlib import Path


def _write_predictions(path: Path, predictions: list[float]) -> None:
    y_true = [0.0, 1.0, 2.0, 10.0]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_index",
                "split",
                "sample_id",
                "method",
                "x",
                "y",
                "t",
                "T",
                "prediction",
                "error",
                "abs_error",
            ],
        )
        writer.writeheader()
        for index, prediction in enumerate(predictions):
            split = "train" if index < 2 else "val" if index == 2 else "test"
            error = prediction - y_true[index]
            writer.writerow(
                {
                    "row_index": index,
                    "split": split,
                    "sample_id": "toy",
                    "method": path.stem,
                    "x": float(index),
                    "y": 0.0,
                    "t": 0.0,
                    "T": y_true[index],
                    "prediction": prediction,
                    "error": error,
                    "abs_error": abs(error),
                }
            )


def test_phase45_prediction_stack_probe_fits_without_test_leakage(tmp_path: Path):
    spec = __import__(
        "importlib.util",
        fromlist=["spec_from_file_location", "module_from_spec"],
    )
    module_path = Path("scripts/server/phase45_prediction_stack_probe.py")
    module_spec = spec.spec_from_file_location("phase45_probe", module_path)
    assert module_spec is not None and module_spec.loader is not None
    probe = spec.module_from_spec(module_spec)
    module_spec.loader.exec_module(probe)

    table = tmp_path / "field.csv"
    table.write_text(
        "x,y,t,T\n"
        "0,0,0,0\n"
        "1,0,0,1\n"
        "2,0,0,2\n"
        "3,0,0,10\n",
        encoding="utf-8",
    )
    split = tmp_path / "split.json"
    split.write_text(
        '{"splits":{"train":[0,1],"val":[2],"test":[3]}}',
        encoding="utf-8",
    )
    expert_a = tmp_path / "expert_a_predictions.csv"
    expert_b = tmp_path / "expert_b_predictions.csv"
    _write_predictions(expert_a, [0.0, 1.0, 2.0, 0.0])
    _write_predictions(expert_b, [10.0, 10.0, 10.0, 10.0])
    output = tmp_path / "summary.json"

    status = probe.main(
        [
            "--table",
            str(table),
            "--target",
            "T",
            "--split-manifest",
            str(split),
            "--prediction",
            str(expert_a),
            "--prediction",
            str(expert_b),
            "--label",
            "expert_a",
            "--label",
            "expert_b",
            "--weight-step",
            "0.5",
            "--hot-quantile",
            "0.5",
            "--json-output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["uses_test_for_selection"] is False
    assert payload["fit_splits"] == ["train", "val"]
    assert payload["weights"] == {"expert_a": 1.0, "expert_b": 0.0}
    assert payload["fit"]["candidate_count"] == 3
    assert payload["stack_metrics"]["test"]["metrics"]["rmse"] == 10.0
    assert "hot_q50" in payload["stack_metrics"]["test"]["region_metrics"]
    assert "expert_a" in payload["expert_metrics"]
