from pathlib import Path

import pytest

from gnnpinn.data.loaders import load_field_table
from gnnpinn.eval.baselines import constant_predictions, regression_metric_table
from gnnpinn.eval.field_baseline import main as field_baseline_main
from gnnpinn.eval.metrics import mae, normalized_rmse, relative_l2, rmse


def test_load_field_table_csv(tmp_path: Path):
    path = tmp_path / "field.csv"
    path.write_text(
        "x,y,t,T\n"
        "0,0,0,300\n"
        "1,0,0.5,325\n",
        encoding="utf-8",
    )

    sample = load_field_table(path)

    assert sample.sample_id == "field"
    assert sample.n_points == 2
    assert sample.coordinates == [[0.0, 0.0], [1.0, 0.0]]
    assert sample.time == [0.0, 0.5]
    assert sample.observations["T"] == [300.0, 325.0]


def test_metrics_for_temperature_baseline():
    y_true = [300.0, 310.0, 330.0]
    y_pred = [300.0, 315.0, 320.0]

    assert mae(y_true, y_pred) == pytest.approx(5.0)
    assert rmse(y_true, y_pred) == pytest.approx((125.0 / 3.0) ** 0.5)
    assert normalized_rmse(y_true, y_pred) == pytest.approx(rmse(y_true, y_pred) / 30.0)
    assert relative_l2(y_true, y_pred) > 0


def test_constant_baseline_metric_table():
    y_true = [1.0, 2.0, 3.0]
    y_pred = constant_predictions(y_true, strategy="mean")
    metrics = regression_metric_table(y_true, y_pred)

    assert y_pred == [2.0, 2.0, 2.0]
    assert metrics["mae"] == pytest.approx(2.0 / 3.0)


def test_constant_baseline_metric_table_handles_zero_relative_l2():
    metrics = regression_metric_table([0.0, 0.0], [1.0, 1.0])

    assert str(metrics["relative_l2"]).startswith("undefined:")


def test_field_baseline_cli_writes_json(tmp_path: Path):
    table = tmp_path / "thermal.csv"
    table.write_text(
        "x,y,t,T\n"
        "0,0,0,300\n"
        "1,0,0.5,330\n",
        encoding="utf-8",
    )
    output = tmp_path / "baseline.json"

    status = field_baseline_main(
        [
            "--table",
            str(table),
            "--target",
            "T",
            "--strategy",
            "first",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert '"baseline": "constant:first"' in output.read_text(encoding="utf-8")


def test_field_baseline_cli_with_split_manifest(tmp_path: Path):
    table = tmp_path / "thermal.csv"
    table.write_text(
        "x,y,t,T\n"
        "0,0,0,0\n"
        "1,0,0,2\n"
        "0,1,0,4\n"
        "1,1,0,8\n",
        encoding="utf-8",
    )
    split = tmp_path / "split.json"
    split.write_text(
        '{"splits":{"train":[0,1],"val":[2],"test":[3]}}',
        encoding="utf-8",
    )
    output = tmp_path / "baseline.json"

    status = field_baseline_main(
        [
            "--table",
            str(table),
            "--target",
            "T",
            "--strategy",
            "mean",
            "--split-manifest",
            str(split),
            "--output",
            str(output),
        ]
    )

    payload = output.read_text(encoding="utf-8")
    assert status == 0
    assert '"split_metrics"' in payload
    assert '"baseline": "constant:mean:fit=train"' in payload
