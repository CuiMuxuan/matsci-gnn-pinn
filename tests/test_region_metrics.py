from __future__ import annotations

from pathlib import Path

from gnnpinn.data.loaders import load_field_table
from gnnpinn.eval.regions import region_metric_tables


def test_region_metric_tables_hot_and_gradient_regions(tmp_path: Path):
    table = tmp_path / "field.csv"
    table.write_text(
        "x,y,t,T,frame_index,row_index,col_index\n"
        "0,0,0,10,0,0,0\n"
        "1,0,0,20,0,0,1\n"
        "0,1,0,30,0,1,0\n"
        "1,1,0,100,0,1,1\n",
        encoding="utf-8",
    )
    sample = load_field_table(table)

    regions = region_metric_tables(
        sample,
        target="T",
        y_pred=[10.0, 20.0, 30.0, 90.0],
        hot_quantiles=[0.75],
        gradient_quantiles=[0.75],
    )

    assert regions["hot_q75"]["n_points"] >= 1
    assert regions["hot_q75"]["metrics"]["rmse"] >= 0
    assert regions["gradient_q75"]["n_points"] >= 1
    assert regions["gradient_q75"]["selector"]["kind"] == "spatial_gradient_quantile"
