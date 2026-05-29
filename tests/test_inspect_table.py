from pathlib import Path

import yaml

from gnnpinn.data.inspect_table import draft_mapping, inspect_table


def test_inspect_table_and_draft_mapping(tmp_path: Path):
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "X_mm,Y_mm,time_s,temperature_K,label\n"
        "0,0,0,300,a\n"
        "1,0,0.5,315,b\n",
        encoding="utf-8",
    )

    inspection = inspect_table(raw)
    mapping = draft_mapping(
        inspection,
        output=tmp_path / "processed" / "field_tables" / "sample.csv",
        sample_id="sample",
    )

    assert inspection["n_rows"] == 2
    assert "label" not in inspection["numeric_columns"]
    assert mapping["columns"]["x"] == "X_mm"
    assert mapping["columns"]["t"] == "time_s"
    assert mapping["observations"]["T"] == "temperature_K"


def test_inspect_table_cli_draft(tmp_path: Path):
    from gnnpinn.data.inspect_table import main

    raw = tmp_path / "raw.csv"
    raw.write_text("x,y,t,T\n0,0,0,300\n", encoding="utf-8")
    draft = tmp_path / "mapping.yaml"

    status = main(["--table", str(raw), "--sample-id", "demo", "--draft-output", str(draft)])

    assert status == 0
    payload = yaml.safe_load(draft.read_text(encoding="utf-8"))
    assert payload["sample_id"] == "demo"
    assert payload["observations"]["T"] == "T"
