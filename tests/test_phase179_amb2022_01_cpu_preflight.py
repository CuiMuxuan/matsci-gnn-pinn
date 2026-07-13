from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase179_amb2022_01_cpu_preflight.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase179_cpu_preflight", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_thermocouple_summary_and_frozen_split(tmp_path: Path):
    module = _load_module()
    thermocouples = tmp_path / "Thermocouples"
    thermocouples.mkdir()
    for build in ("B6", "B7", "B8"):
        channels = module.THERMOCOUPLE_CHANNELS[build]
        fields = ["Time", *channels]
        (thermocouples / f"AMB2022-01-AMMT-{build}-Thermocouple.csv").write_text(
            ",".join(fields) + "\n0," + ",".join("1" for _ in channels) + "\n2," + ",".join("3" for _ in channels) + "\n",
            encoding="utf-8",
        )
    summary = module.summarize_thermocouple_csv(thermocouples / "AMB2022-01-AMMT-B6-Thermocouple.csv")
    assert summary["row_count"] == 2
    assert summary["numeric_ranges"]["Time"] == [0.0, 2.0]
    rows = module.build_split_rows(tmp_path)
    assert [row["build_id"] for row in rows] == ["B6", "B7", "B8"]
    assert rows[-1]["split"] == "test_frozen"
    assert rows[-1]["model_selection_allowed"] == "false"
