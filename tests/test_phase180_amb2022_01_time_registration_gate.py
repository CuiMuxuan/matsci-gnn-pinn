from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase180_amb2022_01_time_registration_gate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase180_time_registration", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_time_registration_blocks_missing_schedule():
    module = _load_module()
    gate = module.build_gate(xypt_seconds=12646.67106, thermocouple_seconds=27634.0, trigger_is_absolute_time=False)
    assert gate["status"] == "phase180_time_registration_blocked_missing_absolute_schedule"
    assert gate["coordinate_time_registration_ready"] is False
    assert gate["model_training_allowed"] is False


def test_clock_seconds_parses_wall_clock():
    module = _load_module()
    assert module.clock_seconds("12:29:06") == 44946.0
