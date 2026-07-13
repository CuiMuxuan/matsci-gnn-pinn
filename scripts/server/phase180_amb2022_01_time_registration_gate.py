#!/usr/bin/env python3
"""Block or admit AMB2022-01 thermal training based on time registration evidence."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "AMB2022_01_DATA_ROOT",
        "/root/matsci-gnn-pinn-data/raw/ambench/2022_3d_build/AMB2022-01/mds2-2607",
    )
)


def clock_seconds(value: str) -> float:
    parsed = dt.time.fromisoformat(value)
    return parsed.hour * 3600 + parsed.minute * 60 + parsed.second + parsed.microsecond / 1_000_000


def build_gate(*, xypt_seconds: float, thermocouple_seconds: float, trigger_is_absolute_time: bool) -> dict[str, Any]:
    gap = thermocouple_seconds - xypt_seconds
    ratio = abs(gap) / max(1.0, thermocouple_seconds)
    registered = trigger_is_absolute_time and ratio <= 0.05
    return {
        "status": "phase180_time_registration_ready" if registered else "phase180_time_registration_blocked_missing_absolute_schedule",
        "xypt_command_seconds": xypt_seconds,
        "thermocouple_window_seconds": thermocouple_seconds,
        "unexplained_gap_seconds": gap,
        "unexplained_gap_fraction": ratio,
        "trigger_is_absolute_time": trigger_is_absolute_time,
        "coordinate_time_registration_ready": registered,
        "model_training_allowed": False,
        "a800_training_allowed_now": False,
        "next_action": (
            "obtain or reconstruct an auditable build-start and layer/recoat schedule before any B6/B7 fitting"
            if not registered
            else "register sensor times to causal XYPT history before fitting"
        ),
    }


def inspect(data_root: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("h5py is required for the time-registration gate") from exc
    h5_path = data_root / "Scan_Strategy" / "AMB2022-01-AMMT-XYPT_v1.h5"
    tc_path = data_root / "Thermocouples" / "AMB2022-01-AMMT-B6-Thermocouple.csv"
    readme = (data_root / "2607_README.txt").read_text(encoding="utf-8", errors="replace")
    with h5py.File(h5_path, "r") as handle:
        xypt = handle["XYPT"]
        digital_rate = float(xypt.attrs["digital_rate"][0])
        layer_ids = sorted(xypt.keys(), key=int)
        sample_count = sum(int(xypt[layer]["P"].shape[-1]) for layer in layer_ids)
    with tc_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    thermocouple_seconds = clock_seconds(rows[-1]["Time"]) - clock_seconds(rows[0]["Time"])
    trigger_text = "trigger vector" in readme.lower() and "instrument trigger" in readme.lower()
    gate = build_gate(
        xypt_seconds=sample_count / digital_rate,
        thermocouple_seconds=thermocouple_seconds,
        trigger_is_absolute_time=not trigger_text,
    )
    return {
        "phase": 180,
        "data_root": str(data_root),
        "digital_rate_hz": digital_rate,
        "layer_count": len(layer_ids),
        "xypt_sample_count": sample_count,
        "thermocouple_source": str(tc_path),
        "readme_trigger_evidence": "T is an instrument trigger, not an absolute time vector" if trigger_text else "not found",
        "gate": gate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = inspect(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
