from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase212_remote_cpu_gate_revalidation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase212_remote_cpu_gate_revalidation", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase192() -> dict[str, object]:
    return {
        "gate": {
            "status": "phase192_amb2022_03_calibration_intake_ready_phase193_identifier_join_design",
            "blocking_audits": [],
        }
    }


def _phase201() -> dict[str, object]:
    return {
        "thermalcal_intake": {
            "raw_signal_arrays_read": False,
            "cross_section_targets_read": False,
            "calibration_fitting_performed": False,
        },
        "gate": {
            "status": "phase201_thermalcal_metadata_audit_ready_phase202_formula_contract_design",
            "blocking_audits": [],
        },
    }


def _phase202() -> dict[str, object]:
    return {
        "formula_contract": {"temperature_conversion_executed": False},
        "gate": {
            "status": "phase202_formula_contract_complete_temperature_conversion_blocked",
            "blocking_audits": [
                "hdf5_formula_contains_undefined_symbol",
                "hdf5_formula_not_unambiguously_mapped_to_official_equation",
            ],
            "calibration_formula_execution_allowed": False,
            "model_training_allowed": False,
        },
    }


class Phase212RemoteCpuGateTest(unittest.TestCase):
    def test_headerless_official_listing_matches_workbook_bytes(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workbook = root / "results.xlsx"
            workbook.write_bytes(b"official workbook bytes")
            digest = module.sha256_file(workbook)
            listing = root / "_filelisting.csv"
            listing.write_text(
                f"{workbook.name},{workbook.stat().st_size},,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,{digest},https://example.invalid/{workbook.name}\n",
                encoding="utf-8",
            )
            provenance = module.verify_workbook(workbook, listing)
        self.assertEqual(provenance["status"], "verified")
        self.assertTrue(provenance["sha256_matches"])

    def test_complete_gate_preserves_formula_and_training_blocks(self) -> None:
        module = _load_module()
        payload = module.build_payload(
            phase192=_phase192(),
            phase201=_phase201(),
            phase202=_phase202(),
            workbook_provenance={"status": "verified"},
        )
        self.assertTrue(payload["gate"]["phase212_complete"])
        self.assertFalse(payload["gate"]["temperature_conversion_allowed"])
        self.assertFalse(payload["gate"]["model_training_allowed"])


if __name__ == "__main__":
    unittest.main()
