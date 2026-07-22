from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase210_remote_data_integrity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase210_remote_data_integrity", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_mds2716_parser_ignores_checksum_sidecar_and_verifies_data_file(tmp_path: Path):
    module = _load_module()
    data_root = tmp_path / "data"
    relative = Path("Thermography/signal.h5")
    destination = data_root / module.MDS2716_LOCAL_ROOT / relative
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"verified bytes")
    metadata = {
        "components": [
            {
                "@type": ["nrdp:DataFile"],
                "filepath": relative.as_posix(),
                "size": destination.stat().st_size,
                "checksum": {"hash": _sha256(destination.read_bytes())},
            },
            {
                "@type": ["nrdp:ChecksumFile"],
                "filepath": "Thermography/signal.h5.sha256",
                "size": 64,
                "checksum": {"hash": _sha256(b"sidecar")},
            },
        ]
    }

    records = module.records_from_mds2716_metadata(metadata, data_root)

    assert len(records) == 1
    verified = module.verify_record(records[0])
    assert verified["status"] == "verified"
    assert verified["sha256_matches"] is True


def test_mds2715_parser_handles_comment_preamble_and_headerless_rows(tmp_path: Path):
    module = _load_module()
    data_root = tmp_path / "data"
    relative = "DataProcessingScripts/temperature.m"
    payload = b"candidate code"
    destination = data_root / module.MDS2715_LOCAL_ROOT / relative
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)
    listing = tmp_path / "_filelisting.csv"
    listing.write_text(
        "# official NIST listing\n"
        f"{relative},{len(payload)},,text/plain,{_sha256(payload)},https://example.invalid/file\n",
        encoding="utf-8",
    )

    records = module.records_from_mds2715_listing(listing, data_root)

    assert len(records) == 1
    assert records[0]["source_relative_path"] == relative
    assert module.verify_record(records[0])["status"] == "verified"


def test_gate_blocks_required_formula_input_when_not_selected():
    module = _load_module()
    records = [
        {
            "record_key": key,
            "status": "verified" if index else "not_selected",
        }
        for index, key in enumerate(module.REQUIRED_RECORD_KEYS)
    ]

    gate = module.build_gate(records)

    assert gate["phase211_formula_identifiability_allowed"] is False
    assert "required_formula_input_not_verified" in gate["blocking_audits"]
