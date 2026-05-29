from pathlib import Path

import yaml

from gnnpinn.data.audit import audit_dataset, determine_readiness


def test_determine_readiness_without_required_local_data():
    readiness = determine_readiness(
        file_count=0,
        modalities=[],
        phase0_gates={"require_local_data": False, "min_existing_files": 1},
    )

    assert readiness == "source_registered_no_local_files"


def test_audit_dataset_detects_matching_modalities(tmp_path: Path):
    local_root = tmp_path / "raw"
    local_root.mkdir()
    (local_root / "sample_temperature.csv").write_text("x,y,t,T\n0,0,0,300\n", encoding="utf-8")

    config = {
        "dataset_id": "toy_ambench",
        "local_root": str(local_root),
        "source_pages": ["https://example.test/data"],
        "expected_modalities": {
            "thermal": {
                "required": False,
                "patterns": ["**/*temperature*"],
            }
        },
        "phase0_gates": {
            "min_existing_files": 1,
            "min_modalities_present": 1,
            "require_local_data": True,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    audit = audit_dataset(config_path, project_root=tmp_path)

    assert audit.dataset_id == "toy_ambench"
    assert audit.file_count == 1
    assert audit.modalities[0].present is True
    assert audit.readiness == "ready_for_phase1"

