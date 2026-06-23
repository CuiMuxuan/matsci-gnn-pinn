from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pandas as pd


def _load_module():
    script = Path("scripts/server/build_phase161_uci_steel_plates_faults_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase161_uci_steel_plates", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXPECTED_MIN_BYTES = 1
    module.MIN_ROWS_FOR_REVIEW = 120
    module.MIN_SPLIT_ROWS = 10
    module.MIN_CLASS_COUNT_PER_SPLIT = 3
    module.MODEL_METHODS = ("knn", "extra_trees")
    return module


def _synthetic_source(path: Path, *, n_per_class: int = 28) -> Path:
    labels = [
        "Pastry",
        "Z_Scratch",
        "K_Scatch",
        "Stains",
        "Dirtiness",
        "Bumps",
        "Other_Faults",
    ]
    columns = [
        "X_Minimum",
        "X_Maximum",
        "Y_Minimum",
        "Y_Maximum",
        "Pixels_Areas",
        "X_Perimeter",
        "Y_Perimeter",
        "Sum_of_Luminosity",
        "Minimum_of_Luminosity",
        "Maximum_of_Luminosity",
        "Length_of_Conveyer",
        "TypeOfSteel_A300",
        "TypeOfSteel_A400",
        "Steel_Plate_Thickness",
        "Edges_Index",
        "Empty_Index",
        "Square_Index",
        "Outside_X_Index",
        "Edges_X_Index",
        "Edges_Y_Index",
        "Outside_Global_Index",
        "LogOfAreas",
        "Log_X_Index",
        "Log_Y_Index",
        "Orientation_Index",
        "Luminosity_Index",
        "SigmoidOfAreas",
        *labels,
    ]
    rows = []
    for class_index, label in enumerate(labels):
        for item in range(n_per_class):
            width = 5 + class_index * 3 + (item % 4)
            height = 8 + class_index * 2 + (item % 5)
            area = width * height + class_index * 7
            x_min = 20 + class_index * 40 + item
            y_min = 1000 + class_index * 500 + item * 3
            min_lum = 40 + class_index * 5
            max_lum = min_lum + 35 + (item % 6)
            label_values = [1 if name == label else 0 for name in labels]
            rows.append(
                [
                    x_min,
                    x_min + width,
                    y_min,
                    y_min + height,
                    area,
                    width + 3,
                    height + 4,
                    area * (min_lum + 10),
                    min_lum,
                    max_lum,
                    1300 + (class_index % 3) * 160,
                    1 if class_index % 2 == 0 else 0,
                    0 if class_index % 2 == 0 else 1,
                    60 + (class_index % 4) * 20,
                    0.02 * (class_index + 1),
                    0.1 + 0.01 * item,
                    min(width, height) / max(width, height),
                    width / 2000.0,
                    0.2 + class_index * 0.03,
                    0.4 + class_index * 0.02,
                    float(class_index % 3),
                    1.0 + class_index * 0.2,
                    0.4 + class_index * 0.1,
                    0.5 + class_index * 0.08,
                    (height - width) / max(height, 1),
                    -0.3 + class_index * 0.05,
                    min(1.0, area / 500.0),
                    *label_values,
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Faults.NNA", pd.DataFrame(rows).to_csv(sep="\t", header=False, index=False))
        archive.writestr("Faults27x7_var", "\n".join(columns) + "\n")
    return path


def test_phase161_builds_baseline_gate_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    raw_path = _synthetic_source(tmp_path / "raw" / "steel_plates_faults.zip")

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        raw_path=raw_path,
        source_url="https://example.invalid/steel.zip",
        allow_download=False,
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 161
    assert gate["status"] in {
        "phase161_uci_steel_plates_faults_ready_focused_review",
        "phase161_uci_steel_plates_faults_closed_no_stable_guarded_gap",
    }
    assert manifest["counts"]["class_count"] == 7
    assert gate["selected_target"] == "target_fault_class"
    assert gate["phase161_model_mechanism_allowed"] is False
    assert gate["phase161_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert (tmp_path / "out" / "phase161_baseline_metric_table.csv").exists()
    assert (tmp_path / "out" / "phase161_uci_steel_plates_faults_baseline_gate.json").exists()


def test_phase161_group_split_keeps_context_groups_together(tmp_path: Path):
    module = _load_module()
    raw_path = _synthetic_source(tmp_path / "raw" / "steel_plates_faults.zip")
    frame = module.load_steel_plates_table(raw_path)
    split = module.split_by_group(frame)

    assert split["group_column"] == "steel_geometry_context_key"
    for context_key, group in frame.groupby("steel_geometry_context_key"):
        assigned = {split["assignments"][int(index)] for index in group.index}
        assert len(assigned) == 1, context_key


def test_phase161_rejects_invalid_one_hot_labels(tmp_path: Path):
    module = _load_module()
    raw_path = _synthetic_source(tmp_path / "raw" / "steel_plates_faults.zip")
    with zipfile.ZipFile(raw_path) as archive:
        names = archive.read("Faults27x7_var")
        rows = archive.read("Faults.NNA").decode("utf-8").splitlines()
    first = rows[0].split("\t")
    first[-1] = "1"
    rows[0] = "\t".join(first)
    bad_path = tmp_path / "raw" / "bad_steel_plates_faults.zip"
    with zipfile.ZipFile(bad_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Faults.NNA", "\n".join(rows) + "\n")
        archive.writestr("Faults27x7_var", names)

    try:
        module.load_steel_plates_table(bad_path)
    except ValueError as exc:
        assert "one-hot" in str(exc)
    else:
        raise AssertionError("Expected invalid one-hot labels to raise ValueError")


def test_phase161_profiles_include_row_order_shortcut_control():
    module = _load_module()
    profiles = module.profile_columns(pd.DataFrame())

    assert profiles["row_order_control"]["role"] == "shortcut_control"
    assert profiles["row_order_control"]["columns"] == ("row_order_fraction",)
