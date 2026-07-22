from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile


SCRIPT = Path(__file__).parents[1] / "scripts/server/phase192_amb2022_03_calibration_intake.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase192_calibration_intake", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_minimal_workbook(path: Path, headers: list[str]) -> None:
    header_cells = "".join(
        f'<c r="{chr(ord("A") + index)}1" t="s"><v>{index}</v></c>'
        for index in range(len(headers))
    )
    shared_strings = "".join(f"<si><t>{header}</t></si>" for header in headers)
    sheet_xml = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
        f'<row r="1">{header_cells}</row>'
        '<row r="2"><c r="A2" t="inlineStr"><is><t>BP1</t></is></c></row>'
        '<row r="3"><c r="A3"><v>1</v></c></row>'
        '<row r="4"/>'
        "</sheetData></worksheet>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/></Relationships>',
        )
        archive.writestr(
            "xl/sharedStrings.xml",
            '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f"{shared_strings}</sst>",
        )
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


class Phase212StdlibWorkbookTest(unittest.TestCase):
    def test_stdlib_workbook_fallback_preserves_phase192_schema(self) -> None:
        module = _load_module()
        original_loader = module._openpyxl

        def missing_openpyxl():
            raise ModuleNotFoundError("forced for standard-library fallback test")

        module._openpyxl = missing_openpyxl
        try:
            with TemporaryDirectory() as temporary_directory:
                workbook_path = Path(temporary_directory) / "cross_sections.xlsx"
                headers = sorted(module.WORKBOOK_REQUIRED_COLUMNS)
                _write_minimal_workbook(workbook_path, headers)
                summary, inventory = module.inspect_workbook(workbook_path)
        finally:
            module._openpyxl = original_loader

        self.assertEqual(summary["sheet_names"], ["Sheet1"])
        self.assertEqual(set(summary["headers"]["Sheet1"]), set(module.WORKBOOK_REQUIRED_COLUMNS))
        self.assertEqual(summary["data_row_count"], 2)
        self.assertEqual(inventory[0]["attributes"], "reader=stdlib_xlsx_xml")


if __name__ == "__main__":
    unittest.main()
