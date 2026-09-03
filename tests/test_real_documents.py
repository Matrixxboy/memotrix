import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
for _p in (str(SRC), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from memotrix.filetypes import extract_file


TEST_FILES_DIR = Path(__file__).parent / "filesForTests"
OUTPUT_DIR = Path(__file__).parent / "doc_test_data"
OUTPUT_DIR.mkdir(exist_ok=True)


def _save_document_output(document, fixture: Path) -> None:
    output_path = OUTPUT_DIR / f"{fixture.stem}_extracted.json"
    payload = {
        "filename": document.metadata["filename"],
        "extension": document.metadata["extension"],
        "kind": document.metadata.get("kind", "unknown"),
        "text": document.text,
        "sections": document.sections,
        "validation": document.validation,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def test_supported_fixture_files_extract_successfully() -> None:
    fixtures = [
        TEST_FILES_DIR / "this_is_wow.txt",
        TEST_FILES_DIR / "SignalHire_exports.csv",
        TEST_FILES_DIR / "donor_report_final.pdf",
        TEST_FILES_DIR / "Realtime_Conversation_Analytics_Report_1.docx",
        TEST_FILES_DIR / "donor_report.pptx",
        TEST_FILES_DIR / "Excel_Practice.xlsx",
    ]

    for fixture in fixtures:
        assert fixture.exists(), f"Fixture missing: {fixture}"
        document = extract_file(fixture)
        _save_document_output(document, fixture)
        assert document.text.strip(), f"No extracted text for {fixture.name}"
        assert document.metadata["filename"] == fixture.name
        assert document.metadata["extension"] == fixture.suffix.lower()


def test_text_fixture_has_sections_and_metadata() -> None:
    fixture = TEST_FILES_DIR / "this_is_wow.txt"
    document = extract_file(fixture)
    _save_document_output(document, fixture)

    assert len(document.sections) >= 1
    assert document.validation["word_count"] > 0
    assert document.metadata["kind"] == "text"


def run_checks() -> None:
    test_supported_fixture_files_extract_successfully()
    test_text_fixture_has_sections_and_metadata()
    print("Document extraction checks passed.")


if __name__ == "__main__":
    run_checks()
