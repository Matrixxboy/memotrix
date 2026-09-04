import pytest
from pathlib import Path
import sys
from unittest.mock import patch
from memotrix.filetypes import extract_file
from memotrix.utils.exceptions import MissingDependencyError
from memotrix.filetypes.document.pdf import PDFExtractor

def test_pdf_extractor_missing_dependency(tmp_path):
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n")

    # Force a mock ImportError for 'fitz' (PyMuPDF)
    with patch.dict(sys.modules, {'fitz': None}):
        extractor = PDFExtractor()
        with pytest.raises(MissingDependencyError, match="PyMuPDF \\(fitz\\) is required for PDF extraction"):
            extractor.extract(pdf_file)
