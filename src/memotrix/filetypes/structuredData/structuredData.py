from pathlib import Path
from typing import Any

from .csv import CSVExtractor
from .excel import ExcelExtractor
from .json import JSONExtractor
from .sql import SQLExtractor
from .xml import XMLExtractor
from .yaml import YAMLExtractor
from memotrix.utils.exceptions import EmptyDocumentError, UnsupportedDocumentTypeError


class StructuredDataExtractor:
    EXTRACTORS = {
        ".csv": CSVExtractor,
        ".xlsx": ExcelExtractor,
        ".json": JSONExtractor,
        ".sql": SQLExtractor,
        ".xml": XMLExtractor,
        ".yaml": YAMLExtractor,
        ".yml": YAMLExtractor,
    }

    @classmethod
    def extract(cls, path: str | Path) -> Any:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"The file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"The path is not a file: {path}")

        extractor_cls = cls.EXTRACTORS.get(path.suffix.lower())
        if extractor_cls is None:
            if path.suffix.lower() == ".xls":
                raise UnsupportedDocumentTypeError(
                    "Unsupported file type: .xls. Convert to .xlsx "
                    "(openpyxl cannot read BIFF .xls files)."
                )
            raise UnsupportedDocumentTypeError(f"Unsupported file type: {path.suffix}")

        document = extractor_cls().extract(path)
        if document is None:
            raise EmptyDocumentError(f"No document content could be extracted from {path}")
        return document