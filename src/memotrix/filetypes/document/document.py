from pathlib import Path
from typing import Any

from memotrix.utils.exceptions import EmptyDocumentError, UnsupportedDocumentTypeError

from .docx import DOCXExtractor
from .epub import EPUBExtractor
from .markdown import MarkdownExtractor
from .pdf import PDFExtractor
from .pptx import PPTXExtractor
from .txt import TXTExtractor
from .html import HTMLExtractor


class DocumentExtractor:
    EXTRACTORS = {
        ".pdf": PDFExtractor,
        ".docx": DOCXExtractor,
        ".pptx": PPTXExtractor,
        ".txt": TXTExtractor,
        ".md": MarkdownExtractor,
        ".markdown": MarkdownExtractor,
        ".html": HTMLExtractor,
        ".htm": HTMLExtractor,
        ".epub": EPUBExtractor,
    }

    @classmethod
    def extract(cls, path: str | Path, *, describe_images: bool = True) -> Any:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"The file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"The path is not a file: {path}")

        extractor_cls = cls.EXTRACTORS.get(path.suffix.lower())
        if extractor_cls is None:
            raise UnsupportedDocumentTypeError(f"Unsupported file type: {path.suffix}")

        extractor = extractor_cls()
        try:
            document = extractor.extract(path, describe_images=describe_images)
        except TypeError:
            document = extractor.extract(path)
        if document is None:
            raise EmptyDocumentError(f"No document content could be extracted from {path}")
        return document
