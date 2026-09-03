from pathlib import Path

from .base import BaseExtractor
from memotrix.utils.outputSturcture import build_document


class SQLExtractor(BaseExtractor):
    supported_extensions = (".sql",)

    def extract(self, path: Path):
        text = path.read_text(encoding="utf-8")
        return build_document(path, text, extra_metadata={"file_type": "sql", "library": "native"})
