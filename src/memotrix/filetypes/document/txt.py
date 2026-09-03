from pathlib import Path

from .base import BaseExtractor
from memotrix.utils.outputSturcture import build_document


class TXTExtractor(BaseExtractor):
    supported_extensions = (".txt",)

    def extract(self, path: Path):
        text = self.read_text(path)
        return build_document(path, text, extra_metadata={"file_type": "text", "kind": "text"})