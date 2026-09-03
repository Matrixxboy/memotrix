from pathlib import Path

from .base import BaseExtractor
from memotrix.utils.outputSturcture import build_document


class MarkdownExtractor(BaseExtractor):
    supported_extensions = (".md", ".markdown")

    def extract(self, path: Path):
        try:
            import markdown_it
        except ImportError:
            markdown_it = None

        text = self.read_text(path)
        metadata = {"file_type": "markdown", "kind": "text", "library": "markdown-it-py" if markdown_it else "native"}
        return build_document(path, text, extra_metadata=metadata)
