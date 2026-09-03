from pathlib import Path

from .base import BaseExtractor
from memotrix.utils.outputSturcture import build_document


def extract_text_from_html(html_content: bytes | str) -> str:
    """Extracts clean text from HTML content using BeautifulSoup when available."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        BeautifulSoup = None  # type: ignore[misc, assignment]

    try:
        if BeautifulSoup is not None:
            soup = BeautifulSoup(html_content, "html.parser")
            return soup.get_text(separator=" ", strip=True)
    except Exception:
        pass

    if isinstance(html_content, bytes):
        return html_content.decode(errors="ignore")
    return str(html_content)

class HTMLExtractor(BaseExtractor):
    supported_extensions = (".html", ".htm")

    def extract(self, path: Path):
        content = self.read_text(path)
        text = extract_text_from_html(content)
        metadata = {"file_type": "html", "kind": "text"}
        return build_document(path, text, extra_metadata=metadata)
