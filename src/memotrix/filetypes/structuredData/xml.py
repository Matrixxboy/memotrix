import xml.etree.ElementTree as ET
from pathlib import Path

from .base import BaseExtractor
from memotrix.utils.outputSturcture import build_document


class XMLExtractor(BaseExtractor):
    supported_extensions = (".xml",)

    def extract(self, path: Path):
        tree = ET.parse(path)
        root = tree.getroot()
        text = ET.tostring(root, encoding="unicode")
        return build_document(path, text, extra_metadata={"file_type": "xml", "library": "xml.etree"})
