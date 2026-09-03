import json
from pathlib import Path

from .base import BaseExtractor
from memotrix.utils.outputSturcture import build_document


class JSONExtractor(BaseExtractor):
    supported_extensions = (".json",)

    def extract(self, path: Path, data=None):
        if data is None:
            with path.open("r", encoding="utf-8-sig") as handle:
                data = json.load(handle)

        text = json.dumps(data, indent=2, ensure_ascii=False)
        return build_document(path, text, extra_metadata={"file_type": "json", "library": "json"})
