from pathlib import Path

from .base import BaseExtractor
from memotrix.utils.outputSturcture import build_document


class YAMLExtractor(BaseExtractor):
    supported_extensions = (".yaml", ".yml")

    def extract(self, path: Path):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("PyYAML is required for YAML extraction") from exc

        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        text = yaml.safe_dump(data, sort_keys=False)
        return build_document(path, text, extra_metadata={"file_type": "yaml", "library": "pyyaml"})
