from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseExtractor(ABC):
    supported_extensions: tuple[str, ...] = ()

    @abstractmethod
    def extract(self, path: Path) -> Any:
        raise NotImplementedError

    def read_text(self, path: Path, encodings: tuple[str, ...] = ("utf-8", "utf-8-sig", "latin-1")) -> str:
        for encoding in encodings:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="ignore")