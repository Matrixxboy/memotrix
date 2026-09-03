from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseExtractor(ABC):
    supported_extensions: tuple[str, ...] = ()

    @abstractmethod
    def extract(self, path: Path) -> Any:
        raise NotImplementedError
