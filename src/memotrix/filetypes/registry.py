"""Extractor Plugin Registry for dynamically registering filetype extractors."""

from typing import Any, Callable, Dict, List
from pathlib import Path

class ExtractorRegistry:
    _extractors: Dict[str, Callable] = {}

    @classmethod
    def register(cls, extension: str, extractor_func: Callable) -> None:
        """Register a new extractor for a specific extension."""
        cls._extractors[extension.lower()] = extractor_func

    @classmethod
    def get_extractor(cls, extension: str) -> Callable | None:
        """Get the extractor function for a specific extension."""
        return cls._extractors.get(extension.lower())

    @classmethod
    def supported_extensions(cls) -> List[str]:
        """List all supported extensions registered in the plugin registry."""
        return sorted(list(cls._extractors.keys()))

def register_extractor(extension: str, extractor_func: Callable) -> None:
    """Helper function to register an extractor globally."""
    ExtractorRegistry.register(extension, extractor_func)
