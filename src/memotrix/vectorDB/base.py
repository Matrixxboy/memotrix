from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Dict, List, Optional, Tuple


PayloadFilters = Optional[Dict[str, Any]]


def source_basename(value: str) -> str:
    if not value:
        return ""
    return value.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


def is_filename_only(value: str) -> bool:
    return bool(value) and "/" not in value and "\\" not in value


def source_keys_match(stored_key: str, query: str) -> bool:
    """Match a stored document key to a delete/lookup string.

    Exact path, exact filename, or basename of the stored path when the
    caller passed a filename (no directory separators). Never ``endswith``.
    """
    if not query or not stored_key:
        return False
    if stored_key == query:
        return True
    if is_filename_only(query) and source_basename(stored_key) == query:
        return True
    return False


def payload_matches_source(payload: Dict[str, Any], source_path: str) -> bool:
    if not source_path:
        return False
    stored_path = str(payload.get("source_path") or "")
    stored_name = str(payload.get("filename") or "")
    if stored_path == source_path or stored_name == source_path:
        return True
    if stored_path and source_keys_match(stored_path, source_path):
        return True
    if stored_name and source_keys_match(stored_name, source_path):
        return True
    return False


def payload_matches_filters(payload: Dict[str, Any], filters: PayloadFilters) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if expected is None:
            continue
        if payload.get(key) != expected:
            return False
    return True


class DenseIndex(ABC):
    """Base interface for a Dense Vector Index (e.g., HNSW)."""

    @abstractmethod
    def add(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        """Add vectors and their associated metadata payloads to the index."""

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        filters: PayloadFilters = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for the top-k most similar vectors. Returns (id, score, payload)."""

    def delete_by_source_path(self, source_path: str) -> int:
        """Remove chunks belonging to a source file. Returns count deleted."""
        return 0

    def update_payload(self, chunk_id: str, payload: Dict[str, Any]) -> None:
        """Update metadata for an existing chunk (access counts, etc.)."""

    def get_payload(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Return a copy of the stored payload, or None if missing."""
        return None

    def payloads_for_source(self, source_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Return (chunk_id, payload) for every chunk from a source file."""
        return []

    def iter_payloads(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Yield ``(chunk_id, payload)`` for every live chunk."""
        return iter(())

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the index to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the index from disk."""


class SparseIndex(ABC):
    """Base interface for a Sparse Inverted Index (e.g., BM25)."""

    @abstractmethod
    def add(self, ids: List[str], texts: List[str], payloads: List[Dict[str, Any]]) -> None:
        """Add texts and their associated metadata payloads to the sparse index."""

    @abstractmethod
    def search(
        self,
        query_text: str,
        k: int = 10,
        filters: PayloadFilters = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for the top-k most relevant texts. Returns (id, score, payload)."""

    def delete_by_source_path(self, source_path: str) -> int:
        """Remove chunks belonging to a source file. Returns count deleted."""
        return 0

    def update_payload(self, chunk_id: str, payload: Dict[str, Any]) -> None:
        """Update metadata for an existing chunk."""

    def get_payload(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Return a copy of the stored payload, or None if missing."""
        return None

    def payloads_for_source(self, source_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Return (chunk_id, payload) for every chunk from a source file."""
        return []

    def iter_payloads(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Yield ``(chunk_id, payload)`` for every live chunk."""
        return iter(())

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the sparse index to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the sparse index from disk."""
