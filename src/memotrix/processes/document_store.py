from typing import Any, Dict, List, Optional, Tuple

from memotrix.vectorDB.base import source_keys_match


def document_key(metadata: dict) -> str:
    return metadata.get("source_path") or metadata.get("filename", "")


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token). Good enough for budget caps."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


class DocumentStore:
    """
    Holds full-file text plus chunk adjacency for neighbor-window expansion.

    Chunks are keyed by (doc_key, section_title, chunk_index) so retrieval can
    expand a hit to prev/next chunks from the same section without dumping the
    entire source file into an agent context window.
    """

    def __init__(self) -> None:
        self._documents: Dict[str, str] = {}
        # doc_key -> section_title -> chunk_index -> chunk payload snapshot
        self._chunks: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}

    def register(self, key: str, text: str) -> None:
        if key and text:
            self._documents[key] = text

    def register_chunk(self, payload: Dict[str, Any]) -> None:
        """Index a chunk for neighbor lookup. Safe to call repeatedly."""
        key = document_key(payload)
        if not key:
            return
        if payload.get("type", "text") != "text":
            return

        section = payload.get("section_title", "")
        chunk_index = payload.get("chunk_index")
        if chunk_index is None:
            return

        by_section = self._chunks.setdefault(key, {})
        by_index = by_section.setdefault(section, {})
        by_index[int(chunk_index)] = {
            "chunk_text": payload.get("chunk_text", ""),
            "chunk_index": int(chunk_index),
            "section_title": section,
            "source_path": payload.get("source_path"),
            "filename": payload.get("filename"),
        }

    def register_chunks(self, payloads: List[Dict[str, Any]]) -> None:
        for payload in payloads:
            self.register_chunk(payload)

    def rehydrate(self, payloads: List[Dict[str, Any]]) -> int:
        """Rebuild adjacency (and concatenated file text) from stored payloads."""
        by_key: Dict[str, List[Dict[str, Any]]] = {}
        for payload in payloads:
            self.register_chunk(payload)
            key = document_key(payload)
            if key:
                by_key.setdefault(key, []).append(payload)
        for key, items in by_key.items():
            ordered = sorted(
                items,
                key=lambda p: (
                    p.get("chunk_index")
                    if p.get("chunk_index") is not None
                    else 10**9,
                    p.get("row_start")
                    if p.get("row_start") is not None
                    else 10**9,
                ),
            )
            text = "\n\n".join(
                str(p.get("chunk_text") or "") for p in ordered if p.get("chunk_text")
            )
            if text:
                self.register(key, text)
        return len(by_key)

    def get(self, key: str) -> Optional[str]:
        return self._documents.get(key)

    def get_for_metadata(self, metadata: dict) -> Optional[str]:
        return self.get(document_key(metadata))

    def delete_source(self, source_path: str) -> bool:
        """Remove full text and chunk adjacency for a source file."""
        if not source_path:
            return False
        removed = False
        for key in list(self._documents.keys()):
            if source_keys_match(key, source_path):
                del self._documents[key]
                removed = True
        for key in list(self._chunks.keys()):
            if source_keys_match(key, source_path):
                del self._chunks[key]
                removed = True
        return removed

    def get_neighbor_window(
        self,
        metadata: Dict[str, Any],
        *,
        window: int = 1,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Return (joined_text, neighbor_payloads) for matched chunk ± window
        within the same section. Falls back to the matched chunk alone.
        """
        key = document_key(metadata)
        section = metadata.get("section_title", "")
        chunk_index = metadata.get("chunk_index")
        matched = metadata.get("chunk_text", "")

        if key not in self._chunks or chunk_index is None:
            return matched, []

        by_index = self._chunks[key].get(section, {})
        if not by_index:
            return matched, []

        center = int(chunk_index)
        indices = sorted(
            i for i in by_index if center - window <= i <= center + window
        )
        neighbors = [by_index[i] for i in indices]
        joined = "\n\n".join(
            n["chunk_text"] for n in neighbors if n.get("chunk_text")
        )
        return joined or matched, neighbors

    def __len__(self) -> int:
        return len(self._documents)
