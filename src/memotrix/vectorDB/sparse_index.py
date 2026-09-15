import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import PayloadFilters, SparseIndex, payload_matches_filters, payload_matches_source


def _bm25_okapi():
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        from memotrix.utils.exceptions import MissingDependencyError

        raise MissingDependencyError(
            "rank-bm25 is required for InMemoryStore. Install with: pip install 'memotrix[local]'"
        ) from exc
    return BM25Okapi


class BM25SparseIndex(SparseIndex):
    """
    In-memory Sparse Index using rank_bm25 (Okapi BM25).
    Designed for fast keyword and exact match search.
    """

    def __init__(self):
        self.ids: List[str] = []
        self.texts: List[str] = []
        self.payloads: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.split(r"\W+", text)
        return [t for t in tokens if t]

    def _rebuild(self) -> None:
        if self.tokenized_corpus:
            self.bm25 = _bm25_okapi()(self.tokenized_corpus)
        else:
            self.bm25 = None

    def add(self, ids: List[str], texts: List[str], payloads: List[Dict[str, Any]]) -> None:
        if not ids:
            return

        for i, uid in enumerate(ids):
            if uid in self.ids:
                idx = self.ids.index(uid)
                self.ids.pop(idx)
                self.texts.pop(idx)
                self.payloads.pop(idx)
                self.tokenized_corpus.pop(idx)

            self.ids.append(uid)
            self.texts.append(texts[i])
            self.payloads.append(payloads[i])
            self.tokenized_corpus.append(self._tokenize(texts[i]))

        self._rebuild()

    def search(
        self,
        query_text: str,
        k: int = 10,
        filters: PayloadFilters = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not self.bm25 or not self.ids:
            return []

        tokenized_query = self._tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in ranked:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            payload = self.payloads[idx]
            if not payload_matches_filters(payload, filters):
                continue
            results.append((self.ids[idx], score, payload))
            if len(results) >= k:
                break

        return results

    def delete_by_source_path(self, source_path: str) -> int:
        keep = []
        deleted = 0
        for i, payload in enumerate(self.payloads):
            if payload_matches_source(payload, source_path):
                deleted += 1
            else:
                keep.append(i)

        if deleted == 0:
            return 0

        self.ids = [self.ids[i] for i in keep]
        self.texts = [self.texts[i] for i in keep]
        self.payloads = [self.payloads[i] for i in keep]
        self.tokenized_corpus = [self.tokenized_corpus[i] for i in keep]
        self._rebuild()
        return deleted

    def update_payload(self, chunk_id: str, payload: Dict[str, Any]) -> None:
        if chunk_id in self.ids:
            idx = self.ids.index(chunk_id)
            self.payloads[idx] = payload

    def get_payload(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        if chunk_id not in self.ids:
            return None
        return dict(self.payloads[self.ids.index(chunk_id)])

    def payloads_for_source(self, source_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        matches: List[Tuple[str, Dict[str, Any]]] = []
        for uid, payload in zip(self.ids, self.payloads):
            if payload_matches_source(payload, source_path):
                matches.append((uid, dict(payload)))
        matches.sort(
            key=lambda item: (
                item[1].get("row_start")
                if item[1].get("row_start") is not None
                else item[1].get("chunk_index")
                if item[1].get("chunk_index") is not None
                else 10**9,
                item[0],
            )
        )
        return matches

    def iter_payloads(self):
        for uid, payload in zip(self.ids, self.payloads):
            yield uid, dict(payload)

    def save(self, path: str) -> None:
        base_path = Path(path)
        base_path.mkdir(parents=True, exist_ok=True)

        state = {
            "ids": self.ids,
            "texts": self.texts,
            "payloads": self.payloads,
            "tokenized_corpus": self.tokenized_corpus,
        }
        with open(base_path / "bm25_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        base_path = Path(path)

        with open(base_path / "bm25_state.json", "r", encoding="utf-8") as f:
            state = json.load(f)

        self.ids = state["ids"]
        self.texts = state["texts"]
        self.payloads = state["payloads"]
        self.tokenized_corpus = state["tokenized_corpus"]
        self._rebuild()
