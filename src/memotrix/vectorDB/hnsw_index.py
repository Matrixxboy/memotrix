import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hnswlib
import numpy as np

from .base import DenseIndex, PayloadFilters, payload_matches_filters, payload_matches_source


class HNSWDenseIndex(DenseIndex):
    """
    In-memory HNSW index using hnswlib.
    Designed for fast semantic search using sentence-transformers.
    """

    def __init__(self, dim: int, *, space: str = "cosine", max_elements: int = 10000):
        if dim <= 0:
            raise ValueError("HNSWDenseIndex requires a positive dim from the embedding model")
        self.space = space
        self.dim = dim
        self.max_elements = max_elements
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(
            max_elements=self.max_elements,
            ef_construction=200,
            M=16,
            allow_replace_deleted=True,
        )
        self.index.set_ef(50)

        self.int_to_str_id: Dict[int, str] = {}
        self.payloads: Dict[str, Dict[str, Any]] = {}
        self.str_to_int_id: Dict[str, int] = {}
        self.current_int_id = 0
        self._deleted: set[int] = set()

    def add(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        if not ids:
            return

        data_labels = []
        data_vectors = np.array(vectors, dtype=np.float32)

        for i, uid in enumerate(ids):
            if uid in self.str_to_int_id:
                int_id = self.str_to_int_id[uid]
                if int_id in self._deleted:
                    self._deleted.discard(int_id)
            else:
                int_id = self.current_int_id
                self.current_int_id += 1
                self.str_to_int_id[uid] = int_id
                self.int_to_str_id[int_id] = uid

            data_labels.append(int_id)
            self.payloads[uid] = payloads[i]

        self.index.add_items(
            data_vectors, np.array(data_labels), replace_deleted=True
        )

    def _hits_from_knn(
        self,
        query_data: np.ndarray,
        fetch: int,
        k: int,
        filters: PayloadFilters,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        labels, distances = self.index.knn_query(query_data, k=max(fetch, 1))
        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for i in range(labels.shape[1]):
            int_id = int(labels[0][i])
            if int_id in self._deleted:
                continue
            str_id = self.int_to_str_id.get(int_id)
            if str_id is None:
                continue
            payload = self.payloads.get(str_id, {})
            if not payload_matches_filters(payload, filters):
                continue
            distance = float(distances[0][i])
            score = 1.0 - distance if self.space == "cosine" else -distance
            results.append((str_id, score, payload))
            if len(results) >= k:
                break
        return results

    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        filters: PayloadFilters = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        live = self.current_int_id - len(self._deleted)
        if live <= 0:
            return []

        try:
            current_count = int(self.index.get_current_count())
        except Exception:
            current_count = live
        max_fetch = max(min(current_count, live, self.max_elements), 1)
        query_data = np.array([query_vector], dtype=np.float32)
        if not filters:
            return self._hits_from_knn(query_data, min(max(k, 1), max_fetch), k, filters)

        fetch = min(max(k * 5, k, 1), max_fetch)
        results = self._hits_from_knn(query_data, fetch, k, filters)
        while len(results) < k and fetch < max_fetch:
            fetch = min(max(fetch * 2, fetch + 1), max_fetch)
            results = self._hits_from_knn(query_data, fetch, k, filters)
        return results

    def delete_by_source_path(self, source_path: str) -> int:
        to_delete = [
            uid
            for uid, payload in self.payloads.items()
            if payload_matches_source(payload, source_path)
        ]
        for uid in to_delete:
            int_id = self.str_to_int_id.get(uid)
            if int_id is None:
                continue
            try:
                self.index.mark_deleted(int_id)
            except RuntimeError:
                pass
            self._deleted.add(int_id)
            self.payloads.pop(uid, None)
            self.str_to_int_id.pop(uid, None)
            self.int_to_str_id.pop(int_id, None)
        return len(to_delete)

    def update_payload(self, chunk_id: str, payload: Dict[str, Any]) -> None:
        if chunk_id in self.payloads:
            self.payloads[chunk_id] = payload

    def get_payload(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        payload = self.payloads.get(chunk_id)
        return dict(payload) if payload is not None else None

    def payloads_for_source(self, source_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        matches: List[Tuple[str, Dict[str, Any]]] = []
        for uid, payload in self.payloads.items():
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
        for uid, payload in self.payloads.items():
            yield uid, dict(payload)

    def save(self, path: str) -> None:
        base_path = Path(path)
        base_path.mkdir(parents=True, exist_ok=True)

        self.index.save_index(str(base_path / "hnsw_index.bin"))

        state = {
            "dim": self.dim,
            "space": self.space,
            "max_elements": self.max_elements,
            "int_to_str_id": self.int_to_str_id,
            "str_to_int_id": self.str_to_int_id,
            "current_int_id": self.current_int_id,
            "payloads": self.payloads,
            "deleted": list(self._deleted),
        }
        with open(base_path / "hnsw_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        base_path = Path(path)

        with open(base_path / "hnsw_state.json", "r", encoding="utf-8") as f:
            state = json.load(f)

        self.dim = state["dim"]
        self.space = state["space"]
        self.max_elements = state["max_elements"]
        self.int_to_str_id = {int(k): v for k, v in state["int_to_str_id"].items()}
        self.str_to_int_id = state["str_to_int_id"]
        self.current_int_id = state["current_int_id"]
        self.payloads = state["payloads"]
        self._deleted = set(state.get("deleted", []))

        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.load_index(
            str(base_path / "hnsw_index.bin"), max_elements=self.max_elements
        )
        self.index.set_ef(50)
