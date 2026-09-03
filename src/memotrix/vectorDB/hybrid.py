from typing import Any, Dict, List, Optional, Tuple

from .base import DenseIndex, PayloadFilters, SparseIndex


def reciprocal_rank_fusion(
    dense_results: List[Tuple[str, float, Dict[str, Any]]],
    sparse_results: List[Tuple[str, float, Dict[str, Any]]],
    k: int = 60
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Fuses two ranked lists using Reciprocal Rank Fusion (RRF).

    RRF_Score(d) = sum(1 / (k + rank(d)))
    """
    rrf_scores: Dict[str, float] = {}
    payloads: Dict[str, Dict[str, Any]] = {}

    for rank, (doc_id, _score, payload) in enumerate(dense_results):
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0.0
            payloads[doc_id] = payload
        rrf_scores[doc_id] += 1.0 / (k + rank + 1)

    for rank, (doc_id, _score, payload) in enumerate(sparse_results):
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0.0
            payloads[doc_id] = payload
        rrf_scores[doc_id] += 1.0 / (k + rank + 1)

    fused_results = [
        (doc_id, score, payloads[doc_id])
        for doc_id, score in rrf_scores.items()
    ]
    fused_results.sort(key=lambda x: x[1], reverse=True)
    return fused_results


class HybridSearchEngine:
    """Orchestrates dense + sparse search and fuses results with RRF."""

    def __init__(self, dense_index: DenseIndex, sparse_index: SparseIndex):
        self.dense = dense_index
        self.sparse = sparse_index

    def search(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: int = 10,
        fusion_k: int = 60,
        filters: PayloadFilters = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        fused, _, _ = self.search_with_channels(
            query_text,
            query_vector,
            top_k=top_k,
            fusion_k=fusion_k,
            filters=filters,
        )
        return fused

    def search_with_channels(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: int = 10,
        fusion_k: int = 60,
        filters: PayloadFilters = None,
    ) -> Tuple[
        List[Tuple[str, float, Dict[str, Any]]],
        List[Tuple[str, float, Dict[str, Any]]],
        List[Tuple[str, float, Dict[str, Any]]],
    ]:
        """
        Returns (fused, dense_hits, sparse_hits) so callers can decide
        whether cross-encoder reranking is worth the latency.
        """
        fetch_k = top_k * 5
        dense_hits = self.dense.search(query_vector, k=fetch_k, filters=filters)
        sparse_hits = self.sparse.search(query_text, k=fetch_k, filters=filters)
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=fusion_k)
        return fused[:top_k], dense_hits, sparse_hits
