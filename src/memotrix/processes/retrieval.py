from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memotrix.processes.document_store import DocumentStore, estimate_tokens
from memotrix.processes.memory_scoring import bump_access, memory_score_multiplier
from memotrix.utils.exceptions import ConfigurationError
from memotrix.vectorDB.base import PayloadFilters
from memotrix.vectorDB.embeddings import (
    RERANK_AGREE_MARGIN,
    RERANK_SKIP_MARGIN,
    QueryEmbeddingCache,
    cache_key_for_query,
)
from memotrix.vectorDB.hybrid import HybridSearchEngine

# Agent loops cannot afford dumping five full files into context.
DEFAULT_TOP_K = 3
DEFAULT_NEIGHBOR_WINDOW = 1
DEFAULT_MAX_TOKENS = 800
_SEARCH_ONLY_KEYS = frozenset(
    {
        "score",
        "reranked",
        "expansion",
        "matched_chunk",
        "neighbor_chunk_indices",
        "tokens_estimate",
        "content",
        "relevance_score",
        "memory_multiplier",
        "truncated",
    }
)


class RetrievalPipeline:
    """
    Handles user queries by embedding the query,
    running a hybrid search, and reranking the results.

    Expansion modes:
      - "neighbors": matched chunk + prev/next in the same section (agent default)
      - "full_file": legacy expand top hit to entire source file
      - "none": return matched chunks only
    """

    def __init__(
        self,
        hybrid_engine: HybridSearchEngine,
        embeddings=None,
        reranker_model_name: str | None = None,
        document_store: Optional[DocumentStore] = None,
        expand_top_file_content: bool = False,
        expand_mode: str = "neighbors",
        neighbor_window: int = DEFAULT_NEIGHBOR_WINDOW,
        *,
        embedding_model_name: str | None = None,
        query_cache: Optional[QueryEmbeddingCache] = None,
        enable_memory_boost: bool = True,
        enable_conditional_rerank: bool = True,
        track_access: bool = True,
        encoder=None,
        fusion_k: int = 60,
        query_cache_size: int = 256,
    ):
        self.hybrid_engine = hybrid_engine
        if embeddings is None and embedding_model_name:
            from memotrix.embeddings import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(model=embedding_model_name)
        if embeddings is None:
            raise ConfigurationError(
                "RetrievalPipeline requires embeddings=... "
                "(HuggingFaceEmbeddings(model=...) or FakeEmbeddings(dim=...))"
            )
        self.embeddings = embeddings
        self.embedding_model_name = getattr(embeddings, "model_name", embedding_model_name or "")
        self.reranker_model_name = reranker_model_name
        self._encoder = encoder
        self._reranker = None
        self.document_store = document_store
        if expand_top_file_content:
            self.expand_mode = "full_file"
        else:
            self.expand_mode = expand_mode
        self.neighbor_window = neighbor_window
        self.query_cache = (
            query_cache
            if query_cache is not None
            else QueryEmbeddingCache(maxsize=query_cache_size)
        )
        self.fusion_k = fusion_k
        self.enable_memory_boost = enable_memory_boost
        self.enable_conditional_rerank = enable_conditional_rerank and bool(reranker_model_name)
        self.track_access = track_access
        self._reranker_loaded = False

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = self.embeddings
        return self._encoder

    @property
    def reranker(self):
        if not self.reranker_model_name:
            return None
        if self._reranker is None:
            from memotrix.vectorDB.embeddings import get_cross_encoder

            self._reranker = get_cross_encoder(self.reranker_model_name)
            self._reranker_loaded = True
        return self._reranker

    def _embed_query(self, query: str) -> List[float]:
        """Phase 6 — cache repeated / near-identical agent queries."""
        key = cache_key_for_query(query, self.embedding_model_name)
        cached = self.query_cache.get(key)
        if cached is not None:
            return cached

        vector = self.embeddings.embed_query(query)
        self.query_cache.put(key, vector)
        return vector

    def _should_skip_rerank(
        self,
        fused: List[Tuple[str, float, Dict[str, Any]]],
        dense_hits: List[Tuple[str, float, Dict[str, Any]]],
        sparse_hits: List[Tuple[str, float, Dict[str, Any]]],
    ) -> bool:
        """
        Phase 2 — skip cross-encoder when channels already agree clearly.
        Saves latency inside tight agent loops.
        """
        if not self.reranker_model_name:
            return True
        if not self.enable_conditional_rerank or len(fused) < 2:
            return False

        margin = fused[0][1] - fused[1][1]
        if margin >= RERANK_SKIP_MARGIN:
            return True

        if dense_hits and sparse_hits and dense_hits[0][0] == sparse_hits[0][0]:
            if margin >= RERANK_AGREE_MARGIN:
                return True
            # Same top-1 in both channels with a solid dense score gap.
            if len(dense_hits) >= 2 and (dense_hits[0][1] - dense_hits[1][1]) >= 0.05:
                return True

        return False

    def _apply_memory_boost(
        self, ranked: List[Tuple[float, Dict[str, Any]]]
    ) -> List[Tuple[float, Dict[str, Any]]]:
        if not self.enable_memory_boost:
            return ranked
        boosted = []
        for score, payload in ranked:
            mult = memory_score_multiplier(payload)
            item = dict(payload)
            item["relevance_score"] = score
            item["memory_multiplier"] = mult
            boosted.append((score * mult, item))
        boosted.sort(key=lambda x: x[0], reverse=True)
        return boosted

    def _stored_payload(self, chunk_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        dense = self.hybrid_engine.dense
        getter = getattr(dense, "get_payload", None)
        stored = getter(chunk_id) if callable(getter) else None
        if stored:
            return dict(stored)
        sparse = self.hybrid_engine.sparse
        getter = getattr(sparse, "get_payload", None)
        stored = getter(chunk_id) if callable(getter) else None
        if stored:
            return dict(stored)
        cleaned = {k: v for k, v in result.items() if k not in _SEARCH_ONLY_KEYS}
        matched = result.get("matched_chunk")
        if matched:
            cleaned["chunk_text"] = matched
        return cleaned

    def _record_access(self, results: List[Dict[str, Any]]) -> None:
        if not self.track_access:
            return
        dense = self.hybrid_engine.dense
        sparse = self.hybrid_engine.sparse
        for result in results:
            chunk_id = result.get("chunk_id")
            if not chunk_id:
                continue
            stored = self._stored_payload(chunk_id, result)
            bumped = bump_access(stored)
            result["last_accessed"] = bumped["last_accessed"]
            result["access_count"] = bumped["access_count"]
            dense.update_payload(chunk_id, bumped)
            sparse.update_payload(chunk_id, bumped)

    def _source_key(self, payload: Dict[str, Any]) -> str:
        return str(payload.get("source_path") or payload.get("filename") or "")

    def _assemble_source_from_index(self, source_key: str) -> Optional[str]:
        """Join every indexed chunk for a source (sorted), when DocumentStore is empty."""
        if not source_key:
            return None
        dense = self.hybrid_engine.dense
        matches = dense.payloads_for_source(source_key)
        if not matches:
            sparse = self.hybrid_engine.sparse
            matches = sparse.payloads_for_source(source_key)
        if not matches:
            return None
        parts: List[str] = []
        seen: set[str] = set()
        for _uid, payload in matches:
            text = (payload.get("chunk_text") or "").strip()
            if text and text not in seen:
                seen.add(text)
                parts.append(text)
        return "\n\n".join(parts) if parts else None

    def _read_source_from_disk(self, source_key: str) -> Optional[str]:
        """Re-extract text from the original uploaded file when present on disk."""
        if not source_key:
            return None
        path = Path(source_key)
        if not path.is_file():
            return None
        try:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                from memotrix.filetypes.structuredData.csv import CSVExtractor

                return CSVExtractor().extract(path).text
            if suffix in {".xlsx", ".xls"}:
                from memotrix.filetypes.structuredData.excel import ExcelExtractor

                return ExcelExtractor().extract(path).text
            if suffix in {".txt", ".md", ".markdown"}:
                return path.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 — fall through to other expand paths
            return None
        return None

    def _resolve_full_source_text(self, payload: Dict[str, Any]) -> Optional[str]:
        source_key = self._source_key(payload)
        if self.document_store:
            full = self.document_store.get_for_metadata(payload)
            if full:
                return full
            # Also try filename-only key
            filename = payload.get("filename")
            if filename:
                full = self.document_store.get(str(filename))
                if full:
                    return full
        assembled = self._assemble_source_from_index(source_key)
        if assembled:
            return assembled
        return self._read_source_from_disk(source_key)

    def _expand_full_file(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Replace hits with complete source-file text for every unique source.

        Fallback order: DocumentStore → all index chunks for source → re-read disk.
        """
        if not results:
            return results

        expanded: List[Dict[str, Any]] = []
        seen_sources: set[str] = set()

        for hit in results:
            item = dict(hit)
            source_key = self._source_key(item)
            if source_key and source_key in seen_sources:
                continue

            full_content = self._resolve_full_source_text(item)
            if full_content:
                if source_key:
                    seen_sources.add(source_key)
                item["matched_chunk"] = item.get("chunk_text", "")
                item["content"] = full_content
                item["chunk_text"] = full_content
                item["expansion"] = "full_file"
                item["tokens_estimate"] = estimate_tokens(full_content)
                expanded.append(item)
            else:
                item["content"] = item.get("chunk_text", "")
                item["expansion"] = "none"
                expanded.append(item)

        return expanded or results

    def _expand_neighbors(
        self,
        results: List[Dict[str, Any]],
        *,
        max_tokens_returned: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Expand each hit to a parent-chunk / sentence-window neighborhood."""
        if not results:
            return results

        budget = max_tokens_returned
        expanded: List[Dict[str, Any]] = []

        for result in results:
            item = dict(result)
            matched = item.get("chunk_text", "")
            item["matched_chunk"] = matched

            if self.document_store:
                window_text, neighbors = self.document_store.get_neighbor_window(
                    item, window=self.neighbor_window
                )
                item["chunk_text"] = window_text or matched
                item["content"] = item["chunk_text"]
                item["neighbor_chunk_indices"] = [
                    n.get("chunk_index") for n in neighbors
                ]
                item["expansion"] = "neighbors"
            else:
                item["content"] = matched
                item["expansion"] = "none"

            used = estimate_tokens(item["chunk_text"])
            if budget is not None:
                if budget <= 0:
                    break
                if used > budget:
                    char_budget = max(budget, 1) * 4
                    text = item["chunk_text"]
                    if len(text) > char_budget:
                        item["chunk_text"] = text[:char_budget].rstrip() + "…"
                        item["content"] = item["chunk_text"]
                        item["truncated"] = True
                        used = estimate_tokens(item["chunk_text"])
                budget -= used

            item["tokens_estimate"] = estimate_tokens(item["chunk_text"])
            expanded.append(item)

        return expanded

    def _apply_expansion(
        self,
        results: List[Dict[str, Any]],
        *,
        max_tokens_returned: Optional[int],
    ) -> List[Dict[str, Any]]:
        if self.expand_mode == "full_file":
            return self._expand_full_file(results)
        if self.expand_mode == "neighbors":
            return self._expand_neighbors(
                results, max_tokens_returned=max_tokens_returned
            )
        return results

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        max_tokens_returned: Optional[int] = DEFAULT_MAX_TOKENS,
        *,
        filters: PayloadFilters = None,
        session_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        expand_mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Agent-facing retrieval entrypoint.

        Args:
            query: Natural-language memory query.
            top_k: Max results (default 3 for agent context budgets).
            max_tokens_returned: Hard cap on total returned text (~4 chars/token).
            filters: Exact-match payload filters (session_id, memory_type, …).
            session_id / memory_type: Convenience shorthands merged into filters.
            expand_mode: Optional per-call override (``neighbors`` / ``full_file`` / ``none``).
        """
        merged_filters: Dict[str, Any] = dict(filters or {})
        if session_id is not None:
            merged_filters["session_id"] = session_id
        if memory_type is not None:
            merged_filters["memory_type"] = memory_type
        filter_arg: PayloadFilters = merged_filters or None

        previous_expand = self.expand_mode
        if expand_mode is not None:
            self.expand_mode = expand_mode

        from memotrix.utils.trace import mlog, mlog_block

        mlog(
            "retrieve",
            f"query={query!r} top_k={top_k} filters={filter_arg} expand={self.expand_mode}",
        )
        mlog_block("retrieve", "RETRIEVAL QUERY", query)

        try:
            query_vector = self._embed_query(query)
            candidates, dense_hits, sparse_hits = self.hybrid_engine.search_with_channels(
                query,
                query_vector,
                top_k=max(top_k * 4, 20),
                fusion_k=self.fusion_k,
                filters=filter_arg,
            )

            if not candidates:
                mlog("retrieve", "no candidates")
                return []

            skip_rerank = self._should_skip_rerank(candidates, dense_hits, sparse_hits)

            ranked: List[Tuple[float, Dict[str, Any]]] = []
            if skip_rerank:
                for doc_id, rrf_score, payload in candidates:
                    item = dict(payload)
                    item["chunk_id"] = doc_id
                    item["reranked"] = False
                    ranked.append((float(rrf_score), item))
            else:
                pairs = []
                valid = []
                for doc_id, rrf_score, payload in candidates:
                    chunk_text = payload.get("chunk_text", "")
                    if not chunk_text:
                        continue
                    pairs.append((query, chunk_text))
                    valid.append((doc_id, rrf_score, payload))

                if not pairs:
                    for doc_id, rrf_score, payload in candidates[:top_k]:
                        item = dict(payload)
                        item["score"] = rrf_score
                        item["chunk_id"] = doc_id
                        item["reranked"] = False
                        ranked.append((float(rrf_score), item))
                else:
                    reranker = self.reranker
                    if reranker is None:
                        for doc_id, rrf_score, payload in candidates:
                            item = dict(payload)
                            item["chunk_id"] = doc_id
                            item["reranked"] = False
                            ranked.append((float(rrf_score), item))
                    else:
                        rerank_scores = reranker.predict(pairs)
                        for i, (doc_id, _rrf, payload) in enumerate(valid):
                            item = dict(payload)
                            item["chunk_id"] = doc_id
                            item["reranked"] = True
                            ranked.append((float(rerank_scores[i]), item))

            ranked = self._apply_memory_boost(ranked)
            ranked.sort(key=lambda x: x[0], reverse=True)

            final_results: List[Dict[str, Any]] = []
            for score, payload in ranked[:top_k]:
                payload["score"] = score
                final_results.append(payload)

            expanded = self._apply_expansion(
                final_results, max_tokens_returned=max_tokens_returned
            )
            self._record_access(expanded)
            mlog(
                "retrieve",
                f"hits={len(expanded)} candidates={len(candidates)} "
                f"skip_rerank={skip_rerank}",
            )
            for i, hit in enumerate(expanded[:8], 1):
                preview = (hit.get("chunk_text") or hit.get("context") or "")[:120]
                mlog(
                    "retrieve",
                    f"  #{i} score={hit.get('score', 0):.4f} "
                    f"file={hit.get('filename') or hit.get('source_path') or '?'} "
                    f"type={hit.get('type')} | {preview!r}",
                )
            return expanded
        finally:
            self.expand_mode = previous_expand

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        max_tokens_returned: Optional[int] = DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Alias for search() — kept for backward compatibility."""
        return self.search(
            query, top_k=top_k, max_tokens_returned=max_tokens_returned, **kwargs
        )
