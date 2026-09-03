"""Tests for Phases 2/4/5/6 helpers that do not need transformer weights."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memotrix.filetypes.structuredData.csv import CSVExtractor
from memotrix.processes.memory_scoring import memory_score_multiplier, stamp_new_memory_payload
from memotrix.vectorDB.base import payload_matches_filters
from memotrix.vectorDB.embeddings import QueryEmbeddingCache, cache_key_for_query, format_query_for_embedding
from memotrix.vectorDB.hnsw_index import HNSWDenseIndex
from memotrix.vectorDB.sparse_index import BM25SparseIndex
from memotrix.processes.retrieval import RetrievalPipeline


def test_csv_embeds_row_batches_not_headers_only():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.csv"
        path.write_text(
            "name,role\nAda,engineer\nGrace,researcher\n",
            encoding="utf-8",
        )
        doc = CSVExtractor().extract(path)
        assert "Ada" in doc.text
        assert "Grace" in doc.text
        assert doc.metadata["row_count"] == 2


def test_payload_filters():
    payload = {"session_id": "s1", "memory_type": "episodic"}
    assert payload_matches_filters(payload, {"session_id": "s1"})
    assert not payload_matches_filters(payload, {"session_id": "s2"})
    assert payload_matches_filters(payload, {"memory_type": "episodic", "session_id": "s1"})


def test_recency_boost_prefers_fresh_memories():
    now = datetime.now(timezone.utc)
    fresh = stamp_new_memory_payload(
        {"last_accessed": now.isoformat(), "access_count": 0, "importance": 1}
    )
    stale = stamp_new_memory_payload(
        {
            "last_accessed": (now - timedelta(days=90)).isoformat(),
            "access_count": 0,
            "importance": 1,
        }
    )
    assert memory_score_multiplier(fresh, now=now) > memory_score_multiplier(stale, now=now)


def test_query_embedding_cache_lru():
    cache = QueryEmbeddingCache(maxsize=2)
    cache.put("a", [1.0])
    cache.put("b", [2.0])
    cache.put("c", [3.0])
    assert cache.get("a") is None
    assert cache.get("b") == [2.0]
    assert len(cache) == 2


def test_bge_query_prefix():
    text = format_query_for_embedding("dark mode", "BAAI/bge-small-en-v1.5")
    assert text.startswith("Represent this sentence")
    assert cache_key_for_query("Dark  Mode", "m") == cache_key_for_query("dark mode", "m")


def test_orphan_cleanup_on_indexes():
    dense = HNSWDenseIndex(dim=4, max_elements=100)
    sparse = BM25SparseIndex()
    dense.add(
        ["a"],
        [[1.0, 0.0, 0.0, 0.0]],
        [{"source_path": "/tmp/old.md", "chunk_text": "old fact"}],
    )
    sparse.add(["a"], ["old fact"], [{"source_path": "/tmp/old.md", "chunk_text": "old fact"}])
    assert dense.delete_by_source_path("/tmp/old.md") == 1
    assert sparse.delete_by_source_path("/tmp/old.md") == 1
    assert dense.search([1.0, 0.0, 0.0, 0.0], k=1) == []


def test_conditional_rerank_skip_on_large_margin():
    pipeline = RetrievalPipeline.__new__(RetrievalPipeline)
    pipeline.enable_conditional_rerank = True
    pipeline.reranker_model_name = "dummy"
    fused = [("a", 0.05, {}), ("b", 0.03, {})]
    assert pipeline._should_skip_rerank(fused, [("a", 0.9, {})], [("a", 1.0, {})]) is True
    tight = [("a", 0.031, {}), ("b", 0.030, {})]
    assert pipeline._should_skip_rerank(tight, [("a", 0.5, {})], [("b", 0.5, {})]) is False
