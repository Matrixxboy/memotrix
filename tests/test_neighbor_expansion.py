"""Unit tests for neighbor-window expansion (no embedding models required)."""

from memotrix.processes.document_store import DocumentStore, estimate_tokens
from memotrix.processes.retrieval import RetrievalPipeline


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 800) == 200


def test_neighbor_window_joins_prev_next():
    store = DocumentStore()
    source = "/tmp/prefs.md"
    for i, text in enumerate(["alpha chunk", "beta chunk", "gamma chunk"]):
        store.register_chunk(
            {
                "source_path": source,
                "filename": "prefs.md",
                "section_title": "UI",
                "chunk_index": i,
                "chunk_text": text,
                "type": "text",
            }
        )

    joined, neighbors = store.get_neighbor_window(
        {
            "source_path": source,
            "section_title": "UI",
            "chunk_index": 1,
            "chunk_text": "beta chunk",
        },
        window=1,
    )
    assert neighbors[0]["chunk_index"] == 0
    assert neighbors[-1]["chunk_index"] == 2
    assert "alpha chunk" in joined
    assert "beta chunk" in joined
    assert "gamma chunk" in joined


def test_neighbor_window_stays_in_section():
    store = DocumentStore()
    source = "/tmp/prefs.md"
    store.register_chunk(
        {
            "source_path": source,
            "section_title": "UI",
            "chunk_index": 0,
            "chunk_text": "dark mode",
            "type": "text",
        }
    )
    store.register_chunk(
        {
            "source_path": source,
            "section_title": "Locale",
            "chunk_index": 0,
            "chunk_text": "Asia/Kolkata",
            "type": "text",
        }
    )

    joined, neighbors = store.get_neighbor_window(
        {
            "source_path": source,
            "section_title": "UI",
            "chunk_index": 0,
            "chunk_text": "dark mode",
        },
        window=1,
    )
    assert len(neighbors) == 1
    assert "Asia/Kolkata" not in joined


def test_max_tokens_trims_across_results():
    """Exercise expansion trim logic without loading transformer models."""
    store = DocumentStore()
    source = "/tmp/big.md"
    long_a = "A" * 2000
    long_b = "B" * 2000
    store.register_chunk(
        {
            "source_path": source,
            "section_title": "S",
            "chunk_index": 0,
            "chunk_text": long_a,
            "type": "text",
        }
    )
    store.register_chunk(
        {
            "source_path": source,
            "section_title": "S",
            "chunk_index": 1,
            "chunk_text": long_b,
            "type": "text",
        }
    )

    pipeline = RetrievalPipeline.__new__(RetrievalPipeline)
    pipeline.document_store = store
    pipeline.expand_mode = "neighbors"
    pipeline.neighbor_window = 0

    results = [
        {
            "source_path": source,
            "section_title": "S",
            "chunk_index": 0,
            "chunk_text": long_a,
        },
        {
            "source_path": source,
            "section_title": "S",
            "chunk_index": 1,
            "chunk_text": long_b,
        },
    ]
    expanded = pipeline._expand_neighbors(results, max_tokens_returned=100)
    assert len(expanded) >= 1
    total = sum(estimate_tokens(r["chunk_text"]) for r in expanded)
    assert total <= 100 + 5  # small slack for ellipsis / rounding
    assert expanded[0].get("matched_chunk") == long_a
