"""SDK lifecycle: add_text / search / delete / list_sources / config / rehydrate."""

from pathlib import Path

from memotrix import Memory
from memotrix.config import ChunkingConfig, IngestConfig, MemoryConfig, RetrievalConfig
from memotrix.embeddings import FakeEmbeddings, OpenAIEmbeddings
from memotrix.filetypes import extract_file, supported_extensions
from memotrix.processes.document_store import DocumentStore
from memotrix.utils.exceptions import ConfigurationError
from memotrix.utils.models import DocumentData


def test_add_text_search_delete_list():
    memory = Memory(embeddings=FakeEmbeddings(dim=8), backend="memory")
    added = memory.add_text(
        "The user prefers dark mode in the editor.",
        source_id="prefs.md",
        memory_type="semantic",
        session_id="s1",
    )
    assert added["chunks"] >= 1
    assert added["path"] == "prefs.md"

    sources = memory.list_sources()
    assert len(sources) == 1
    assert sources[0]["source_path"] == "prefs.md"
    assert sources[0]["chunks"] >= 1

    hits = memory.search("dark mode", top_k=3, session_id="s1")
    assert hits
    assert any("dark mode" in (h.get("chunk_text") or "").lower() for h in hits)

    deleted = memory.delete("prefs.md")
    assert deleted >= 1
    assert memory.list_sources() == []
    memory.close()


def test_chunking_config_changes_chunk_count():
    text = ("Paragraph about apples and cider. " * 20) + "\n\n" + (
        "Paragraph about oranges and zest. " * 20
    )
    small = MemoryConfig(
        chunking=ChunkingConfig(chunk_size=80, chunk_overlap=0),
        ingest=IngestConfig(enable_dedup=False),
    )
    large = MemoryConfig(
        chunking=ChunkingConfig(chunk_size=5000, chunk_overlap=0),
        ingest=IngestConfig(enable_dedup=False),
    )
    m1 = Memory(embeddings=FakeEmbeddings(dim=8), backend="memory", config=small)
    m2 = Memory(embeddings=FakeEmbeddings(dim=8), backend="memory", config=large)
    r1 = m1.add_text(text, source_id="small.txt")
    r2 = m2.add_text(text, source_id="large.txt")
    assert r1["chunks"] > r2["chunks"]
    m1.close()
    m2.close()


def test_retrieval_config_fusion_and_cache():
    config = MemoryConfig(
        retrieval=RetrievalConfig(fusion_k=17, query_cache_size=4, enable_rerank=False)
    )
    memory = Memory(embeddings=FakeEmbeddings(dim=8), backend="memory", config=config)
    assert memory.retrieval.fusion_k == 17
    assert memory.retrieval.query_cache.maxsize == 4
    memory.close()


def test_rehydrate_neighbors_from_payloads():
    config = MemoryConfig(
        chunking=ChunkingConfig(chunk_size=40, chunk_overlap=0),
        ingest=IngestConfig(enable_dedup=False),
        retrieval=RetrievalConfig(
            expand_mode="neighbors",
            neighbor_window=1,
            enable_rerank=False,
            top_k=1,
        ),
    )
    embeddings = FakeEmbeddings(dim=8)
    memory = Memory(embeddings=embeddings, backend="memory", config=config)
    text = (
        "Alpha unique token lives here in the first window. "
        "Beta unique token is the middle section of the document. "
        "Gamma unique token finishes the last window of text."
    )
    memory.add_text(text, source_id="story.txt")

    restarted = Memory(
        embeddings=embeddings,
        store=memory._bundle,
        document_store=DocumentStore(),
        config=config,
    )
    assert len(restarted.document_store) >= 1
    hits = restarted.search("Beta unique token", top_k=1, expand_mode="neighbors")
    assert hits
    hit = hits[0]
    assert hit.get("expansion") == "neighbors"
    neighbors = hit.get("neighbor_chunk_indices") or []
    assert len(neighbors) > 1
    joined = hit.get("chunk_text") or ""
    matched = hit.get("matched_chunk") or ""
    assert joined != matched or "Beta unique token" in joined
    memory.close()
    restarted.close()


def test_extract_file_hook(tmp_path: Path):
    target = tmp_path / "note.txt"
    target.write_text("ignored on disk", encoding="utf-8")

    def fake_extract(path, **_kwargs):
        return DocumentData(
            metadata={"filename": Path(path).name},
            sections=[{"title": "", "content": "hello from hook"}],
            text="hello from hook",
            validation={},
        )

    memory = Memory(
        embeddings=FakeEmbeddings(dim=8),
        backend="memory",
        extract_file=fake_extract,
        config=MemoryConfig(ingest=IngestConfig(enable_dedup=False, describe_images=False)),
    )
    result = memory.add(target)
    assert result["chunks"] >= 1
    hits = memory.search("hello from hook", top_k=1)
    assert hits
    memory.close()


def test_html_extract(tmp_path: Path):
    html = tmp_path / "page.html"
    html.write_text(
        "<html><body><h1>Revenue</h1><p>Forty two million</p></body></html>",
        encoding="utf-8",
    )
    doc = extract_file(html, describe_images=False)
    blob = (doc.text or "") + " ".join(s.get("content") or "" for s in doc.sections)
    assert "Revenue" in blob
    assert ".html" in supported_extensions()


def test_openai_embeddings_requires_model():
    try:
        OpenAIEmbeddings(model="")
        raise AssertionError("expected ConfigurationError")
    except ConfigurationError:
        pass


def test_chunk_overlap_shares_markers():
    from memotrix.processes.chunking import chunk_text, generate_chunk_id

    paras = [f"Marker{i:02d} unique paragraph about topic {i}." for i in range(8)]
    chunks = chunk_text("\n\n".join(paras), chunk_size=80, overlap=50)
    assert len(chunks) >= 2
    shared = False
    for left, right in zip(chunks, chunks[1:]):
        left_marks = {p[:8] for p in paras if p in left}
        right_marks = {p[:8] for p in paras if p in right}
        if left_marks & right_marks:
            shared = True
            break
    assert shared

    same = "identical body text"
    id_a = generate_chunk_id(
        same, {"filename": "a.txt", "source_path": "/tmp/a.txt", "chunk_index": 0}
    )
    id_b = generate_chunk_id(
        same, {"filename": "a.txt", "source_path": "/other/a.txt", "chunk_index": 0}
    )
    id_c = generate_chunk_id(
        same, {"filename": "a.txt", "source_path": "/tmp/a.txt", "chunk_index": 1}
    )
    assert id_a != id_b
    assert id_a != id_c


def test_reingest_clears_stale_neighbors():
    config = MemoryConfig(
        chunking=ChunkingConfig(chunk_size=40, chunk_overlap=0),
        ingest=IngestConfig(enable_dedup=False),
        retrieval=RetrievalConfig(
            expand_mode="neighbors",
            neighbor_window=2,
            enable_rerank=False,
            top_k=1,
        ),
    )
    memory = Memory(embeddings=FakeEmbeddings(dim=8), backend="memory", config=config)
    memory.add_text(
        "Alpha stale token lives in the first window. "
        "Beta stale token is the middle section of the document. "
        "Gamma stale token finishes the last window of text.",
        source_id="story.txt",
    )
    memory.add_text("Short replacement only.", source_id="story.txt")
    hits = memory.search("Short replacement", top_k=1, expand_mode="neighbors")
    assert hits
    joined = hits[0].get("chunk_text") or ""
    assert "stale token" not in joined
    assert "Short replacement" in joined
    memory.close()


def test_search_does_not_persist_expanded_payload():
    config = MemoryConfig(
        chunking=ChunkingConfig(chunk_size=40, chunk_overlap=0),
        ingest=IngestConfig(enable_dedup=False),
        retrieval=RetrievalConfig(
            expand_mode="neighbors",
            neighbor_window=1,
            enable_rerank=False,
            top_k=1,
        ),
    )
    memory = Memory(embeddings=FakeEmbeddings(dim=8), backend="memory", config=config)
    memory.add_text(
        "Alpha unique token lives here in the first window. "
        "Beta unique token is the middle section of the document. "
        "Gamma unique token finishes the last window of text.",
        source_id="story.txt",
    )
    hits = memory.search("Beta unique token", top_k=1, expand_mode="neighbors")
    assert hits
    hit = hits[0]
    assert hit.get("expansion") == "neighbors"
    stored = memory.dense.get_payload(hit["chunk_id"])
    assert stored is not None
    assert stored.get("chunk_text") == hit.get("matched_chunk")
    assert "expansion" not in stored
    assert "neighbor_chunk_indices" not in stored
    memory.close()


def test_delete_filename_removes_absolute_source(tmp_path: Path):
    note = tmp_path / "note.txt"
    note.write_text("Filename delete should drop the indexed note.", encoding="utf-8")
    memory = Memory(
        embeddings=FakeEmbeddings(dim=8),
        backend="memory",
        config=MemoryConfig(ingest=IngestConfig(enable_dedup=False, describe_images=False)),
    )
    memory.add(note)
    assert memory.list_sources()
    deleted = memory.delete("note.txt")
    assert deleted >= 1
    assert memory.list_sources() == []
    memory.close()


def test_delete_does_not_endswith_unrelated_sources():
    memory = Memory(
        embeddings=FakeEmbeddings(dim=8),
        backend="memory",
        config=MemoryConfig(ingest=IngestConfig(enable_dedup=False)),
    )
    memory.add_text("report contents", source_id="/tmp/report_a.txt")
    memory.add_text("short file", source_id="/tmp/a.txt")
    memory.delete("/tmp/a.txt")
    paths = {item["source_path"] for item in memory.list_sources()}
    assert "/tmp/report_a.txt" in paths
    assert "/tmp/a.txt" not in paths
    memory.close()


def test_dedup_does_not_merge_across_sources():
    memory = Memory(
        embeddings=FakeEmbeddings(dim=8),
        backend="memory",
        config=MemoryConfig(ingest=IngestConfig(enable_dedup=True)),
    )
    text = "The user prefers dark mode in the editor."
    memory.add_text(text, source_id="prefs-a")
    memory.add_text(text, source_id="prefs-b")
    paths = {item["source_path"] for item in memory.list_sources()}
    assert paths == {"prefs-a", "prefs-b"}
    memory.close()


def test_hnsw_filtered_search_finds_distant_matches():
    from memotrix.vectorDB.hnsw_index import HNSWDenseIndex

    dim = 8
    index = HNSWDenseIndex(dim=dim, max_elements=200)
    query = [1.0] + [0.0] * (dim - 1)
    ids, vecs, payloads = [], [], []
    for i in range(50):
        ids.append(f"other-{i}")
        vecs.append([0.99 - i * 0.001] + [0.01] * (dim - 1))
        payloads.append({"session_id": "other", "chunk_text": f"other {i}"})
    for i in range(2):
        ids.append(f"keep-{i}")
        vecs.append([0.01] + [0.8] * (dim - 1))
        payloads.append({"session_id": "keep", "chunk_text": f"keep {i}"})
    index.add(ids, vecs, payloads)
    hits = index.search(query, k=2, filters={"session_id": "keep"})
    assert len(hits) == 2
    assert all(hit[2]["session_id"] == "keep" for hit in hits)


def test_connection_without_backend_selects_postgres(monkeypatch):
    from memotrix.vectorstores import InMemoryStore, VectorStoreBundle

    captured = {}

    class FakePg:
        def __init__(self, connection, embeddings, table_name="memotrix_chunks"):
            captured["connection"] = connection
            self.connection = connection
            inner = InMemoryStore(embeddings)
            self.bundle = VectorStoreBundle(
                dense=inner.dense, sparse=inner.sparse, native=self
            )

        def close(self):
            pass

    monkeypatch.setattr("memotrix.memory.PostgresStore", FakePg)
    memory = Memory(
        embeddings=FakeEmbeddings(dim=4),
        connection="postgresql://u:p@localhost:5432/memotrix",
    )
    assert captured["connection"].startswith("postgresql://")
    memory.close()


def test_connection_with_memory_backend_raises():
    try:
        Memory(
            embeddings=FakeEmbeddings(dim=4),
            backend="memory",
            connection="postgresql://u:p@localhost:5432/memotrix",
        )
        raise AssertionError("expected ConfigurationError")
    except ConfigurationError as exc:
        assert "incompatible" in str(exc)


def test_search_rejects_unknown_kwargs():
    import inspect

    memory = Memory(embeddings=FakeEmbeddings(dim=8), backend="memory")
    assert "kwargs" not in inspect.signature(Memory.search).parameters
    try:
        memory.search("q", unknown=1)
        raise AssertionError("expected TypeError")
    except TypeError:
        pass
    memory.close()


def test_in_memory_store_has_close():
    from memotrix.vectorstores import InMemoryStore

    store = InMemoryStore(FakeEmbeddings(dim=4), max_elements=10)
    store.close()
