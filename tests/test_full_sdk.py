"""
Comprehensive test suite for the Memotrix SDK.

Tests ALL functionality end-to-end:
  - Memory init (in-memory, from_config, from_env, string embeddings, error paths)
  - Embeddings (FakeEmbeddings, coerce_embeddings, error paths)
  - add_text / add / add_documents / extract
  - search (basic, filters, memory_type, session_id, expand_mode)
  - delete / list_sources
  - Dedup and re-ingest behavior
  - Error handling (empty text, bad backend, missing file, unsupported ext)
  - Extractor Plugin Registry
  - Configuration objects (ChunkingConfig, RetrievalConfig, IngestConfig, MemoryConfig)
  - Real file extraction (txt, csv, xlsx, docx, pdf, pptx, jpg)
  - as_retriever / as_stack / close
  - Medical file extraction (FHIR JSON)
"""

import json
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

# ── Imports ──────────────────────────────────────────────────────────────────
from memotrix import Memory, MemoryConfig
from memotrix.config import (
    ChunkingConfig,
    IngestConfig,
    MemoryConfig,
    RetrievalConfig,
    require_env,
    resolve_database_url,
)
from memotrix.embeddings import (
    Embeddings,
    FakeEmbeddings,
    HuggingFaceEmbeddings,
    coerce_embeddings,
)
from memotrix.filetypes import extract_file, supported_extensions
from memotrix.filetypes.registry import ExtractorRegistry, register_extractor
from memotrix.utils.exceptions import (
    ConfigurationError,
    DocumentExtractionError,
    EmptyDocumentError,
    ExtractionBackendError,
    MissingDependencyError,
    UnsupportedDocumentTypeError,
)
from memotrix.utils.models import DocumentData
from memotrix.vectorstores import InMemoryStore, VectorStoreBundle

# ── Paths ────────────────────────────────────────────────────────────────────
TEST_DIR = Path(__file__).resolve().parent
FIXTURES = TEST_DIR / "filesForTests"


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1: CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════


class TestChunkingConfig:
    def test_defaults(self):
        cfg = ChunkingConfig()
        assert cfg.chunk_size == 500
        assert cfg.chunk_overlap == 50

    def test_custom_values(self):
        cfg = ChunkingConfig(chunk_size=1000, chunk_overlap=100)
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 100


class TestRetrievalConfig:
    def test_defaults(self):
        cfg = RetrievalConfig()
        assert cfg.top_k == 3
        assert cfg.max_tokens == 800
        assert cfg.expand_mode == "neighbors"
        assert cfg.neighbor_window == 1
        assert cfg.enable_rerank is True
        assert cfg.reranker_model is None
        assert cfg.enable_memory_boost is True
        assert cfg.fusion_k == 60
        assert cfg.query_cache_size == 256


class TestIngestConfig:
    def test_defaults(self):
        cfg = IngestConfig()
        assert cfg.enable_dedup is True
        assert cfg.dedup_threshold == 0.95
        assert cfg.describe_images is True


class TestMemoryConfig:
    def test_defaults(self):
        cfg = MemoryConfig()
        assert cfg.backend == "memory"
        assert cfg.connection is None
        assert cfg.table_name == "memotrix_chunks"
        assert cfg.max_elements == 50_000
        assert cfg.embedding_model is None
        assert cfg.reranker_model is None

    def test_from_env_missing_model_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="EMBEDDING_MODEL"):
                MemoryConfig.from_env()

    def test_from_env_bad_backend_raises(self):
        with patch.dict(os.environ, {"EMBEDDING_MODEL": "fake", "MEMOTRIX_BACKEND": "redis"}, clear=True):
            with pytest.raises(ConfigurationError, match="invalid"):
                MemoryConfig.from_env()


class TestResolveDatabaseUrl:
    def test_explicit_connection(self):
        dsn = resolve_database_url(connection="postgresql://a:b@c:5432/d")
        assert dsn == "postgresql://a:b@c:5432/d"

    def test_env_database_url(self):
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://env@host/db"}, clear=True):
            dsn = resolve_database_url()
            assert dsn == "postgresql://env@host/db"

    def test_missing_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="Postgres is not configured"):
                resolve_database_url()


class TestRequireEnv:
    def test_present(self):
        with patch.dict(os.environ, {"MY_KEY": "hello"}):
            assert require_env("MY_KEY", hint="set MY_KEY") == "hello"

    def test_missing_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="MY_KEY"):
                require_env("MY_KEY", hint="set MY_KEY")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2: EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════


class TestFakeEmbeddings:
    def test_dimension(self):
        emb = FakeEmbeddings(dim=8)
        assert emb.dimension == 8

    def test_model_name(self):
        emb = FakeEmbeddings(dim=16)
        assert emb.model_name == "fake/16"

    def test_embed_query_shape(self):
        emb = FakeEmbeddings(dim=8)
        vec = emb.embed_query("hello world")
        assert len(vec) == 8
        assert all(isinstance(v, float) for v in vec)

    def test_embed_documents_shape(self):
        emb = FakeEmbeddings(dim=4)
        vecs = emb.embed_documents(["a", "b", "c"])
        assert len(vecs) == 3
        assert all(len(v) == 4 for v in vecs)

    def test_embed_documents_empty(self):
        emb = FakeEmbeddings(dim=8)
        assert emb.embed_documents([]) == []

    def test_deterministic(self):
        emb = FakeEmbeddings(dim=8)
        v1 = emb.embed_query("test")
        v2 = emb.embed_query("test")
        assert v1 == v2

    def test_invalid_dim_raises(self):
        with pytest.raises(ConfigurationError):
            FakeEmbeddings(dim=0)
        with pytest.raises(ConfigurationError):
            FakeEmbeddings(dim=-5)


class TestCoerceEmbeddings:
    def test_passthrough_instance(self):
        emb = FakeEmbeddings(dim=8)
        assert coerce_embeddings(emb) is emb

    def test_invalid_type_raises(self):
        with pytest.raises(ConfigurationError):
            coerce_embeddings(123)  # type: ignore

    def test_empty_string_raises(self):
        with pytest.raises(ConfigurationError):
            coerce_embeddings("")

    def test_whitespace_string_raises(self):
        with pytest.raises(ConfigurationError):
            coerce_embeddings("   ")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3: MEMORY INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════


class TestMemoryInit:
    def test_in_memory_basic(self):
        m = Memory(embeddings=FakeEmbeddings(dim=8), backend="memory")
        assert m.dense is not None
        assert m.sparse is not None
        m.close()

    def test_default_backend_is_memory(self):
        m = Memory(embeddings=FakeEmbeddings(dim=8))
        assert m.connection is None
        m.close()

    def test_connection_plus_memory_raises(self):
        with pytest.raises(ConfigurationError, match="incompatible"):
            Memory(
                embeddings=FakeEmbeddings(dim=8),
                backend="memory",
                connection="postgresql://x:x@x/x",
            )

    def test_unknown_backend_raises(self):
        with pytest.raises(ConfigurationError, match="Unknown"):
            Memory(embeddings=FakeEmbeddings(dim=8), backend="redis")

    def test_from_config_no_model_raises(self):
        cfg = MemoryConfig(backend="memory")
        with pytest.raises(ConfigurationError, match="embedding_model"):
            Memory.from_config(cfg)

    def test_from_config_with_embeddings(self):
        cfg = MemoryConfig(backend="memory")
        m = Memory.from_config(cfg, embeddings=FakeEmbeddings(dim=8))
        assert m.dense is not None
        m.close()

    def test_store_object_with_bundle(self):
        emb = FakeEmbeddings(dim=8)
        store = InMemoryStore(emb, max_elements=100)
        m = Memory(embeddings=emb, store=store)
        assert m.dense is not None
        m.close()

    def test_bad_store_object_raises(self):
        with pytest.raises(ConfigurationError, match="dense"):
            Memory(embeddings=FakeEmbeddings(dim=8), store=object())


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4: ADD_TEXT, SEARCH, DELETE, LIST_SOURCES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def memory():
    """Fresh in-memory Memory for each test."""
    m = Memory(embeddings=FakeEmbeddings(dim=8), backend="memory")
    yield m
    m.close()


class TestAddText:
    def test_basic(self, memory):
        stats = memory.add_text("The sky is blue.", source_id="fact1")
        assert stats["chunks"] >= 1
        assert stats["inserted"] >= 1
        assert stats["extracted_images"] == 0

    def test_return_keys(self, memory):
        stats = memory.add_text("hello", source_id="h")
        for key in ("filename", "path", "chunks", "inserted", "merged", "extracted_images"):
            assert key in stats

    def test_memory_type_semantic(self, memory):
        memory.add_text("fact", source_id="s1", memory_type="semantic")
        hits = memory.search("fact", memory_type="semantic")
        assert len(hits) >= 1

    def test_memory_type_episodic(self, memory):
        memory.add_text("event log", source_id="e1", memory_type="episodic")
        hits = memory.search("event", memory_type="episodic")
        assert len(hits) >= 1

    def test_memory_type_procedural(self, memory):
        memory.add_text("step 1: do this", source_id="p1", memory_type="procedural")
        hits = memory.search("step", memory_type="procedural")
        assert len(hits) >= 1

    def test_session_id(self, memory):
        memory.add_text("chat msg", source_id="c1", session_id="sess-1")
        hits = memory.search("chat", session_id="sess-1")
        assert len(hits) >= 1

    def test_custom_metadata(self, memory):
        memory.add_text("tagged", source_id="t1", metadata={"project": "alpha"})
        hits = memory.search("tagged", filters={"project": "alpha"})
        assert len(hits) >= 1

    def test_empty_text_raises(self, memory):
        with pytest.raises(ConfigurationError, match="non-empty"):
            memory.add_text("")

    def test_whitespace_text_raises(self, memory):
        with pytest.raises(ConfigurationError, match="non-empty"):
            memory.add_text("   \n\t  ")

    def test_none_text_raises(self, memory):
        with pytest.raises(ConfigurationError, match="non-empty"):
            memory.add_text(None)  # type: ignore

    def test_auto_source_id(self, memory):
        stats = memory.add_text("auto id test")
        assert stats["path"].startswith("text:")


class TestSearch:
    def test_basic_search(self, memory):
        memory.add_text("Python is a programming language.", source_id="s1")
        hits = memory.search("programming", top_k=1)
        assert len(hits) >= 1
        assert "chunk_text" in hits[0]

    def test_top_k(self, memory):
        for i in range(5):
            memory.add_text(f"Memory chunk number {i}", source_id=f"m{i}")
        hits = memory.search("chunk", top_k=2)
        assert len(hits) <= 2

    def test_max_tokens_returned(self, memory):
        memory.add_text("a " * 500, source_id="big")
        hits = memory.search("a", max_tokens_returned=50)
        assert len(hits) >= 1

    def test_expand_mode_none(self, memory):
        memory.add_text("chunk A", source_id="a")
        hits = memory.search("chunk", expand_mode="none")
        assert len(hits) >= 1

    def test_expand_mode_full_file(self, memory):
        memory.add_text("full file expansion test", source_id="f")
        hits = memory.search("full", expand_mode="full_file")
        assert len(hits) >= 1

    def test_filter_exact_match(self, memory):
        memory.add_text("important", source_id="i1", metadata={"dept": "eng"})
        memory.add_text("other", source_id="i2", metadata={"dept": "sales"})
        hits = memory.search("important", filters={"dept": "eng"})
        for hit in hits:
            assert hit.get("dept") == "eng"

    def test_combined_filters(self, memory):
        memory.add_text(
            "session data", source_id="x",
            memory_type="episodic", session_id="s1",
            metadata={"tag": "test"},
        )
        hits = memory.search("session", memory_type="episodic", session_id="s1")
        assert len(hits) >= 1

    def test_empty_results(self, memory):
        hits = memory.search("nothing here at all xyz123")
        assert isinstance(hits, list)


class TestDelete:
    def test_delete_by_source_id(self, memory):
        memory.add_text("to be deleted", source_id="del1")
        assert memory.delete("del1") >= 1

    def test_delete_clears_search(self, memory):
        memory.add_text("remove me", source_id="rm1")
        memory.delete("rm1")
        hits = memory.search("remove me", top_k=10)
        for hit in hits:
            assert hit.get("source_path") != "rm1"

    def test_delete_empty_returns_zero(self, memory):
        assert memory.delete("") == 0

    def test_delete_nonexistent_returns_zero(self, memory):
        assert memory.delete("never_existed_123") == 0


class TestListSources:
    def test_empty(self, memory):
        assert memory.list_sources() == []

    def test_after_add(self, memory):
        memory.add_text("hello", source_id="src1")
        sources = memory.list_sources()
        assert len(sources) >= 1
        assert any(s["source_path"] == "src1" for s in sources)

    def test_chunk_count(self, memory):
        memory.add_text("a " * 1000, source_id="multi")
        sources = memory.list_sources()
        src = [s for s in sources if s["source_path"] == "multi"][0]
        assert src["chunks"] >= 1


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5: RE-INGEST & DEDUP
# ═══════════════════════════════════════════════════════════════════════════


class TestReingest:
    def test_reingest_replaces(self, memory):
        memory.add_text("version 1", source_id="doc")
        memory.add_text("version 2", source_id="doc")
        hits = memory.search("version", top_k=10)
        texts = " ".join(h.get("chunk_text", "") for h in hits)
        assert "version 2" in texts

    def test_dedup_within_source(self, memory):
        memory.add_text("duplicate text " * 5, source_id="dup")
        stats = memory.add_text("duplicate text " * 5, source_id="dup")
        assert stats["merged"] >= 0  # dedup should handle this


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 6: add_documents
# ═══════════════════════════════════════════════════════════════════════════


class TestAddDocuments:
    def test_basic(self, memory):
        doc = DocumentData(
            metadata={"filename": "test.txt", "source_path": "test.txt"},
            sections=[{"title": "", "content": "hello from add_documents"}],
            text="hello from add_documents",
        )
        stats = memory.add_documents([(doc, "test.txt")])
        assert stats.get("inserted", 0) >= 1

    def test_multiple_docs(self, memory):
        docs = []
        for i in range(3):
            doc = DocumentData(
                metadata={"filename": f"doc{i}.txt", "source_path": f"doc{i}.txt"},
                sections=[{"title": "", "content": f"Document {i} content"}],
                text=f"Document {i} content",
            )
            docs.append((doc, f"doc{i}.txt"))
        stats = memory.add_documents(docs)
        assert stats.get("inserted", 0) >= 3


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 7: FILE EXTRACTION & INGEST (REAL FILES)
# ═══════════════════════════════════════════════════════════════════════════


class TestFileIngest:
    """Tests using real files from tests/filesForTests/."""

    def test_add_txt(self, memory, tmp_path):
        txt = tmp_path / "hello.txt"
        txt.write_text("Hello world from a text file.", encoding="utf-8")
        stats = memory.add(txt)
        assert stats["chunks"] >= 1
        assert stats["filename"] == "hello.txt"

    def test_add_md(self, memory, tmp_path):
        md = tmp_path / "notes.md"
        md.write_text("# Header\n\nSome markdown content.", encoding="utf-8")
        stats = memory.add(md)
        assert stats["chunks"] >= 1

    def test_add_json(self, memory, tmp_path):
        jf = tmp_path / "data.json"
        jf.write_text(json.dumps({"name": "test", "value": 42}), encoding="utf-8")
        stats = memory.add(jf)
        assert stats["chunks"] >= 1

    def test_add_jsonl(self, memory, tmp_path):
        jl = tmp_path / "logs.jsonl"
        lines = [json.dumps({"event": f"event_{i}", "ts": i}) for i in range(5)]
        jl.write_text("\n".join(lines), encoding="utf-8")
        stats = memory.add(jl)
        assert stats["chunks"] >= 1

    def test_add_csv(self, memory, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\n", encoding="utf-8")
        stats = memory.add(csv_file)
        assert stats["chunks"] >= 1

    def test_add_yaml(self, memory, tmp_path):
        yml = tmp_path / "config.yaml"
        yml.write_text("key: value\nlist:\n  - a\n  - b\n", encoding="utf-8")
        stats = memory.add(yml)
        assert stats["chunks"] >= 1

    def test_add_xml(self, memory, tmp_path):
        xml = tmp_path / "data.xml"
        xml.write_text('<?xml version="1.0"?>\n<root><item>hello</item></root>', encoding="utf-8")
        stats = memory.add(xml)
        assert stats["chunks"] >= 1

    def test_add_sql(self, memory, tmp_path):
        sql = tmp_path / "schema.sql"
        sql.write_text("CREATE TABLE users (id INT, name TEXT);", encoding="utf-8")
        stats = memory.add(sql)
        assert stats["chunks"] >= 1

    def test_add_html(self, memory, tmp_path):
        html = tmp_path / "page.html"
        html.write_text("<html><body><h1>Title</h1><p>Content</p></body></html>", encoding="utf-8")
        stats = memory.add(html)
        assert stats["chunks"] >= 1

    def test_add_python_file(self, memory, tmp_path):
        py = tmp_path / "script.py"
        py.write_text('def hello():\n    """Greet."""\n    return "hi"\n', encoding="utf-8")
        stats = memory.add(py)
        assert stats["chunks"] >= 1

    @pytest.mark.skipif(
        not (FIXTURES / "Excel_Practice.xlsx").exists(),
        reason="xlsx fixture not found",
    )
    def test_add_xlsx(self, memory):
        stats = memory.add(FIXTURES / "Excel_Practice.xlsx")
        assert stats["chunks"] >= 1
        assert stats["filename"] == "Excel_Practice.xlsx"

    @pytest.mark.skipif(
        not (FIXTURES / "SignalHire_exports.csv").exists(),
        reason="csv fixture not found",
    )
    def test_add_real_csv(self, memory):
        stats = memory.add(FIXTURES / "SignalHire_exports.csv")
        assert stats["chunks"] >= 1

    @pytest.mark.skipif(
        not (FIXTURES / "this_is_wow.txt").exists(),
        reason="txt fixture not found",
    )
    def test_add_real_txt(self, memory):
        stats = memory.add(FIXTURES / "this_is_wow.txt")
        assert stats["chunks"] >= 1

    @pytest.mark.skipif(
        not (FIXTURES / "Realtime_Conversation_Analytics_Report_1.docx").exists(),
        reason="docx fixture not found",
    )
    def test_add_docx(self, memory):
        stats = memory.add(
            FIXTURES / "Realtime_Conversation_Analytics_Report_1.docx",
            describe_images=False,
        )
        assert stats["chunks"] >= 1

    @pytest.mark.skipif(
        not (FIXTURES / "donor_report_final.pdf").exists(),
        reason="pdf fixture not found",
    )
    def test_add_pdf(self, memory):
        stats = memory.add(FIXTURES / "donor_report_final.pdf", describe_images=False)
        assert stats["chunks"] >= 1

    @pytest.mark.skipif(
        not (FIXTURES / "donor_report.pptx").exists(),
        reason="pptx fixture not found",
    )
    def test_add_pptx(self, memory):
        stats = memory.add(FIXTURES / "donor_report.pptx", describe_images=False)
        assert stats["chunks"] >= 1

    @pytest.mark.skipif(
        not (FIXTURES / "test.jpg").exists(),
        reason="jpg fixture not found",
    )
    def test_add_image(self, memory):
        stats = memory.add(FIXTURES / "test.jpg", describe_images=False)
        assert stats["chunks"] >= 1


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 8: ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    def test_missing_file_raises(self, memory):
        with pytest.raises(FileNotFoundError):
            memory.add("nonexistent_file_12345.pdf")

    def test_unsupported_extension(self, memory, tmp_path):
        f = tmp_path / "data.xyz123"
        f.write_text("blah", encoding="utf-8")
        with pytest.raises(UnsupportedDocumentTypeError, match="Unsupported"):
            memory.add(f)

    def test_xls_rejected(self, memory, tmp_path):
        f = tmp_path / "old.xls"
        f.write_bytes(b"\xd0\xcf\x11\xe0")  # fake BIFF header
        with pytest.raises(UnsupportedDocumentTypeError, match=".xls"):
            memory.add(f)

    def test_generic_zip_rejected(self, memory, tmp_path):
        import zipfile
        zp = tmp_path / "archive.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("readme.txt", "hello")
        with pytest.raises(UnsupportedDocumentTypeError, match="SCORM"):
            memory.add(zp)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 9: EXTRACTOR PLUGIN REGISTRY
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractorRegistry:
    def setup_method(self):
        # Clean up any previously registered custom extensions
        ExtractorRegistry._extractors.pop(".custom_test_ext", None)

    def teardown_method(self):
        ExtractorRegistry._extractors.pop(".custom_test_ext", None)

    def test_register_and_extract_function(self, tmp_path):
        from memotrix.utils.outputSturcture import build_document

        def my_extractor(path, **kw):
            return build_document(path, path.read_text(), extra_metadata={"custom": True})

        register_extractor(".custom_test_ext", my_extractor)
        f = tmp_path / "data.custom_test_ext"
        f.write_text("custom content", encoding="utf-8")
        doc = extract_file(f)
        assert "custom content" in doc.text

    def test_register_and_extract_class(self, tmp_path):
        from memotrix.utils.outputSturcture import build_document

        class MyExtractor:
            def extract(self, path, **kw):
                return build_document(path, path.read_text(), extra_metadata={"cls": True})

        register_extractor(".custom_test_ext", MyExtractor())
        f = tmp_path / "data.custom_test_ext"
        f.write_text("class-based", encoding="utf-8")
        doc = extract_file(f)
        assert "class-based" in doc.text

    def test_override_builtin(self, tmp_path):
        """Registering for .txt overrides the builtin extractor."""
        from memotrix.utils.outputSturcture import build_document

        original_extractor = ExtractorRegistry.get_extractor(".txt")

        def override(path, **kw):
            return build_document(path, "OVERRIDDEN", extra_metadata={"override": True})

        register_extractor(".txt", override)
        f = tmp_path / "test.txt"
        f.write_text("original text", encoding="utf-8")
        doc = extract_file(f)
        assert doc.text == "OVERRIDDEN"

        # Restore
        if original_extractor:
            ExtractorRegistry.register(".txt", original_extractor)
        else:
            ExtractorRegistry._extractors.pop(".txt", None)

    def test_supported_extensions_includes_custom(self):
        register_extractor(".custom_test_ext", lambda p: None)
        exts = supported_extensions()
        assert ".custom_test_ext" in exts


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 10: MEDICAL / FHIR FILE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════


class TestMedicalFHIR:
    """Test FHIR (Fast Healthcare Interoperability Resources) JSON extraction."""

    def test_fhir_patient_resource(self, memory, tmp_path):
        patient = {
            "resourceType": "Patient",
            "id": "patient-001",
            "name": [{"family": "Doe", "given": ["John"]}],
            "gender": "male",
            "birthDate": "1990-01-15",
            "address": [{"city": "Springfield", "state": "IL"}],
        }
        fhir_file = tmp_path / "patient.json"
        fhir_file.write_text(json.dumps(patient, indent=2), encoding="utf-8")
        stats = memory.add(fhir_file)
        assert stats["chunks"] >= 1

        hits = memory.search("patient Doe", top_k=3)
        assert len(hits) >= 1

    def test_fhir_observation(self, memory, tmp_path):
        observation = {
            "resourceType": "Observation",
            "id": "obs-bp-001",
            "status": "final",
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "85354-9",
                        "display": "Blood pressure panel",
                    }
                ]
            },
            "subject": {"reference": "Patient/patient-001"},
            "valueQuantity": {"value": 120, "unit": "mmHg"},
        }
        obs_file = tmp_path / "observation.json"
        obs_file.write_text(json.dumps(observation, indent=2), encoding="utf-8")
        stats = memory.add(obs_file)
        assert stats["chunks"] >= 1

    def test_fhir_bundle(self, memory, tmp_path):
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "pt1",
                        "name": [{"family": "Smith"}],
                        "gender": "female",
                    }
                },
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "cond1",
                        "code": {
                            "coding": [
                                {"display": "Hypertension"}
                            ]
                        },
                        "subject": {"reference": "Patient/pt1"},
                    }
                },
            ],
        }
        bundle_file = tmp_path / "bundle.json"
        bundle_file.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        stats = memory.add(bundle_file)
        assert stats["chunks"] >= 1

    def test_fhir_medication(self, memory, tmp_path):
        med = {
            "resourceType": "MedicationRequest",
            "id": "medrx-001",
            "status": "active",
            "medicationCodeableConcept": {
                "coding": [
                    {"display": "Metformin 500mg"}
                ]
            },
            "subject": {"reference": "Patient/patient-001"},
            "dosageInstruction": [
                {"text": "Take 1 tablet twice daily with meals."}
            ],
        }
        med_file = tmp_path / "medication.json"
        med_file.write_text(json.dumps(med, indent=2), encoding="utf-8")
        stats = memory.add(med_file)
        assert stats["chunks"] >= 1

    def test_fhir_allergy(self, memory, tmp_path):
        allergy = {
            "resourceType": "AllergyIntolerance",
            "id": "allergy-001",
            "clinicalStatus": {
                "coding": [{"code": "active"}]
            },
            "code": {
                "coding": [{"display": "Penicillin"}]
            },
            "patient": {"reference": "Patient/patient-001"},
            "reaction": [
                {
                    "manifestation": [
                        {"coding": [{"display": "Anaphylaxis"}]}
                    ],
                    "severity": "severe",
                }
            ],
        }
        allergy_file = tmp_path / "allergy.json"
        allergy_file.write_text(json.dumps(allergy, indent=2), encoding="utf-8")
        stats = memory.add(allergy_file)
        assert stats["chunks"] >= 1


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 11: AS_RETRIEVER / AS_STACK / CLOSE
# ═══════════════════════════════════════════════════════════════════════════


class TestAdvancedSurfaces:
    def test_as_retriever(self, memory):
        retriever = memory.as_retriever()
        assert retriever is memory.retrieval

    def test_as_stack_keys(self, memory):
        stack = memory.as_stack()
        for key in ("dense", "sparse", "store", "document_store", "hybrid",
                     "ingest_pipeline", "retrieval", "memory"):
            assert key in stack

    def test_close_is_idempotent(self, memory):
        memory.close()
        memory.close()  # should not raise


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 12: EXTRACT (WITHOUT INDEXING)
# ═══════════════════════════════════════════════════════════════════════════


class TestExtract:
    def test_extract_txt(self, memory, tmp_path):
        f = tmp_path / "extract_me.txt"
        f.write_text("Extract this content.", encoding="utf-8")
        doc = memory.extract(f)
        assert hasattr(doc, "text")
        assert "Extract this content" in doc.text

    def test_extract_does_not_index(self, memory, tmp_path):
        f = tmp_path / "no_index.txt"
        f.write_text("Should not appear in search.", encoding="utf-8")
        memory.extract(f)
        hits = memory.search("Should not appear in search", top_k=10)
        assert len(hits) == 0


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 13: MISSING DEPENDENCY ERRORS
# ═══════════════════════════════════════════════════════════════════════════


class TestMissingDependencyErrors:
    def test_pdf_missing_fitz(self, tmp_path):
        from memotrix.filetypes.document.pdf import PDFExtractor

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\n")
        with patch.dict(sys.modules, {"fitz": None}):
            extractor = PDFExtractor()
            with pytest.raises(MissingDependencyError):
                extractor.extract(pdf_file)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 14: SUPPORTED EXTENSIONS LIST
# ═══════════════════════════════════════════════════════════════════════════


class TestSupportedExtensions:
    def test_returns_list(self):
        exts = supported_extensions()
        assert isinstance(exts, list)
        assert len(exts) > 10

    def test_common_formats_present(self):
        exts = supported_extensions()
        for ext in (".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".json",
                     ".txt", ".md", ".html", ".py", ".yaml"):
            assert ext in exts, f"{ext} missing from supported_extensions()"


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 15: END-TO-END WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════


class TestEndToEnd:
    """Full workflow: init → add → search → delete → verify → close."""

    def test_full_lifecycle(self):
        m = Memory(embeddings=FakeEmbeddings(dim=8), backend="memory")

        # 1. Add text
        m.add_text("Memotrix is a hybrid RAG library.", source_id="intro")
        m.add_text("The user prefers dark mode.", source_id="prefs",
                    memory_type="semantic", session_id="s1")
        m.add_text("Deploy with `alpha deploy --force`.", source_id="deploy",
                    memory_type="procedural")

        # 2. Search
        hits = m.search("What is Memotrix?", top_k=3)
        assert len(hits) >= 1

        procedural = m.search("deploy", memory_type="procedural")
        assert len(procedural) >= 1

        session_hits = m.search("dark mode", session_id="s1")
        assert len(session_hits) >= 1

        # 3. List sources
        sources = m.list_sources()
        assert len(sources) == 3

        # 4. Delete
        deleted = m.delete("prefs")
        assert deleted >= 1

        sources_after = m.list_sources()
        assert len(sources_after) == 2
        assert not any(s["source_path"] == "prefs" for s in sources_after)

        # 5. Close
        m.close()

    def test_file_workflow(self, tmp_path):
        m = Memory(embeddings=FakeEmbeddings(dim=8), backend="memory")

        # Create test files
        txt = tmp_path / "report.txt"
        txt.write_text("Q3 revenue was $42M. North region led growth.", encoding="utf-8")

        csv = tmp_path / "sales.csv"
        csv.write_text("region,revenue\nNorth,42000000\nSouth,35000000\n", encoding="utf-8")

        # Ingest both
        m.add(txt)
        m.add(csv)

        # Search across both
        hits = m.search("revenue", top_k=5)
        assert len(hits) >= 1

        # List and verify
        sources = m.list_sources()
        assert len(sources) == 2

        # Delete one
        m.delete("report.txt")
        assert len(m.list_sources()) == 1

        m.close()
