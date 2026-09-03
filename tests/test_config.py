"""Framework config: no silent DSN, dim, or embedding-model defaults."""

import pytest

from memotrix import Memory, MemoryConfig
from memotrix.embeddings import FakeEmbeddings, HuggingFaceEmbeddings
from memotrix.utils.exceptions import ConfigurationError
from memotrix.vectorDB.hnsw_index import HNSWDenseIndex
from memotrix.vectorDB.postgres_store import PostgresChunkStore
from memotrix.vectorstores import InMemoryStore, PostgresStore


def test_hnsw_requires_dimension():
    with pytest.raises(TypeError):
        HNSWDenseIndex()  # type: ignore[call-arg]
    idx = HNSWDenseIndex(dim=8)
    assert idx.dim == 8


def test_postgres_store_requires_connection():
    embeddings = FakeEmbeddings(dim=8)
    with pytest.raises(ConfigurationError, match="connection string"):
        PostgresStore(connection="", embeddings=embeddings)
    with pytest.raises(ConfigurationError, match="connection string"):
        PostgresChunkStore(connection="", dim=8)


def test_postgres_store_requires_positive_dim():
    with pytest.raises(ConfigurationError, match="dim"):
        PostgresChunkStore(connection="postgresql://u:p@localhost:5432/db", dim=0)


def test_huggingface_requires_model_name():
    with pytest.raises(ConfigurationError, match="model name"):
        HuggingFaceEmbeddings(model="")


def test_fake_embeddings_dimension_is_explicit():
    embeddings = FakeEmbeddings(dim=12)
    assert embeddings.dimension == 12
    assert len(embeddings.embed_query("hello")) == 12
    docs = embeddings.embed_documents(["a", "bb"])
    assert len(docs) == 2
    assert len(docs[0]) == 12


def test_memory_in_memory_infers_dim(monkeypatch):
    embeddings = FakeEmbeddings(dim=6)
    memory = Memory(embeddings=embeddings, backend="memory")
    assert memory.dense.dim == 6
    stack = memory.as_stack()
    assert stack["retrieval"] is memory.retrieval
    memory.close()


def test_memory_postgres_without_dsn_raises(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for key in (
        "DATABASE_USER",
        "POSTGRES_USER",
        "DATABASE_PASSWORD",
        "POSTGRES_PASSWORD",
        "DATABASE_HOST",
        "POSTGRES_HOST",
        "DATABASE_PORT",
        "POSTGRES_PORT",
        "DATABASE_NAME",
        "POSTGRES_DB",
    ):
        monkeypatch.delenv(key, raising=False)
    embeddings = FakeEmbeddings(dim=4)
    with pytest.raises(ConfigurationError, match="Postgres is not configured"):
        Memory(embeddings=embeddings, backend="postgres")


def test_from_env_requires_embedding_model(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("MEMOTRIX_BACKEND", "memory")
    with pytest.raises(ConfigurationError, match="EMBEDDING_MODEL"):
        MemoryConfig.from_env()


def test_from_env_postgres_requires_database_url(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "fake-model")
    monkeypatch.setenv("MEMOTRIX_BACKEND", "postgres")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for key in (
        "DATABASE_USER",
        "POSTGRES_USER",
        "DATABASE_PASSWORD",
        "POSTGRES_PASSWORD",
        "DATABASE_HOST",
        "POSTGRES_HOST",
        "DATABASE_PORT",
        "POSTGRES_PORT",
        "DATABASE_NAME",
        "POSTGRES_DB",
    ):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ConfigurationError, match="Postgres is not configured"):
        MemoryConfig.from_env()


def test_no_default_password_in_resolver(monkeypatch):
    from memotrix.config import resolve_database_url

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_USER", "alice")
    monkeypatch.delenv("DATABASE_PASSWORD", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    monkeypatch.setenv("DATABASE_HOST", "db.internal")
    monkeypatch.setenv("DATABASE_PORT", "5432")
    monkeypatch.setenv("DATABASE_NAME", "app")
    with pytest.raises(ConfigurationError, match="PASSWORD"):
        resolve_database_url()


def test_in_memory_store_uses_embedding_dim():
    embeddings = FakeEmbeddings(dim=7)
    store = InMemoryStore(embeddings, max_elements=50)
    assert store.dense.dim == 7


def test_postgres_bool_filter_json_text():
    from memotrix.vectorDB.postgres_store import _filter_json_text

    assert _filter_json_text(True) == "true"
    assert _filter_json_text(False) == "false"
    assert _filter_json_text(1) == "1"
    assert _filter_json_text("keep") == "keep"
