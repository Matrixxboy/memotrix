"""
Stable public facade for external consumers.

Prefer the LangChain-style API::

    from memotrix import Memory
    from memotrix.embeddings import HuggingFaceEmbeddings

    memory = Memory(
        embeddings=HuggingFaceEmbeddings(model="..."),
        backend="postgres",
        connection=dsn,
    )
"""

from __future__ import annotations

from typing import Any, Optional, Union

from memotrix.config import MemoryConfig
from memotrix.embeddings import Embeddings, HuggingFaceEmbeddings
from memotrix.memory import Memory
from memotrix.processes.document_store import DocumentStore, estimate_tokens
from memotrix.processes.ingestion import IngestionPipeline
from memotrix.processes.retrieval import RetrievalPipeline
from memotrix.utils.ai_integration import (
    OpenAIProvider,
    OpenRouterProvider,
    configure_ai_provider,
    get_ai_router,
)
from memotrix.utils.exceptions import ConfigurationError
from memotrix.vectorDB.hnsw_index import HNSWDenseIndex
from memotrix.vectorDB.hybrid import HybridSearchEngine
from memotrix.vectorDB.sparse_index import BM25SparseIndex
from memotrix.vectorstores import InMemoryStore, PostgresStore

BackendName = str
RetrievalStack = dict


def build_retrieval_stack(
    backend: str = "memory",
    *,
    embeddings: Union[Embeddings, str, None] = None,
    embedding_model: Optional[str] = None,
    connection: Optional[str] = None,
    config: Optional[MemoryConfig] = None,
    custom_dense: Any = None,
    custom_sparse: Any = None,
    expand_mode: Optional[str] = None,
    max_elements: Optional[int] = None,
) -> dict:
    """
    Compose a retrieval stack.

    ``embeddings`` (instance or model name) is required. Dimension is inferred
    from the model. Postgres requires ``connection`` or DATABASE_URL.
    """
    cfg = config or MemoryConfig(backend=backend if backend in ("memory", "postgres") else "memory")
    if expand_mode:
        cfg.retrieval.expand_mode = expand_mode
    if max_elements is not None:
        cfg.max_elements = max_elements

    model_or_emb = embeddings or embedding_model or cfg.embedding_model
    if model_or_emb is None:
        raise ConfigurationError(
            "build_retrieval_stack requires embeddings=HuggingFaceEmbeddings(model=...) "
            "or embedding_model='org/model'"
        )

    store = None
    if custom_dense is not None and custom_sparse is not None:
        class _Custom:
            dense = custom_dense
            sparse = custom_sparse
            native = None
            bundle = None

        store = _Custom()

    if connection and store is None and backend == "memory":
        backend = "postgres"
        cfg.backend = "postgres"

    memory = Memory(
        embeddings=model_or_emb,
        store=store,
        connection=connection,
        backend=backend,
        config=cfg,
    )
    return memory.as_stack()


__all__ = [
    "BackendName",
    "RetrievalStack",
    "build_retrieval_stack",
    "Memory",
    "MemoryConfig",
    "RetrievalPipeline",
    "HybridSearchEngine",
    "DocumentStore",
    "IngestionPipeline",
    "HNSWDenseIndex",
    "BM25SparseIndex",
    "PostgresStore",
    "InMemoryStore",
    "HuggingFaceEmbeddings",
    "estimate_tokens",
    "OpenAIProvider",
    "OpenRouterProvider",
    "configure_ai_provider",
    "get_ai_router",
    "create_postgres_indexes",
]


def __getattr__(name: str):
    if name == "create_postgres_indexes":
        from memotrix.vectorDB.postgres_store import create_postgres_indexes

        return create_postgres_indexes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
