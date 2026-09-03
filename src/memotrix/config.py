"""Explicit configuration objects — no silent credentials, models, or dimensions."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from memotrix.utils.exceptions import ConfigurationError

BackendName = Literal["memory", "postgres"]


def _env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def require_env(name: str, *, hint: str) -> str:
    value = _env(name)
    if not value:
        raise ConfigurationError(f"{name} is not set. {hint}")
    return value


def resolve_database_url(*, connection: Optional[str] = None) -> str:
    """
    Resolve a Postgres DSN from an explicit value or environment.

    Never invents credentials. Either pass ``connection`` or set ``DATABASE_URL``,
    or provide every discrete part (user, password, host, port, database).
    """
    if connection and connection.strip():
        return connection.strip()

    url = _env("DATABASE_URL")
    if url:
        return url

    dialect = _env("DATABASE_DIALECT") or "postgresql"
    user = _env("DATABASE_USER") or _env("POSTGRES_USER")
    password = _env("DATABASE_PASSWORD") or _env("POSTGRES_PASSWORD")
    host = _env("DATABASE_HOST") or _env("POSTGRES_HOST")
    port = _env("DATABASE_PORT") or _env("POSTGRES_PORT")
    db = _env("DATABASE_NAME") or _env("POSTGRES_DB")

    missing = [
        label
        for label, value in (
            ("DATABASE_USER/POSTGRES_USER", user),
            ("DATABASE_PASSWORD/POSTGRES_PASSWORD", password),
            ("DATABASE_HOST/POSTGRES_HOST", host),
            ("DATABASE_PORT/POSTGRES_PORT", port),
            ("DATABASE_NAME/POSTGRES_DB", db),
        )
        if not value
    ]
    if missing:
        raise ConfigurationError(
            "Postgres is not configured. Pass connection=... to Memory / PostgresStore, "
            "set DATABASE_URL, or set all of: "
            + ", ".join(missing)
        )
    return f"{dialect}://{user}:{password}@{host}:{port}/{db}"


@dataclass
class ChunkingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class RetrievalConfig:
    top_k: int = 3
    max_tokens: int = 800
    expand_mode: str = "neighbors"
    neighbor_window: int = 1
    enable_rerank: bool = True
    reranker_model: Optional[str] = None
    enable_memory_boost: bool = True
    fusion_k: int = 60
    query_cache_size: int = 256


@dataclass
class IngestConfig:
    enable_dedup: bool = True
    dedup_threshold: float = 0.95
    describe_images: bool = True


@dataclass
class MemoryConfig:
    """
    All tunables for a Memory instance.

    Embeddings (model + dimension) and the vector store connection are *not*
    defaulted here — pass them to ``Memory(...)`` or ``Memory.from_env()``.
    """

    backend: BackendName = "memory"
    connection: Optional[str] = None
    table_name: str = "memotrix_chunks"
    max_elements: int = 50_000
    embedding_model: Optional[str] = None
    reranker_model: Optional[str] = None
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """
        Load from environment. Missing embedding model or (for postgres) DSN
        raises ConfigurationError instead of falling back to baked-in secrets.
        """
        backend_raw = (_env("MEMOTRIX_BACKEND") or "memory").lower()
        if backend_raw not in ("memory", "postgres"):
            raise ConfigurationError(
                f"MEMOTRIX_BACKEND={backend_raw!r} is invalid; use 'memory' or 'postgres'"
            )
        backend: BackendName = backend_raw  # type: ignore[assignment]
        embedding_model = require_env(
            "EMBEDDING_MODEL",
            hint="Example: EMBEDDING_MODEL=BAAI/bge-small-en-v1.5",
        )
        connection = None
        if backend == "postgres":
            connection = resolve_database_url()

        retrieval = RetrievalConfig(reranker_model=_env("RERANKER_MODEL"))
        return cls(
            backend=backend,
            connection=connection,
            table_name=_env("MEMOTRIX_TABLE") or "memotrix_chunks",
            embedding_model=embedding_model,
            reranker_model=_env("RERANKER_MODEL"),
            retrieval=retrieval,
        )
