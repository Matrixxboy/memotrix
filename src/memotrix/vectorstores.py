"""Vector store constructors. Connection and dimension are always explicit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from memotrix.embeddings import Embeddings
from memotrix.utils.exceptions import ConfigurationError
from memotrix.vectorDB.hnsw_index import HNSWDenseIndex
from memotrix.vectorDB.sparse_index import BM25SparseIndex


@dataclass
class VectorStoreBundle:
    dense: Any
    sparse: Any
    native: Any = None

    def close(self) -> None:
        native = self.native
        closer = getattr(native, "close", None)
        if callable(closer):
            closer()


class InMemoryStore:
    """HNSW + BM25 in-process store. Dimension comes from ``embeddings``."""

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        max_elements: int = 50_000,
        space: str = "cosine",
    ) -> None:
        dim = embeddings.dimension
        self.embeddings = embeddings
        self.bundle = VectorStoreBundle(
            dense=HNSWDenseIndex(dim=dim, max_elements=max_elements, space=space),
            sparse=BM25SparseIndex(),
            native=None,
        )

    @property
    def dense(self):
        return self.bundle.dense

    @property
    def sparse(self):
        return self.bundle.sparse

    def close(self) -> None:
        self.bundle.close()


class PostgresStore:
    """
    PostgreSQL + pgvector store.

    Requires an explicit connection string. Vector width is taken from
    ``embeddings.dimension`` so the table matches the model.
    """

    def __init__(
        self,
        connection: str,
        embeddings: Embeddings,
        *,
        table_name: str = "memotrix_chunks",
    ) -> None:
        if not connection or not str(connection).strip():
            raise ConfigurationError(
                "PostgresStore requires an explicit connection string "
                "(pass connection= or set DATABASE_URL)"
            )
        from memotrix.vectorDB.postgres_store import create_postgres_indexes

        dim = embeddings.dimension
        dense, sparse, store = create_postgres_indexes(
            connection=str(connection).strip(),
            dim=dim,
            table_name=table_name,
        )
        self.embeddings = embeddings
        self.connection = str(connection).strip()
        self.bundle = VectorStoreBundle(dense=dense, sparse=sparse, native=store)

    @classmethod
    def from_conn_string(
        cls,
        connection: str,
        embeddings: Embeddings,
        *,
        table_name: str = "memotrix_chunks",
    ) -> "PostgresStore":
        return cls(connection=connection, embeddings=embeddings, table_name=table_name)

    @property
    def dense(self):
        return self.bundle.dense

    @property
    def sparse(self):
        return self.bundle.sparse

    def close(self) -> None:
        self.bundle.close()
