from .base import DenseIndex, SparseIndex
from .hybrid import HybridSearchEngine, reciprocal_rank_fusion

# Postgres backends are optional — import lazily so memory-only boots work
# without psycopg installed.


def __getattr__(name: str):
    if name in {
        "PgDenseIndex",
        "PgSparseIndex",
        "PostgresChunkStore",
        "create_postgres_indexes",
    }:
        from . import postgres_store

        return getattr(postgres_store, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DenseIndex",
    "SparseIndex",
    "PgDenseIndex",
    "PgSparseIndex",
    "PostgresChunkStore",
    "create_postgres_indexes",
    "HybridSearchEngine",
    "reciprocal_rank_fusion",
]
