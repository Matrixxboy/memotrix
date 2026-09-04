import json
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row

from .base import DenseIndex, PayloadFilters, SparseIndex, is_filename_only
from memotrix.utils.exceptions import ConfigurationError

DEFAULT_TABLE = "memotrix_chunks"


def _filter_json_text(value: Any) -> str:
    """JSONB ``->>`` text form so bool/number filters match stored JSON."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    return json.dumps(value)


def _filter_clauses(filters: PayloadFilters) -> Tuple[List[sql.Composable], List[Any]]:
    clauses: List[sql.Composable] = []
    params: List[Any] = []
    if not filters:
        return clauses, params
    for key, value in filters.items():
        if value is None:
            continue
        clauses.append(sql.SQL("payload->>%s = %s"))
        params.extend([str(key), _filter_json_text(value)])
    return clauses, params


def _source_match_clause(source_path: str) -> Tuple[sql.Composable, List[Any]]:
    if is_filename_only(source_path):
        clause = sql.SQL(
            """(
                payload->>'source_path' = %s
                OR payload->>'filename' = %s
                OR regexp_replace(
                    coalesce(payload->>'source_path', ''),
                    '^.*[/\\\\]',
                    ''
                ) = %s
            )"""
        )
        return clause, [source_path, source_path, source_path]
    clause = sql.SQL(
        "(payload->>'source_path' = %s OR payload->>'filename' = %s)"
    )
    return clause, [source_path, source_path]


class PostgresChunkStore:
    """PostgreSQL + pgvector storage for chunk embeddings and full-text search."""

    def __init__(
        self,
        connection: str,
        *,
        dim: int,
        table_name: str = DEFAULT_TABLE,
        dsn: str | None = None,
    ) -> None:
        conn = (connection or dsn or "").strip()
        if not conn:
            raise ConfigurationError(
                "PostgresChunkStore requires an explicit connection string"
            )
        if dim <= 0:
            raise ConfigurationError(
                "PostgresChunkStore requires dim from the embedding model "
                "(embeddings.dimension), not a hardcoded value"
            )
        self.dsn = conn
        self.dim = dim
        self.table_name = table_name
        self._conn: psycopg.Connection | None = None

    def connect(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, autocommit=True)
            register_vector(self._conn)
            self._ensure_schema()
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None

    def _ensure_schema(self) -> None:
        conn = self._conn
        assert conn is not None

        if not self.table_name.isidentifier():
            raise ValueError(f"Invalid table name: {self.table_name}")

        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id TEXT PRIMARY KEY,
                chunk_text TEXT NOT NULL,
                embedding vector({self.dim}),
                payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                search_vector tsvector GENERATED ALWAYS AS (
                    to_tsvector('english', coalesce(chunk_text, ''))
                ) STORED,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )

        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_hnsw_idx
            ON {self.table_name}
            USING hnsw (embedding vector_cosine_ops)
            """
        )

        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_search_gin_idx
            ON {self.table_name}
            USING GIN (search_vector)
            """
        )

        # Helpful for metadata filters used by agent memory.
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_payload_gin_idx
            ON {self.table_name}
            USING GIN (payload jsonb_path_ops)
            """
        )

    def upsert(
        self,
        chunk_id: str,
        chunk_text: str,
        payload: Dict[str, Any],
        embedding: Optional[List[float]] = None,
    ) -> None:
        conn = self.connect()

        if embedding is not None:
            conn.execute(
                sql.SQL(
                    """
                    INSERT INTO {table} (id, chunk_text, embedding, payload, updated_at)
                    VALUES (%s, %s, %s, %s::jsonb, now())
                    ON CONFLICT (id) DO UPDATE SET
                        chunk_text = EXCLUDED.chunk_text,
                        embedding = EXCLUDED.embedding,
                        payload = EXCLUDED.payload,
                        updated_at = now()
                    """
                ).format(table=sql.Identifier(self.table_name)),
                (chunk_id, chunk_text, embedding, json.dumps(payload)),
            )
            return

        conn.execute(
            sql.SQL(
                """
                INSERT INTO {table} (id, chunk_text, payload, updated_at)
                VALUES (%s, %s, %s::jsonb, now())
                ON CONFLICT (id) DO UPDATE SET
                    chunk_text = EXCLUDED.chunk_text,
                    payload = EXCLUDED.payload,
                    updated_at = now()
                """
            ).format(table=sql.Identifier(self.table_name)),
            (chunk_id, chunk_text, json.dumps(payload)),
        )

    def update_payload(self, chunk_id: str, payload: Dict[str, Any]) -> None:
        conn = self.connect()
        conn.execute(
            sql.SQL(
                """
                UPDATE {table}
                SET payload = %s::jsonb, updated_at = now()
                WHERE id = %s
                """
            ).format(table=sql.Identifier(self.table_name)),
            (json.dumps(payload), chunk_id),
        )

    def delete_by_source_path(self, source_path: str) -> int:
        conn = self.connect()
        where, params = _source_match_clause(source_path)
        result = conn.execute(
            sql.SQL("DELETE FROM {table} WHERE {where}").format(
                table=sql.Identifier(self.table_name),
                where=where,
            ),
            params,
        )
        return result.rowcount or 0

    def get_payload(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        conn = self.connect()
        row = conn.execute(
            sql.SQL("SELECT payload FROM {table} WHERE id = %s").format(
                table=sql.Identifier(self.table_name)
            ),
            (chunk_id,),
        ).fetchone()
        if not row:
            return None
        return dict(row[0] or {})

    def payloads_for_source(self, source_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        conn = self.connect()
        where, params = _source_match_clause(source_path)
        rows = conn.execute(
            sql.SQL("SELECT id, payload FROM {table} WHERE {where}").format(
                table=sql.Identifier(self.table_name),
                where=where,
            ),
            params,
        ).fetchall()
        matches = [(row[0], dict(row[1] or {})) for row in rows]
        matches.sort(
            key=lambda item: (
                item[1].get("row_start")
                if item[1].get("row_start") is not None
                else item[1].get("chunk_index")
                if item[1].get("chunk_index") is not None
                else 10**9,
                item[0],
            )
        )
        return matches

    def iter_payloads(self):
        conn = self.connect()
        rows = conn.execute(
            sql.SQL(
                """
                SELECT id, payload, chunk_text
                FROM {table}
                """
            ).format(table=sql.Identifier(self.table_name))
        ).fetchall()
        for row in rows:
            payload = dict(row[1] or {})
            if not payload.get("chunk_text") and row[2]:
                payload["chunk_text"] = row[2]
            yield row[0], payload

    def search_dense(
        self,
        query_vector: List[float],
        k: int,
        filters: PayloadFilters = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        conn = self.connect()
        filter_sql, filter_params = _filter_clauses(filters)
        where = sql.SQL("WHERE embedding IS NOT NULL")
        if filter_sql:
            where = sql.SQL("{} AND {}").format(
                where, sql.SQL(" AND ").join(filter_sql)
            )

        query = sql.SQL(
            """
            SELECT
                id,
                1 - (embedding <=> %s::vector) AS score,
                payload
            FROM {table}
            {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
        ).format(table=sql.Identifier(self.table_name), where=where)

        params = [query_vector, *filter_params, query_vector, k]
        rows = conn.execute(query, params).fetchall()
        return [(row[0], float(row[1]), dict(row[2])) for row in rows]

    def search_sparse(
        self,
        query_text: str,
        k: int,
        filters: PayloadFilters = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        conn = self.connect()
        filter_sql, filter_params = _filter_clauses(filters)
        where = sql.SQL(
            "WHERE search_vector @@ websearch_to_tsquery('english', %s)"
        )
        if filter_sql:
            where = sql.SQL("{} AND {}").format(
                where, sql.SQL(" AND ").join(filter_sql)
            )

        query = sql.SQL(
            """
            SELECT
                id,
                ts_rank_cd(search_vector, websearch_to_tsquery('english', %s)) AS score,
                payload
            FROM {table}
            {where}
            ORDER BY score DESC
            LIMIT %s
            """
        ).format(table=sql.Identifier(self.table_name), where=where)

        params = [query_text, query_text, *filter_params, k]
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return [
            (row["id"], float(row["score"]), dict(row["payload"]))
            for row in rows
            if float(row["score"]) > 0.0
        ]

        if results:
            return results

        # Fallback to plainto_tsquery if websearch_to_tsquery returned 0 hits
        where_plain = sql.SQL(
            "WHERE search_vector @@ plainto_tsquery('english', %s)"
        )
        if filter_sql:
            where_plain = sql.SQL("{} AND {}").format(
                where_plain, sql.SQL(" AND ").join(filter_sql)
            )

        query_plain = sql.SQL(
            """
            SELECT
                id,
                ts_rank_cd(search_vector, plainto_tsquery('english', %s)) AS score,
                payload
            FROM {table}
            {where}
            ORDER BY score DESC
            LIMIT %s
            """
        ).format(table=sql.Identifier(self.table_name), where=where_plain)

        params_plain = [query_text, query_text, *filter_params, k]
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query_plain, params_plain)
            rows_plain = cur.fetchall()

        return [
            (row["id"], float(row["score"]), dict(row["payload"]))
            for row in rows_plain
            if float(row["score"]) > 0.0
        ]

    def clear(self) -> None:
        conn = self.connect()
        conn.execute(
            sql.SQL("TRUNCATE TABLE {table}").format(
                table=sql.Identifier(self.table_name)
            )
        )


class PgDenseIndex(DenseIndex):
    """Dense vector search backed by pgvector."""

    def __init__(self, store: PostgresChunkStore) -> None:
        self.store = store

    def add(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> None:
        for chunk_id, vector, payload in zip(ids, vectors, payloads):
            chunk_text = payload.get("chunk_text", "")
            self.store.upsert(chunk_id, chunk_text, payload, embedding=vector)

    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        filters: PayloadFilters = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        return self.store.search_dense(query_vector, k, filters=filters)

    def delete_by_source_path(self, source_path: str) -> int:
        return self.store.delete_by_source_path(source_path)

    def update_payload(self, chunk_id: str, payload: Dict[str, Any]) -> None:
        self.store.update_payload(chunk_id, payload)

    def get_payload(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        return self.store.get_payload(chunk_id)

    def payloads_for_source(self, source_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        return self.store.payloads_for_source(source_path)

    def iter_payloads(self):
        yield from self.store.iter_payloads()

    def save(self, path: str) -> None:
        """Persistence is handled by PostgreSQL."""

    def load(self, path: str) -> None:
        """Persistence is handled by PostgreSQL."""


class PgSparseIndex(SparseIndex):
    """Keyword search backed by PostgreSQL full-text search."""

    def __init__(self, store: PostgresChunkStore) -> None:
        self.store = store

    def add(
        self,
        ids: List[str],
        texts: List[str],
        payloads: List[Dict[str, Any]],
    ) -> None:
        for chunk_id, text, payload in zip(ids, texts, payloads):
            self.store.upsert(chunk_id, text, payload)

    def search(
        self,
        query_text: str,
        k: int = 10,
        filters: PayloadFilters = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        return self.store.search_sparse(query_text, k, filters=filters)

    def delete_by_source_path(self, source_path: str) -> int:
        # Dense path already deletes rows; sparse shares the same table.
        return 0

    def update_payload(self, chunk_id: str, payload: Dict[str, Any]) -> None:
        self.store.update_payload(chunk_id, payload)

    def get_payload(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        return self.store.get_payload(chunk_id)

    def payloads_for_source(self, source_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        return self.store.payloads_for_source(source_path)

    def iter_payloads(self):
        yield from self.store.iter_payloads()

    def save(self, path: str) -> None:
        """Persistence is handled by PostgreSQL."""

    def load(self, path: str) -> None:
        """Persistence is handled by PostgreSQL."""


def create_postgres_indexes(
    connection: str,
    *,
    dim: int,
    table_name: str = DEFAULT_TABLE,
    dsn: str | None = None,
) -> tuple[PgDenseIndex, PgSparseIndex, PostgresChunkStore]:
    store = PostgresChunkStore(
        connection=connection or dsn or "",
        dim=dim,
        table_name=table_name,
    )
    store.connect()
    return PgDenseIndex(store), PgSparseIndex(store), store
