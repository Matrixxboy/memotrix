# API Reference: Vector Stores

Vector Stores manage the storage and retrieval of Dense (vector) embeddings, Sparse (keyword) weights, and raw metadata payloads.

Memotrix bundles these capabilities into a `VectorStoreBundle` interface, which exposes `.dense` and `.sparse` properties.

## `InMemoryStore`

The `InMemoryStore` uses the `hnswlib` C++ library for lightning-fast, in-process approximate nearest neighbors search (Dense). It uses a custom `rank-bm25` implementation for sparse keyword search.

```python
from memotrix.vectorstores import InMemoryStore
```

### Initialization

```python
InMemoryStore(
    embeddings: Embeddings,
    max_elements: int = 50000,
    space: str = "cosine"
)
```

**Parameters:**
- `embeddings`: Required. The active embeddings instance (determines the vector dimension).
- `max_elements`: The maximum number of chunks the HNSW index can hold. It **cannot** be resized dynamically after initialization. Defaults to 50,000.
- `space`: The distance metric. Defaults to `"cosine"`.

**Persistence:**
The `InMemoryStore` is entirely volatile. When the Python process exits, all data is lost. 

---

## `PostgresStore`

The `PostgresStore` provides persistent, ACID-compliant, scalable storage using PostgreSQL and the `pgvector` extension. 

```python
from memotrix.vectorstores import PostgresStore
```

### Initialization

```python
PostgresStore(
    connection: str,
    embeddings: Embeddings,
    table_name: str = "memotrix_chunks"
)
```

**Parameters:**
- `connection`: Required. A valid PostgreSQL Data Source Name (DSN) string. (e.g. `postgresql://user:pass@localhost/db`).
- `embeddings`: Required. The active embeddings instance. The `vector` column in Postgres will be sized exactly to `embeddings.dimension`.
- `table_name`: The name of the table to create/use. Defaults to `"memotrix_chunks"`.

### Schema

Upon initialization, `PostgresStore` ensures the `pgvector` extension is installed and executes a `CREATE TABLE IF NOT EXISTS` statement.

The schema looks roughly like this:
```sql
CREATE TABLE IF NOT EXISTS memotrix_chunks (
    id UUID PRIMARY KEY,
    source_path TEXT,
    chunk_index INTEGER,
    chunk_text TEXT,
    embedding vector({dim}),
    metadata JSONB,
    sparse_vector tsvector
);
```

It also automatically builds an HNSW index on the `embedding` column and a GIN index on the `sparse_vector` column to ensure blazing fast retrieval even with millions of rows.

### `from_conn_string()`

Alternatively, you can initialize it using the classmethod:
```python
store = PostgresStore.from_conn_string(
    "postgresql://localhost/db",
    embeddings=my_embeddings
)
```
