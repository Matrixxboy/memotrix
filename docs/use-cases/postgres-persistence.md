---
title: "PostgreSQL persistence"
---

# Use case: persist memory in PostgreSQL

In-memory HNSW/BM25 dies with the process. `PostgresStore` writes chunks to PostgreSQL + pgvector.

```bash
pip install "memotrix[postgres,memory]"
```

You must supply a DSN. Example (use **your** credentials, not a library default):

```python
import os
from memotrix import Memory
from memotrix.embeddings import HuggingFaceEmbeddings
from memotrix.vectorstores import PostgresStore

embeddings = HuggingFaceEmbeddings(model=os.environ["EMBEDDING_MODEL"])
store = PostgresStore(
    connection=os.environ["DATABASE_URL"],
    embeddings=embeddings,
    table_name="memotrix_chunks",
)
memory = Memory(embeddings=embeddings, store=store)
```

`connection=` on `Memory` without `backend="memory"` selects Postgres automatically.

`Memory.from_env()` requires `EMBEDDING_MODEL` and, when `MEMOTRIX_BACKEND=postgres`, a resolvable DSN.

## Schema (created if missing)

The store enables `pgvector` and creates a table sized to `embeddings.dimension`, with HNSW on the vector column and GIN on `tsvector`. If you change embedding width, use a new `table_name` or drop the old table.

## Docker

Repo `docker-compose.yml` image: `pgvector/pgvector:pg16`. Inject `POSTGRES_*` yourself.

## Limitations

- Network and DB latency dominate vs in-memory.
- One table per embedding dimension.
- No built-in backup; use Postgres tooling.

Demo: `DemoCodeLive/demos/11_postgres.py` (skipped unless `DATABASE_URL` is set).
