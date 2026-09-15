---
title: "Configuration"
---

# Configuration

Memotrix never invents a database password, host, embedding model, or vector dimension. Missing values raise `ConfigurationError`.

## Constructing Memory

```python
from memotrix import Memory, MemoryConfig
from memotrix.config import ChunkingConfig, RetrievalConfig, IngestConfig
from memotrix.embeddings import HuggingFaceEmbeddings
from memotrix.vectorstores import InMemoryStore, PostgresStore
```

| Constructor | When |
|---|---|
| `Memory(embeddings, store=InMemoryStore(...))` | Explicit in-process store |
| `Memory(embeddings, backend="memory")` | Same, store created for you |
| `Memory(embeddings, store=PostgresStore(...))` | Explicit Postgres |
| `Memory(embeddings, connection=dsn)` | Postgres inferred from `connection=` |
| `Memory.from_config(config)` | Dataclass tunables + env-style model name |
| `Memory.from_env()` | All required values from the environment |

`embeddings` may be an `Embeddings` instance or a HuggingFace model name string (`coerce_embeddings`).

## Environment variables

Used by `Memory.from_env()` / `MemoryConfig.from_env()` / `HuggingFaceEmbeddings.from_env()` / `resolve_database_url()`.

| Variable | Required when | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `from_env()` | HuggingFace model id, e.g. `BAAI/bge-small-en-v1.5` |
| `MEMOTRIX_BACKEND` | optional | `memory` (default) or `postgres` |
| `DATABASE_URL` | backend is `postgres` | Postgres DSN |
| `DATABASE_USER` / `DATABASE_PASSWORD` / `DATABASE_HOST` / `DATABASE_PORT` / `DATABASE_NAME` | postgres without `DATABASE_URL` | Discrete DSN parts (also `POSTGRES_*` aliases) |
| `DATABASE_DIALECT` | optional | Default `postgresql` |
| `MEMOTRIX_TABLE` | optional | Chunk table name, default `memotrix_chunks` |
| `RERANKER_MODEL` | optional | Cross-encoder model name |
| `OPENAI_API_KEY` | `OpenAIEmbeddings` without `api_key=` | OpenAI embeddings / vision |
| `OPENAI_MODEL` | optional | Chat model for `OpenAIProvider` (not embeddings) |

## MemoryConfig defaults

These are **library defaults**, not secrets:

| Field | Default | Meaning |
|---|---|---|
| `backend` | `"memory"` | `memory` or `postgres` |
| `table_name` | `"memotrix_chunks"` | Postgres table |
| `max_elements` | `50000` | HNSW capacity (in-memory; **not** resizable) |
| `chunking.chunk_size` | `500` | Characters per chunk |
| `chunking.chunk_overlap` | `50` | Overlap characters |
| `retrieval.top_k` | `3` | Hits returned |
| `retrieval.max_tokens` | `800` | Soft budget after neighbor expansion |
| `retrieval.expand_mode` | `"neighbors"` | `neighbors` / `full_file` / `none` |
| `retrieval.neighbor_window` | `1` | ± chunks around a hit |
| `retrieval.enable_rerank` | `True` | Used only if a reranker model is set |
| `retrieval.fusion_k` | `60` | RRF constant |
| `retrieval.query_cache_size` | `256` | Cached query embeddings |
| `retrieval.enable_memory_boost` | `True` | Recency / access / importance multiplier |
| `ingest.enable_dedup` | `True` | Near-duplicate merge |
| `ingest.dedup_threshold` | `0.95` | Cosine threshold |
| `ingest.describe_images` | `True` | Vision/caption path for images |

Example:

```python
config = MemoryConfig(
    backend="memory",
    max_elements=10_000,
    chunking=ChunkingConfig(chunk_size=800, chunk_overlap=80),
    retrieval=RetrievalConfig(top_k=5, expand_mode="none", enable_memory_boost=False),
    ingest=IngestConfig(enable_dedup=False),
)
memory = Memory(embeddings=embeddings, config=config)
```

## Custom extractors at construction

```python
memory = Memory(embeddings=embeddings, extract_file=my_router)
```

`extract_file` must accept `(path, *, describe_images, generate_srt)` and return `DocumentData`. You can also register by extension: see [Custom extractors](./use-cases/custom-extractors.md).

## Docker Postgres

This repo’s `docker-compose.yml` starts `pgvector/pgvector:pg16`. Credentials come from **your** environment (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_PORT`). The compose file does not ship a default password.

See [Postgres persistence](./use-cases/postgres-persistence.md).
