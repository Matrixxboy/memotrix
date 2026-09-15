---
title: "Memory API"
---

# `Memory`

```python
from memotrix import Memory
```

Implementation: `src/memotrix/memory.py`.

## Constructor

```python
Memory(
    embeddings: Embeddings | str,
    *,
    store=None,
    connection: str | None = None,
    backend: str | None = None,
    config: MemoryConfig | None = None,
    document_store: DocumentStore | None = None,
    extract_file=None,
)
```

- `store` — `InMemoryStore`, `PostgresStore`, `VectorStoreBundle`, or any object with `.dense` and `.sparse`.
- `backend` — `"memory"` or `"postgres"`. If `connection` is set and backend is omitted, Postgres is selected.
- `connection` with `backend="memory"` raises `ConfigurationError`.

### Class methods

- `Memory.from_config(config, embeddings=None)` — requires `config.embedding_model` if `embeddings` is omitted.
- `Memory.from_env()` — `MemoryConfig.from_env()`.

## `add(path, *, describe_images=None, generate_srt=False) -> dict`

Extract + ingest. See [file ingestion](../use-cases/file-ingestion.md).

## `add_text(text, *, source_id=None, memory_type="semantic", session_id=None, metadata=None) -> dict`

Empty/whitespace text raises `ConfigurationError`.

## `add_documents(items) -> dict`

`items` is a sequence of `DocumentData` or `(DocumentData, source_path)` tuples. Bypasses file extractors.

## `search(query, *, top_k=None, max_tokens_returned=None, filters=None, session_id=None, memory_type=None, expand_mode=None) -> list[dict]`

Returns hit dictionaries. Common keys (when present):

| Key | Meaning |
|---|---|
| `chunk_text` / `content` | Text after expansion |
| `matched_chunk` | Pre-expansion match |
| `score` | Rank score (RRF or rerank × memory multiplier) |
| `chunk_id` | Internal id |
| `source_path` / `filename` | Source |
| `memory_type` / `session_id` | Payload filters |
| `expansion` | `neighbors` / `full_file` / `none` |
| `tokens_estimate` | Approximate tokens |
| `reranked` | Whether cross-encoder ran |
| `access_count` / `last_accessed` | If access tracking is on |

## `delete(source_path) -> int`

Deletes from dense, sparse, and document store. Empty string returns `0`. Return value is `max(dense_deleted, sparse_deleted)`.

## `list_sources() -> list[dict]`

Unique `source_path` / `filename` with `chunks` counts from dense payloads.

## `extract(path, ...)`

Returns `DocumentData`.

## `as_retriever()` / `as_stack()` / `close()`

`close()` closes the Postgres pool when using `PostgresStore`; no-op for in-memory native store.
