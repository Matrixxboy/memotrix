---
title: "Vector stores API"
---

# Vector stores

```python
from memotrix.vectorstores import InMemoryStore, PostgresStore, VectorStoreBundle
```

## `InMemoryStore(embeddings, *, max_elements=50_000, space="cosine")`

HNSW (`hnswlib`) + BM25 (`rank-bm25`). `max_elements` cannot grow after init. Data is process-local.

## `PostgresStore(connection, embeddings, *, table_name="memotrix_chunks")`

Empty `connection` raises `ConfigurationError`. `from_conn_string` is an alias constructor.

## Custom stores

Any object with `.dense` and `.sparse` (and optional `.native`, `.bundle`, `.close`) can be passed as `store=`. `build_retrieval_stack(..., custom_dense=, custom_sparse=)` wraps that pattern.

## `build_retrieval_stack`

```python
from memotrix import build_retrieval_stack

stack = build_retrieval_stack(
    backend="memory",
    embeddings=embeddings,
)
# stack["memory"], stack["retrieval"], stack["dense"], ...
```

Requires embeddings or `embedding_model`. Used when you want the internal dict instead of only the facade.
