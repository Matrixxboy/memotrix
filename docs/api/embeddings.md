---
title: "Embeddings API"
---

# Embeddings

```python
from memotrix.embeddings import (
    Embeddings,
    HuggingFaceEmbeddings,
    OpenAIEmbeddings,
    FakeEmbeddings,
    coerce_embeddings,
)
```

All implementations must provide `embed_documents`, `embed_query`, and `dimension`.

## `HuggingFaceEmbeddings(model, *, query_instruction=None)`

Requires `memotrix[embeddings]` / `[memory]`. Dimension from `get_sentence_embedding_dimension()` (or a probe encode). Query text may be prefixed with a BGE-style instruction via `format_query_for_embedding` unless `query_instruction` is set.

`HuggingFaceEmbeddings.from_env()` reads `EMBEDDING_MODEL`.

Passing a string to `Memory(embeddings="BAAI/bge-small-en-v1.5")` calls `coerce_embeddings` → `HuggingFaceEmbeddings`.

## `OpenAIEmbeddings(model, *, api_key=None)`

Requires `memotrix[openai]` and `api_key` or `OPENAI_API_KEY`. Dimension inferred from the first API response. Batches documents in groups of 500.

## `FakeEmbeddings(dim)`

Deterministic hash-like vectors. For tests and DemoCodeLive. **Not** semantically meaningful.

## Custom embeddings

Subclass `Embeddings` and implement the three members. Pass the instance to `Memory` and to `InMemoryStore` / `PostgresStore` so the index width matches.
