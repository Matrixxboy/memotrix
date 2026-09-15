---
title: "Quickstart"
---

# Quickstart

Five minutes from install to a search hit. Copy these snippets as-is.

## 1. Install

```bash
pip install "memotrix[memory]"
```

## 2. Remember a fact

```python
from memotrix import Memory
from memotrix.embeddings import HuggingFaceEmbeddings
from memotrix.vectorstores import InMemoryStore

embeddings = HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")
memory = Memory(embeddings=embeddings, store=InMemoryStore(embeddings))

memory.add_text(
    "User prefers dark mode and Python 3.12.",
    source_id="prefs",
    memory_type="semantic",
)

hits = memory.search("What theme does the user prefer?", top_k=3)
for hit in hits:
    print(hit.get("score"), hit.get("chunk_text", "")[:200])

memory.delete("prefs")
memory.close()
```

`HuggingFaceEmbeddings` downloads the model the first time. For tests and demos that must run offline, use `FakeEmbeddings(dim=8)` instead (deterministic, not semantically meaningful).

## 3. Ingest a file

```python
memory.add("notes.txt")
hits = memory.search("what is Memotrix?", top_k=3)
print(memory.list_sources())
```

`add` detects the extension, extracts text, chunks it (default 500 characters, 50 overlap), embeds the chunks, and writes both the dense and sparse indexes.

## 4. Persist with Postgres

```bash
pip install "memotrix[postgres,memory]"
```

```python
import os
from memotrix import Memory
from memotrix.embeddings import HuggingFaceEmbeddings
from memotrix.vectorstores import PostgresStore

embeddings = HuggingFaceEmbeddings(model=os.environ["EMBEDDING_MODEL"])
memory = Memory(
    embeddings=embeddings,
    store=PostgresStore(connection=os.environ["DATABASE_URL"], embeddings=embeddings),
)
```

Or `Memory.from_env()` when `EMBEDDING_MODEL` and (for Postgres) `DATABASE_URL` are set. See [Configuration](./configuration.md).

## 5. Live demos

[DemoCodeLive](https://github.com/Matrixxboy/memotrix/tree/main/DemoCodeLive) contains one script per use case, installed with `pip install memotrix`.

## 6. Plug into an LLM

Memotrix returns a list of hit dicts. You format `chunk_text` into a prompt. See [Build a RAG agent](./use-cases/rag-agent.md).

## Next

- [Every use case](./use-cases/index.md)
- [Memory API](./api/memory.md)
- [DemoCodeLive](https://github.com/Matrixxboy/memotrix/tree/main/DemoCodeLive)
