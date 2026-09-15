---
title: "Memotrix"
description: "Composable hybrid memory and RAG for AI agents"
---

# Memotrix

Memotrix is a **Python library** that gives an agent long-term memory over files and free-text facts. You pick embeddings, pick a store, then call `add` / `add_text` / `search` / `delete`.

It is **not** a hosted service, **not** a chat UI, and **not** an LLM. You retrieve chunks, then you pass those chunks into your own model.

```python
from memotrix import Memory
from memotrix.embeddings import HuggingFaceEmbeddings
from memotrix.vectorstores import InMemoryStore

embeddings = HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")
memory = Memory(embeddings=embeddings, store=InMemoryStore(embeddings))

memory.add_text("User prefers dark mode.", memory_type="semantic", source_id="prefs")
hits = memory.search("what theme does the user want?", top_k=3)
memory.close()
```

## What it does

| You need | Memotrix does |
|---|---|
| Agent long-term memory | `add_text` for facts, chat turns, procedures (`semantic` / `episodic` / `procedural`) |
| RAG over files | `add("report.pdf")` then `search("what is the revenue?")` |
| Hybrid retrieval | Dense vectors (HNSW or pgvector) + sparse keywords (BM25 or Postgres `tsvector`), fused with RRF |
| Tight context | Default `top_k=3` and neighbor-window expansion, not five full files |
| Explicit config | You pass the embedding model and DSN. Nothing is silently defaulted. |

## Install

```bash
pip install memotrix
```

A bare install only includes `python-dotenv`. To construct `Memory` you need an extra:

```bash
pip install "memotrix[memory]"
```

See [Installation](./installation.md) for every extra (`postgres`, `extractors`, `openai`, `audio`, `all`).

## Documentation map

| Section | Start here |
|---|---|
| Get started | [Installation](./installation.md) · [Quickstart](./quickstart.md) · [Configuration](./configuration.md) |
| How it works | [Concepts](./concepts/overview.md) · [Architecture](./concepts/architecture.md) · [Hybrid search](./concepts/hybrid-search.md) |
| Every use case | [Use cases](./use-cases/index.md) |
| API | [`Memory`](./api/memory.md) · [Embeddings](./api/embeddings.md) · [Stores](./api/vectorstores.md) |
| Live demos | [DemoCodeLive](https://github.com/Matrixxboy/memotrix/tree/main/DemoCodeLive) |
| Internal architecture notes | [docBook](https://github.com/Matrixxboy/memotrix/tree/main/docBook) |

## Public API surface

From `memotrix`:

- `Memory`, `MemoryConfig`, `build_retrieval_stack`
- `HuggingFaceEmbeddings`, `OpenAIEmbeddings`, `FakeEmbeddings`, `Embeddings`
- `InMemoryStore`, `PostgresStore`
- `ConfigurationError`

Package version on PyPI: **0.2.0**. Source in this repo: **0.2.1** (Pillow no longer required to import `Memory`; `__version__` matches `pyproject.toml`). Python **3.10+**. License: MIT.

## Live demo project

Runnable scripts that import the **PyPI** package (not this repo’s `src/`):

[`DemoCodeLive/`](https://github.com/Matrixxboy/memotrix/tree/main/DemoCodeLive)

```bash
cd DemoCodeLive
python -m venv .venv
.venv/Scripts/activate   # Windows
pip install "memotrix[memory]"
python run_all.py
```
