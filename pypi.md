# Memotrix

**Composable hybrid memory and RAG for AI agents.**

Memotrix lets an agent remember files and free-text facts the same way LangChain-style tools compose: pick embeddings, pick a store, then `add` / `search` / `delete`. It is a Python library — not an LLM, not a chat UI, and not a hosted service.

It chunks documents, stores **dense vectors** (semantic) plus a **keyword index** (BM25 or Postgres full-text), and retrieves a small context window with hybrid search, optional rerank, and neighbor expansion.

- [GitHub](https://github.com/Matrixxboy/memotrix)
- [User guide](https://github.com/Matrixxboy/memotrix/blob/main/memory/docs/USER_GUIDE.md)

---

## Why use it

| You need | Memotrix does |
|---|---|
| Agent long-term memory | `add_text` for facts, chat turns, procedures (`semantic` / `episodic` / `procedural`) |
| RAG over files | `add("report.pdf")` then `search("what is the revenue?")` |
| Hybrid retrieval | Dense (HNSW or pgvector) + sparse (BM25 or Postgres `tsvector`), fused with RRF |
| Tight context | Default `top_k=3` and neighbor-window expansion, not five full files |
| No silent secrets | You pass the embedding model and DSN. Nothing is defaulted. |

---

## Install

Python 3.10+. A bare `pip install memotrix` only installs `python-dotenv`. Use an extra:

```bash
pip install memotrix[memory]
```

Postgres + file extractors:

```bash
pip install memotrix[postgres,memory,extractors]
```

| Extra | Enables |
|---|---|
| `memory` | in-process HNSW + BM25 + Sentence-Transformers (minimum to construct `Memory`) |
| `local` | HNSW + BM25 |
| `embeddings` | HuggingFace / Sentence-Transformers |
| `postgres` | PostgreSQL + pgvector |
| `extractors` | PDF, Office, HTML, CSV, images, RDF, … |
| `openai` | OpenAI embeddings and vision |
| `audio` | Whisper transcription |
| `all` | everything above |

---

## Quick start

```python
from memotrix import Memory
from memotrix.embeddings import HuggingFaceEmbeddings
from memotrix.vectorstores import InMemoryStore

embeddings = HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")
memory = Memory(embeddings=embeddings, store=InMemoryStore(embeddings))

memory.add_text("User prefers dark mode.", memory_type="semantic", source_id="prefs")
hits = memory.search("what theme does the user want?", top_k=3)
memory.delete("prefs")
memory.close()
```

Files:

```python
memory.add("report.pdf")
hits = memory.search("what is the revenue?", memory_type="semantic")
print(memory.list_sources())
```

Postgres:

```python
import os
from memotrix.vectorstores import PostgresStore

embeddings = HuggingFaceEmbeddings(model=os.environ["EMBEDDING_MODEL"])
memory = Memory(
    embeddings=embeddings,
    store=PostgresStore(connection=os.environ["DATABASE_URL"], embeddings=embeddings),
)
```

Or from the environment (`EMBEDDING_MODEL` required; `DATABASE_URL` when `MEMOTRIX_BACKEND=postgres`):

```python
from memotrix import Memory
memory = Memory.from_env()
```

---

## Agent memory

```python
memory.add_text("Shipped hybrid search.", memory_type="episodic", session_id="2026-09-03")
memory.add_text("Always cite source_path.", memory_type="procedural")
memory.search("how should answers be cited?", memory_type="procedural")
```

`session_id` and `memory_type` are exact-match payload filters.

---

## What it can ingest

PDF, DOCX, PPTX, TXT, Markdown, HTML, EPUB, CSV, Excel (`.xlsx`), JSON (FHIR / GeoJSON / chat sniff), YAML, XML, SQL, images, video, audio (`[audio]`), source code, SCORM, knowledge graphs, GeoJSON, email (`.eml` / `.mbox`), chat exports, and `.log` files.

Generic `.zip` and BIFF `.xls` are not supported. Convert spreadsheets to `.xlsx`.

Plug in your own extractor:

```python
memory = Memory(embeddings=embeddings, extract_file=my_extractor)
```

---

## Links

- Source: [github.com/Matrixxboy/memotrix](https://github.com/Matrixxboy/memotrix)
- Full API: [User guide](https://github.com/Matrixxboy/memotrix/blob/main/memory/docs/USER_GUIDE.md)
- License: MIT
