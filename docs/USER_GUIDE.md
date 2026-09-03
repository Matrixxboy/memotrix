# Memotrix user guide

This is the complete usage reference for the `memotrix` Python SDK (the package in `memory/`). Copy-ready scripts live in [`examples/`](../examples/). Architecture internals live in [`Documentation.md`](../Documentation.md); if those two disagree, **this guide and the code win**.

Memotrix is a composable hybrid memory / RAG library for agents: extract files or free-text facts, chunk them, store dense vectors plus a keyword index, and retrieve a small context window. It is **not** an LLM, a chat UI, or a hosted service.

---

## Table of contents

1. [Install](#1-install)
2. [Core concepts](#2-core-concepts)
3. [Constructing `Memory`](#3-constructing-memory)
4. [Embeddings](#4-embeddings)
5. [Vector stores](#5-vector-stores)
6. [Ingest](#6-ingest)
7. [Search](#7-search)
8. [Delete, list, close](#8-delete-list-close)
9. [Configuration and environment](#9-configuration-and-environment)
10. [Memory types and filters](#10-memory-types-and-filters)
11. [Supported file types](#11-supported-file-types)
12. [PostgreSQL](#12-postgresql)
13. [Agent loop](#13-agent-loop)
14. [Custom extractors and indexes](#14-custom-extractors-and-indexes)
15. [Exceptions](#15-exceptions)
16. [Advanced surfaces](#16-advanced-surfaces)
17. [Examples](#17-examples)
18. [Troubleshooting](#18-troubleshooting)

Package-root exports (`from memotrix import …`):

- `Memory`, `MemoryConfig`
- `Embeddings`, `HuggingFaceEmbeddings`, `FakeEmbeddings`, `OpenAIEmbeddings`
- `PostgresStore`, `InMemoryStore`
- `ConfigurationError`
- `build_retrieval_stack`
- `__version__` (`0.1.0`)

---

## 1. Install

Python 3.10 or newer. There is **no** default password, host, database name, embedding model, or embedding dimension.

```bash
pip install memotrix[memory]
```

| Extra | Enables |
|---|---|
| `memory` | `local` + `embeddings` (minimum to construct `Memory`) |
| `local` | in-process HNSW + BM25 |
| `embeddings` | Sentence-Transformers / HuggingFace models |
| `postgres` | PostgreSQL + pgvector |
| `extractors` | PDF, Office, HTML, CSV, images, RDF, … |
| `openai` | OpenAI chat/vision + `OpenAIEmbeddings` |
| `audio` | Whisper transcription (`faster-whisper`) |
| `dev` | pytest |
| `all` | postgres + local + embeddings + extractors + openai + audio |

Editable checkout from this repo:

```bash
pip install -e ./memory[postgres,local,embeddings,extractors,openai]
```

Publishing this package to PyPI is documented in [PUBLISHING.md](PUBLISHING.md).

---

## 2. Core concepts

- **Dense index** — vector similarity (HNSW in-process, or pgvector).
- **Sparse index** — keyword search (BM25 in-process, or PostgreSQL `tsvector`).
- **Hybrid search** — Reciprocal Rank Fusion of both channels (`fusion_k`, default 60).
- **Chunk** — a slice of a document or memory with payload metadata (`chunk_text`, `source_path`, `memory_type`, …).
- **DocumentStore** — in-process adjacency used to expand a hit to neighboring chunks. Rehydrated from stored payloads when you reopen a Postgres-backed `Memory`.
- **`memory_type`** — convention, not an enum: `semantic` (facts), `episodic` (events / chat turns), `procedural` (how-to). Default for `add` / `add_text` is `semantic`.

---

## 3. Constructing `Memory`

```python
from memotrix import Memory
from memotrix.embeddings import HuggingFaceEmbeddings, FakeEmbeddings
from memotrix.vectorstores import InMemoryStore, PostgresStore
```

### 3.1 In-process

```python
embeddings = FakeEmbeddings(dim=8)  # tests only
# embeddings = HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")
memory = Memory(embeddings=embeddings, backend="memory")
# equivalent:
memory = Memory(embeddings=embeddings, store=InMemoryStore(embeddings))
```

`InMemoryStore` options: `max_elements` (default 50_000), `space` (`cosine`). Exceeding `max_elements` raises; the index does not auto-resize.

### 3.2 PostgreSQL

```python
memory = Memory(
    embeddings=embeddings,
    store=PostgresStore(connection=os.environ["DATABASE_URL"], embeddings=embeddings),
)
# or
memory = Memory(
    embeddings=embeddings,
    backend="postgres",
    connection=os.environ["DATABASE_URL"],
)
# or omit backend when connection= is set — postgres is selected automatically
memory = Memory(embeddings=embeddings, connection=os.environ["DATABASE_URL"])
```

`Memory(..., backend="memory", connection=dsn)` raises `ConfigurationError` (those two options contradict).

`PostgresStore` options: `connection` (required), `table_name` (default `memotrix_chunks`). Vector width is `embeddings.dimension`. Alternate constructor: `PostgresStore.from_conn_string(connection, embeddings, table_name=...)`.

### 3.3 String embedding model

```python
memory = Memory(embeddings="BAAI/bge-small-en-v1.5", backend="memory")
```

A string is coerced to `HuggingFaceEmbeddings(model=...)`.

### 3.4 `from_config` / `from_env`

```python
from memotrix import Memory, MemoryConfig
from memotrix.config import ChunkingConfig, RetrievalConfig, IngestConfig

cfg = MemoryConfig(
    backend="memory",
    embedding_model="BAAI/bge-small-en-v1.5",
    chunking=ChunkingConfig(chunk_size=500, chunk_overlap=50),
    retrieval=RetrievalConfig(top_k=3, max_tokens=800, expand_mode="neighbors"),
    ingest=IngestConfig(enable_dedup=True, describe_images=True),
)
memory = Memory.from_config(cfg)
# or pass a ready Embeddings instance:
memory = Memory.from_config(cfg, embeddings=FakeEmbeddings(dim=8))

memory = Memory.from_env()  # requires EMBEDDING_MODEL; postgres also needs a DSN
```

### 3.5 Custom extractor hook

```python
memory = Memory(
    embeddings=embeddings,
    extract_file=my_extractor,  # (path, *, describe_images, generate_srt) -> DocumentData
)
```

### 3.6 Shared `document_store`

Pass `document_store=DocumentStore()` only if you need to share adjacency across `Memory` instances. Normally one is created for you and rehydrated from index payloads.

---

## 4. Embeddings

Import from `memotrix.embeddings` (also re-exported at the package root).

| Class | Constructor | Notes |
|---|---|---|
| `HuggingFaceEmbeddings` | `model: str`, optional `query_instruction=` | Loads Sentence-Transformers; BGE models get a query prefix automatically. `from_env()` reads `EMBEDDING_MODEL`. |
| `FakeEmbeddings` | `dim: int` | Deterministic vectors for tests. `model_name` is `fake/{dim}`. |
| `OpenAIEmbeddings` | `model: str`, optional `api_key=` | Needs `memotrix[openai]`. Key from arg or `OPENAI_API_KEY`. Dimension inferred from the first API response. |

ABC methods every implementation must provide: `embed_documents(texts)`, `embed_query(text)`, `dimension`, `model_name`.

---

## 5. Vector stores

```python
from memotrix.vectorstores import InMemoryStore, PostgresStore, VectorStoreBundle
```

`store=` on `Memory` accepts:

- `InMemoryStore` / `PostgresStore` (objects with `.bundle`)
- a `VectorStoreBundle(dense=..., sparse=..., native=...)`
- any object with `.dense` and `.sparse`
- the string `"memory"` or `"postgres"` (as a backend name)

Both stores expose `.dense` and `.sparse`. Call `memory.close()` (or `store.close()`) when done; `InMemoryStore.close()` is a no-op over the in-process indexes.

---

## 6. Ingest

### 6.1 `add(path)` — files

```python
stats = memory.add("report.pdf")
stats = memory.add("report.pdf", describe_images=False, generate_srt=False)
```

Returns `filename`, `path` (resolved absolute path), `chunks`, `inserted`, `merged`, `extracted_images`. Sets `metadata.source_path` to the resolved path and default `memory_type="semantic"`.

- `describe_images` defaults to `config.ingest.describe_images` (True). When False, PDF/DOCX/PPTX/standalone images skip vision captioning.
- `generate_srt` is forwarded to audio/video extractors.

### 6.2 `add_text(text)` — facts, chat turns, procedures

```python
memory.add_text(
    "User prefers dark mode.",
    source_id="prefs",           # default: "text:<uuid>"
    memory_type="semantic",      # semantic | episodic | procedural
    session_id="2026-09-03",     # optional exact-match filter later
    metadata={"title": "prefs"}, # merged into chunk payload
)
```

Empty / whitespace-only text raises `ConfigurationError`. Return shape matches `add` (`extracted_images=0`). **Delete with the same `source_id`** (or the auto `text:<uuid>` from the return value’s `path`).

### 6.3 `add_documents(items)` — already extracted

```python
from memotrix.utils.models import DocumentData

doc = DocumentData(
    metadata={"filename": "note.txt", "source_path": "note.txt"},
    sections=[{"title": "", "content": "hello"}],
    text="hello",
)
memory.add_documents([doc])
# or (doc, source_path) tuples
memory.add_documents([(doc, "note.txt")])
```

Returns `{"chunks", "merged", "inserted"}`.

### 6.4 `extract(path)` — extract without indexing

```python
doc = memory.extract("report.pdf", describe_images=False, generate_srt=False)
print(doc.text, doc.tables, doc.images)
```

Same routing as `add`, using the instance `extract_file` hook when set.

### 6.5 Re-ingest and dedup

Re-adding the same `source_path` / `source_id` deletes prior chunks (indexes **and** neighbor adjacency) then inserts the new ones.

Near-duplicate merge (cosine ≥ `dedup_threshold`, default 0.95) only happens **within the same source**. Two files that happen to contain the same sentence stay as two memories.

---

## 7. Search

```python
hits = memory.search(
    "what theme does the user want?",
    top_k=3,                      # default RetrievalConfig.top_k (3)
    max_tokens_returned=800,      # default max_tokens
    filters={"project": "alpha"}, # exact-match payload dict
    session_id="s1",              # merged into filters
    memory_type="procedural",     # merged into filters
    expand_mode="neighbors",      # neighbors | full_file | none
)
```

Unknown keyword arguments raise `TypeError` (the facade does not accept `**kwargs`).

`RetrievalPipeline.retrieve(...)` is an alias of `search`.

### Expand modes

| Mode | Behavior |
|---|---|
| `neighbors` (default) | Matched chunk ± `neighbor_window` (default 1) in the same section, capped by `max_tokens_returned` |
| `full_file` | Unique sources expanded to full stored text |
| `none` | Matched chunks only |

### Hit fields (typical)

Payload metadata plus retrieval fields:

- `chunk_id`, `chunk_text` (possibly expanded), `matched_chunk` (original slice)
- `score`, `reranked`, `expansion`, `neighbor_chunk_indices`, `tokens_estimate`
- `filename`, `source_path`, `memory_type`, `session_id`, `type`, `section_title`
- `created_at`, `last_accessed`, `access_count`, `importance`
- `relevance_score`, `memory_multiplier` when memory boost is on

Access counts are bumped on the **stored** payload; expanded `chunk_text` is not written back into the index.

### Rerank

`enable_rerank` defaults to True, but the cross-encoder is **off** unless `reranker_model` or `RERANKER_MODEL` is set. Conditional skip still applies when a reranker is loaded (large RRF gap or dense/sparse agree).

---

## 8. Delete, list, close

```python
n = memory.delete("prefs")                 # source_id from add_text
n = memory.delete("note.txt")              # filename of an added file
n = memory.delete(r"C:\data\report.pdf")   # exact source_path from add()
print(memory.list_sources())               # [{source_path, filename, chunks}, ...]
memory.close()
```

`delete` removes dense + sparse rows and DocumentStore adjacency. You may pass either the stored `source_path` or the **filename** (basename). Unrelated names that merely *end with* the same suffix (e.g. `report_a.txt` vs `a.txt`) are not deleted.

Return value is the number of index rows removed (`max(dense, sparse)`).

---

## 9. Configuration and environment

```python
from memotrix.config import (
    MemoryConfig, ChunkingConfig, RetrievalConfig, IngestConfig,
    require_env, resolve_database_url,
)
```

### `ChunkingConfig`

- `chunk_size=500`, `chunk_overlap=50` (characters). Overlap is applied between packed units.

### `RetrievalConfig`

- `top_k=3`, `max_tokens=800`, `expand_mode="neighbors"`, `neighbor_window=1`
- `enable_rerank=True`, `reranker_model=None` (must be a model id to actually rerank)
- `enable_memory_boost=True` (mild recency / access / importance multiplier)
- `fusion_k=60`, `query_cache_size=256`

### `IngestConfig`

- `enable_dedup=True`, `dedup_threshold=0.95`, `describe_images=True`

### `MemoryConfig` extra fields

- `backend`: `"memory"` \| `"postgres"`
- `connection`, `table_name="memotrix_chunks"`, `max_elements=50000`
- `embedding_model`, `reranker_model`

### Environment (`Memory.from_env` / `.env.example`)

| Variable | When |
|---|---|
| `EMBEDDING_MODEL` | always required for `from_env()` |
| `MEMOTRIX_BACKEND` | `memory` (default) or `postgres` |
| `DATABASE_URL` | postgres DSN |
| `DATABASE_*` / `POSTGRES_*` | discrete DSN parts if no URL (`USER`, `PASSWORD`, `HOST`, `PORT`, `NAME`/`DB`) |
| `DATABASE_DIALECT` | default `postgresql` when building from parts |
| `MEMOTRIX_TABLE` | table name (default `memotrix_chunks`) |
| `RERANKER_MODEL` | optional cross-encoder id |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | OpenAI embeddings and vision |
| `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` / `OPENROUTER_BASE_URL` | provider class exists; not auto-selected by the vision router today |
| `GOOGLE_API_KEY` | Gemini vision/text fallback |
| `MEMOTRIX_TRACE` | `1` (default) or `0` |
| `MEMOTRIX_TRACE_MAX` | trace clip length (default 4000) |

`from_env()` does **not** load chunk size, `top_k`, dedup, or expand mode from the environment — set those on `MemoryConfig`.

---

## 10. Memory types and filters

```python
memory.add_text("Shipped hybrid search.", memory_type="episodic", session_id="2026-09-03")
memory.add_text("Always cite source_path.", memory_type="procedural")
memory.search("how should answers be cited?", memory_type="procedural")
memory.search("dark mode", filters={"session_id": "s1", "memory_type": "semantic"})
```

Filters are exact-match on payload keys (`payload.get(key) == expected`). `session_id=` and `memory_type=` on `search` are merged into `filters`.

---

## 11. Supported file types

`from memotrix.filetypes import extract_file, supported_extensions`

`extract_file(path, describe_images=True, generate_srt=False)` routes by extension (and JSON content sniff). Requires `memotrix[extractors]` for Office/PDF/HTML/RDF, plus `[audio]` for Whisper.

| Kind | Extensions | Notes |
|---|---|---|
| PDF | `.pdf` | Text + embedded/scanned images; `describe_images` |
| Word | `.docx` | Paragraphs **and tables**; embedded images |
| PowerPoint | `.pptx` | Slide text + pictures |
| Text | `.txt` | WhatsApp-looking logs route to chat |
| Markdown | `.md`, `.markdown` | Native text (headings kept as source text) |
| HTML | `.html`, `.htm` | Tags stripped |
| EPUB | `.epub` | Chapter HTML |
| CSV | `.csv` | Header-prefixed row batches |
| Excel | `.xlsx` only | `.xls` (BIFF) is rejected — convert to `.xlsx` |
| JSON | `.json` | Pretty dump, unless sniffed as FHIR / GeoJSON / chat / JSON-LD |
| JSONL | `.jsonl` | Chat if messages look like chat; skipped bad lines counted in `skipped_lines` |
| YAML | `.yaml`, `.yml` | |
| XML | `.xml` | |
| SQL | `.sql` | Raw text |
| Images | `.png` `.jpg` `.jpeg` `.webp` `.bmp` `.gif` `.tiff` `.tif` `.svg` `.heic` | Vision caption unless `describe_images=False` |
| Video | `.mp4` `.mkv` `.avi` `.mov` `.webm` `.m4v` `.wmv` | ffmpeg + Whisper; `generate_srt` |
| Audio | `.mp3` `.wav` `.aac` `.m4a` `.ogg` `.flac` `.wma` | `[audio]` extra |
| Code | `.py` `.js` `.ts` `.java` `.cpp` `.cc` `.cxx` `.go` `.rs` | Function/class blocks |
| SCORM | `.scorm`, or `.zip` **with** `imsmanifest.xml` | Generic zips raise `UnsupportedDocumentTypeError` |
| Knowledge graphs | `.ttl` `.nt` `.nq` `.rdf` `.owl` `.trig` `.jsonld` `.graphml` | Triples as English |
| GeoJSON | `.geojson` or sniffed `.json` | |
| FHIR | sniffed `.json` with `resourceType` | |
| Email | `.eml` `.mbox` | mbox stops after 500 messages |
| Logs | `.log` | Timestamp windows |

JSON sniff order: FHIR → GeoJSON → JSON-LD (`@graph` / `@context`+`@id`) → chat. Catalog JSON like `[{"name": "Widget", "text": "..."}]` is **not** treated as chat (chat needs `role` / `author` / `from` / `sender`, or `name` plus a timestamp field).

Call `supported_extensions()` for the live list.

---

## 12. PostgreSQL

Needs `pgvector` (see [`docker-compose.yml`](../docker-compose.yml)):

```bash
# from memory/, with POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB in .env
docker compose up -d
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

Table default `memotrix_chunks` (`MEMOTRIX_TABLE`). Schema uses `vector({embeddings.dimension})`. Changing embedding width against an existing table is not migrated automatically — use a new table name or drop the old one.

`create_postgres_indexes(connection, dim=..., table_name=...)` is available from `memotrix.api` for advanced wiring.

---

## 13. Agent loop

Pattern from [`examples/03_agent_loop.py`](../examples/03_agent_loop.py):

```python
hits = memory.search(query, top_k=3, max_tokens_returned=400)
context = "\n\n".join(hit.get("chunk_text") or "" for hit in hits)
# pass context + query to your LLM
```

Keep `top_k` and `max_tokens_returned` small so the model sees a neighborhood, not five full files.

---

## 14. Custom extractors and indexes

### Custom `extract_file`

Return a `DocumentData` (`memotrix.utils.models`). Helpers: `build_document` in `memotrix.utils.outputSturcture`.

### Custom dense + sparse

```python
from memotrix import Memory, build_retrieval_stack

memory = Memory(embeddings=embeddings, store=my_store)  # .dense and .sparse

stack = build_retrieval_stack(
    backend="memory",
    embeddings=embeddings,
    custom_dense=my_dense,
    custom_sparse=my_sparse,
    expand_mode="neighbors",
    max_elements=10_000,
)
# stack keys: dense, sparse, store, document_store, hybrid, ingest_pipeline, retrieval, memory
```

If `connection=` is passed to `build_retrieval_stack` with the default `backend="memory"`, the stack switches to postgres.

---

## 15. Exceptions

```python
from memotrix.utils.exceptions import (
    ConfigurationError,            # also: from memotrix import ConfigurationError
    DocumentExtractionError,
    UnsupportedDocumentTypeError,
    EmptyDocumentError,
    ExtractionBackendError,
)
```

| Exception | Typical cause |
|---|---|
| `ConfigurationError` | Missing model / DSN / empty `add_text` / bad `store` / contradictory `backend`+`connection` |
| `UnsupportedDocumentTypeError` | Unknown extension, generic `.zip`, `.xls` |
| `EmptyDocumentError` | Extractor produced no content |
| `ExtractionBackendError` | Optional library missing for that type |
| `FileNotFoundError` | Path does not exist |

---

## 16. Advanced surfaces

```python
retriever = memory.as_retriever()   # RetrievalPipeline
stack = memory.as_stack()           # dict used by a server/agent
memory.dense / memory.sparse / memory.hybrid / memory.ingest_pipeline
```

`memotrix.api` also lists `RetrievalPipeline`, `HybridSearchEngine`, `DocumentStore`, `IngestionPipeline`, `HNSWDenseIndex`, `BM25SparseIndex`, `estimate_tokens`, `OpenAIProvider`, `OpenRouterProvider`, `configure_ai_provider`, `get_ai_router`, `create_postgres_indexes`. These are for advanced composition, not the minimum SDK.

Tracing: `MEMOTRIX_TRACE=1` (default) logs ingest/retrieve; set `0` to silence.

---

## 17. Examples

| File | What it shows |
|---|---|
| [`examples/01_in_memory.py`](../examples/01_in_memory.py) | `FakeEmbeddings`, `add_text`, `search`, `close` |
| [`examples/02_files_postgres.py`](../examples/02_files_postgres.py) | `add` a file; Postgres when `DATABASE_URL` is set |
| [`examples/03_agent_loop.py`](../examples/03_agent_loop.py) | Retrieve → join `chunk_text` as LLM context |

---

## 18. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `connection=` seems ignored | Pass `backend="postgres"` or omit `backend`. `backend="memory"` plus `connection=` raises. |
| `delete("note.txt")` returns 0 | Use the `source_id` / `path` returned by `add_text`, or the filename / resolved path from `add`. |
| Search ignores `enable_rerank=True` | Set `RERANKER_MODEL` or `RetrievalConfig.reranker_model`. |
| `max_elements` error on ingest | `InMemoryStore` cap (default 50k). Raise `max_elements` on the store / `MemoryConfig`. |
| Vision calls on every DOCX image | Pass `describe_images=False` or `IngestConfig(describe_images=False)`. |
| `.zip` ingest fails | Only SCORM zips with `imsmanifest.xml` (or `.scorm`) are supported. |
| `.xls` rejected | Convert to `.xlsx`. |
| Postgres bool filters miss | Boolean payload values are stored as JSON `true`/`false`; filters now match that form. |
| Neighbor windows look stale after overwrite | Re-ingest now clears DocumentStore adjacency for that source. |
| Chat extractor ate a product JSON | Sniff requires a real chat author field, not just `{name, text}`. |
