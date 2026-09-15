# Memotrix

Composable hybrid memory / RAG for agents — pip-installable, like LangChain.

**Website docs (merged use cases + API):** [docs/index.md](docs/index.md)  
**Live demos (PyPI install):** [DemoCodeLive/](DemoCodeLive/README.md)  
**Architecture notebook:** [docBook/00-index.md](docBook/00-index.md)  
**Publish to PyPI:** [docs/PUBLISHING.md](docs/PUBLISHING.md)

## Install

Minimum (in-process HNSW + BM25 + local embeddings):

```bash
pip install memotrix[memory]
```

Postgres + file extractors:

```bash
pip install memotrix[postgres,memory,extractors]
```

Editable (this repo):

```bash
pip install -e ./memory[postgres,local,embeddings,extractors,openai]
```

| Extra | What it enables |
|---|---|
| `memory` | `local` + `embeddings` (minimum to construct `Memory`) |
| `local` | in-process HNSW + BM25 |
| `embeddings` | Sentence-Transformers / HuggingFace models |
| `postgres` | PostgreSQL + pgvector |
| `extractors` | PDF, Office, HTML, CSV, images, … |
| `openai` | OpenAI chat/vision + `OpenAIEmbeddings` |
| `audio` | Whisper transcription for audio files |
| `all` | everything above |

There is **no** default password, host, database name, embedding model, or embedding dimension.

## Quick start

```python
from memotrix import Memory
from memotrix.embeddings import HuggingFaceEmbeddings, FakeEmbeddings
from memotrix.vectorstores import PostgresStore, InMemoryStore

embeddings = HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")
memory = Memory(
    embeddings=embeddings,
    store=InMemoryStore(embeddings),
)

memory.add_text("User prefers dark mode.", memory_type="semantic")
hits = memory.search("what theme does the user want?", top_k=3)
memory.delete("text:…")  # or the source_id you passed
```

Files:

```python
memory.add("report.pdf")
hits = memory.search("what is the revenue?", filters={"memory_type": "semantic"})
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

From environment (fails if required vars are missing):

```python
from memotrix import Memory
memory = Memory.from_env()
```

| Variable | When |
|---|---|
| `EMBEDDING_MODEL` | always for `from_env()` |
| `DATABASE_URL` (or discrete `DATABASE_*` / `POSTGRES_*` parts) | `MEMOTRIX_BACKEND=postgres` |
| `MEMOTRIX_BACKEND` | `memory` (default) or `postgres` |
| `RERANKER_MODEL` | optional |

## Agent memory types

`add_text` is the path for facts, chat turns, and procedures — not only files.

```python
memory.add_text("Shipped hybrid search.", memory_type="episodic", session_id="2026-09-03")
memory.add_text("Always cite source_path.", memory_type="procedural")
memory.search("how should answers be cited?", memory_type="procedural")
```

`session_id` and `memory_type` are exact-match payload filters.

## Custom extractors and embeddings

```python
memory = Memory(
    embeddings=embeddings,
    extract_file=my_extractor,  # (path, *, describe_images, generate_srt) -> DocumentData
)
```

`OpenAIEmbeddings` implements the same `Embeddings` ABC (install `[openai]`). HuggingFace remains the local default.

Custom dense/sparse indexes can be passed as `store=` if they expose `.dense` and `.sparse`.

## Supported file types (wired)

PDF, DOCX, PPTX, TXT, Markdown, HTML, EPUB, CSV, Excel (`.xlsx`), JSON (including FHIR / GeoJSON / chat sniff), YAML, XML, SQL, images, video, audio (`[audio]`), source code, SCORM (`.scorm` or a zip that contains `imsmanifest.xml`), knowledge graphs (Turtle / N-Triples / RDF / JSON-LD / GraphML), GeoJSON, email (`.eml` / `.mbox`), chat JSON/JSONL/WhatsApp txt, and `.log` files. Generic `.zip` and BIFF `.xls` are not supported.

Domain apps (marine, education, finance, medical) ingest those files with `memory.add(path)` and store free-form notes with `memory.add_text(...)`.

See `DemoCodeLive/` (PyPI `pip install "memotrix[memory]"`), plus `examples/01_in_memory.py`, `examples/02_files_postgres.py`, `examples/03_agent_loop.py`.

Full usage: [docs/index.md](docs/index.md) · every use case: [docs/use-cases/index.md](docs/use-cases/index.md).
Releasing the package: [docs/PUBLISHING.md](docs/PUBLISHING.md).
