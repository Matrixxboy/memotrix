---
title: "Installation"
---

# Installation

Python **3.10+** is required (`requires-python` in `pyproject.toml`).

## PyPI

```bash
pip install memotrix
```

That command installs the package and its only required dependency: `python-dotenv`. It does **not** install HNSW, BM25, Sentence-Transformers, Postgres drivers, or file extractors. In 0.2.0, importing `Memory` still loads Pillow via `utils.ai_integration`; install `pillow` or use `memotrix[images]`. `InMemoryStore` needs `hnswlib` from `[local]` / `[memory]` (Windows: MSVC to compile). Details: [Troubleshooting](./guides/troubleshooting.md).

To actually construct `Memory` and run hybrid search:

```bash
pip install "memotrix[memory]"
```

Quotes are required in zsh and some other shells.

## Extras

| Extra | Enables | When you need it |
|---|---|---|
| `memory` | `local` + `embeddings` | Minimum to construct `Memory` with local embeddings |
| `local` | `hnswlib`, `rank-bm25`, `numpy` | In-process HNSW + BM25 (`InMemoryStore`) |
| `embeddings` | `sentence-transformers` | `HuggingFaceEmbeddings` |
| `postgres` | `psycopg[binary]`, `pgvector` | `PostgresStore` |
| `extractors` | PDF, Office, HTML, CSV/Excel, images, RDF, YAML | `memory.add("file.pdf")` for those formats |
| `pdf` / `docx` / `pptx` / `excel` / `images` / `html` / `knowledge` / `yaml` | Granular extractors | Install only the formats you use |
| `documents` | `pdf,docx,pptx,excel,html` | Office + HTML + PDF together |
| `data` | `excel,knowledge,yaml` | Spreadsheets + RDF + YAML |
| `all-extractors` | documents + data + images | Every wired extractor extra |
| `openai` | `openai` | `OpenAIEmbeddings` and vision captions |
| `audio` | `faster-whisper` | Audio / video transcription |
| `all` | postgres + local + embeddings + extractors + openai + audio | Full install |
| `dev` | `pytest` | Running the library test suite from source |

Examples:

```bash
pip install "memotrix[memory]"
pip install "memotrix[postgres,memory,extractors]"
pip install "memotrix[memory,pdf]"
pip install "memotrix[all]"
```

## What works without extras

These formats use the Python standard library (or code already in the wheel) once `[local]` is installed so indexes exist:

- `.txt`, `.md`, `.json`, `.jsonl`, `.csv`, `.sql`, `.xml`, `.py` / `.js` / `.ts` / `.java` / `.go` / `.rs` / C++, `.log`, `.eml`, `.mbox`, `.geojson`

These need extras:

| Format | Extra |
|---|---|
| PDF | `pdf` or `extractors` |
| DOCX / PPTX | `docx` / `pptx` |
| Excel `.xlsx` | `excel` |
| HTML / EPUB | `html` (EPUB uses stdlib zip + HTML helper) |
| YAML | `yaml` |
| Images | `images` |
| RDF / Turtle / GraphML | `knowledge` |
| Audio / video | `audio` (and `ffmpeg` on PATH) |
| OpenAI embeddings | `openai` |

Not supported: generic `.zip` (unless it is a SCORM package containing `imsmanifest.xml`), BIFF `.xls` (convert to `.xlsx`).

## From this repository (development)

```bash
pip install -e ".[postgres,local,embeddings,extractors,openai]"
```

For **usage demos**, prefer the PyPI install inside [DemoCodeLive](https://github.com/Matrixxboy/memotrix/tree/main/DemoCodeLive) so you are testing the published wheel, not `PYTHONPATH` pointing at `src/`.

## Verify

```python
import memotrix
from memotrix import Memory
from memotrix.embeddings import FakeEmbeddings

memory = Memory(embeddings=FakeEmbeddings(dim=8), backend="memory")
print("ok", memotrix.__file__)
memory.close()
```

If this import resolves to a path under this repo’s `src/memotrix`, you are **not** using the PyPI wheel. Use a clean virtualenv as shown in DemoCodeLive.
