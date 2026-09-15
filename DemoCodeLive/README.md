# DemoCodeLive

Runnable demos for **PyPI** [`memotrix`](https://pypi.org/project/memotrix/) **0.2.0**.

These scripts must import from `site-packages`, not this repository’s `src/` folder.

## Setup (Windows / bash)

```bash
cd DemoCodeLive
python -m venv .venv

# Windows Git Bash / PowerShell:
.venv/Scripts/activate

# macOS / Linux:
# source .venv/bin/activate

pip install -U pip
pip install "memotrix[memory]"
```

`pip install memotrix` alone is not enough to construct `Memory` (no HNSW / BM25). `[memory]` adds the in-process stack and Sentence-Transformers. These demos use **`FakeEmbeddings`** so they do **not** download a HuggingFace model.

Optional:

```bash
pip install "memotrix[postgres,extractors,yaml]"
```

## Run

```bash
python run_all.py
python demos/01_hello_memory.py
```

| Script | Use case |
|---|---|
| `01_hello_memory.py` | `add_text` + `search` |
| `02_ingest_files.py` | `add` a text file |
| `03_memory_types.py` | semantic / episodic / procedural filters |
| `04_sessions_and_filters.py` | `session_id` + metadata filters |
| `05_search_expansion.py` | `neighbors` / `none` / `full_file` |
| `06_extract_only.py` | `extract` without indexing |
| `07_custom_extractor.py` | `register_extractor` |
| `08_list_and_delete.py` | `list_sources` + `delete` |
| `09_rag_loop.py` | Build an LLM prompt from hits (no paid API by default) |
| `10_file_types.py` | txt, md, json, csv, py, log, chat json |
| `11_postgres.py` | Postgres if `DATABASE_URL` is set; otherwise skips |
| `12_from_env.py` | `Memory.from_env` if `EMBEDDING_MODEL` is set; otherwise skips |
| `13_huggingface.py` | Real `HuggingFaceEmbeddings` (downloads model; optional) |

## Known environment notes

On this machine, `pip install "memotrix[local]"` failed to build **hnswlib** (no official Windows wheel; needs MSVC). A bare `pip install memotrix==0.2.0` succeeds and imports from `site-packages`. Importing `Memory` also requires **Pillow** (unconditional import in `memotrix.utils.ai_integration` in 0.2.0).

```bash
pip install memotrix==0.2.0 pillow
pip install "memotrix[local]"   # needs a C++ compiler on Windows
```

The PyPI **0.2.0** wheel reports `__version__ = "0.1.0"`. This repo is **0.2.1** (version string fixed; Pillow is lazy-imported). Publish 0.2.1 to PyPI to ship those fixes.

## Docs

Website docs: [`../docs/index.md`](../docs/index.md) · [use cases](../docs/use-cases/index.md) · [run notes](./RUN_NOTES.md)
