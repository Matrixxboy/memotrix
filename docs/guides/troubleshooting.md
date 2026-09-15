---
title: "Troubleshooting"
---

# Troubleshooting

## `hnswlib` will not install on Windows

Cause: `hnswlib` is built from source; Microsoft C++ Build Tools are missing. There is no official Windows binary wheel for 0.8.0.

Fix: install [Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), or run the demos on Linux/macOS, or use WSL.

Then: `pip install "memotrix[local]"`.

## `ModuleNotFoundError: PIL` after `pip install memotrix`

Cause: 0.2.0 imports Pillow from `memotrix.utils.ai_integration` even when you do not ingest images.

Fix: `pip install pillow` or `pip install "memotrix[images]"`.

## PyPI `__version__`

Package **0.2.0** on PyPI still reports `memotrix.__version__ == "0.1.0"`. Source **0.2.1** syncs `__version__` with `pyproject.toml`. Until you publish 0.2.1, use `importlib.metadata.version("memotrix")`.

## `ModuleNotFoundError: hnswlib` / `rank_bm25` / `sentence_transformers`

Cause: `pip install memotrix` without extras.

Fix: `pip install "memotrix[memory]"` (or `[local]` if you only need FakeEmbeddings + HNSW). On Windows, see the hnswlib section above.

Verify: `python -c "from memotrix.vectorstores import InMemoryStore"`.

## Importing `memotrix` uses the git checkout, not PyPI

Cause: `PYTHONPATH` includes `src/` or you ran from an editable install.

Fix: new venv as in DemoCodeLive; `print(memotrix.__file__)` should be under `site-packages`.

## `ConfigurationError: EMBEDDING_MODEL is not set`

Cause: `Memory.from_env()` without env.

Fix: set `EMBEDDING_MODEL` or pass `embeddings=` explicitly.

## `ConfigurationError: Postgres is not configured`

Cause: backend postgres without DSN.

Fix: `DATABASE_URL` or `connection=`.

## `UnsupportedDocumentTypeError`

Cause: unknown suffix, generic zip, or `.xls`.

Fix: convert files; or `register_extractor`.

## `EmptyDocumentError`

Cause: blank file.

Fix: skip empty sources.

## Search returns empty or irrelevant hits

Causes: `FakeEmbeddings`; filters that match nothing; empty index; `memory_type` / `session_id` mismatch (exact match).

Fix: confirm `list_sources()`; drop filters; use a real embedding model.

## Postgres dimension errors after changing models

Cause: `vector(n)` column width is fixed.

Fix: new `table_name` or drop table.

## Whisper / ffmpeg failures

Cause: missing `audio` extra or ffmpeg.

Fix: install extra; `ffmpeg -version`.

## HNSW capacity

Cause: ingesting more than `max_elements`.

Fix: raise `MemoryConfig.max_elements` **before** creating the store (cannot resize later).
