---
title: "Documentation Audit"
category: "Audit"
status: "Updated"
last_updated: "2026-09-15"
---

# Documentation Audit

Public website docs: [`docs/`](../docs/index.md) (merged with this DocBook). Live demos: [`DemoCodeLive/`](../DemoCodeLive/README.md).

## Repository coverage

- [x] `Memory` facade, stores, embeddings, extractors, hybrid search documented
- [x] Every wired file-type group has a use-case page
- [x] No REST API documented (none in this package)
- [x] Postgres schema described from `PostgresStore` behavior
- [x] Security notes are documentation-level, not a pentest

## Accuracy

- [x] No fabricated benchmark numbers
- [x] PyPI 0.2.0 extras and env vars match `pyproject.toml`
- [x] `__version__` mismatch (0.1.0 in wheel vs 0.2.0 dist) recorded
- [x] Demo install on Windows without MSVC recorded as failed `hnswlib` build

## Navigation

- Website hub: `docs/index.md` + `docs.json`
- DocBook hub: `00-index.md` points at `docs/`
