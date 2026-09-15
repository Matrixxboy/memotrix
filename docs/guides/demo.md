---
title: "Live demos"
---

# Live demos

Full copy-paste scripts live in [DemoCodeLive](https://github.com/Matrixxboy/memotrix/tree/main/DemoCodeLive). They import **PyPI** `memotrix`, not this repo’s `src/`.

```bash
pip install memotrix
pip install "memotrix[local]"   # required for InMemoryStore
pip install pillow              # required to import Memory in 0.2.0
```

See [run notes](https://github.com/Matrixxboy/memotrix/blob/main/DemoCodeLive/RUN_NOTES.md) for an actual install attempt and failures recorded on Windows without MSVC.

Map of scripts to [use cases](../use-cases/index.md):

| Demo | Use case page |
|---|---|
| 01 | [Agent memory](../use-cases/agent-long-term-memory.md) |
| 02 | [Document Q&A](../use-cases/document-question-answering.md) |
| 03–04 | [Memory types](../concepts/memory-types.md) |
| 05 | [Chunking](../concepts/chunking-and-expansion.md) |
| 07 | [Custom extractors](../use-cases/custom-extractors.md) |
| 09 | [RAG agent](../use-cases/rag-agent.md) |
| 10 | [File ingestion](../use-cases/file-ingestion.md) |
| 11 | [Postgres](../use-cases/postgres-persistence.md) |
