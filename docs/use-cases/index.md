---
title: "Use cases"
description: "Every supported Memotrix usage pattern"
---

# Use cases

Each page maps to a real API in `memotrix`. Live scripts: [DemoCodeLive](https://github.com/Matrixxboy/memotrix/tree/main/DemoCodeLive).

## Agent and RAG

| Use case | What you call | Document |
|---|---|---|
| Long-term facts, sessions, procedures | `add_text` + `memory_type` / `session_id` | [Agent long-term memory](./agent-long-term-memory.md) |
| Question answering over files | `add` + `search` | [Document Q&A](./document-question-answering.md) |
| Retrieve then prompt an LLM | `search` → your chat API | [RAG agent](./rag-agent.md) |
| Durable store across processes | `PostgresStore` / `from_env` | [Postgres](./postgres-persistence.md) |

## Ingestion

| Use case | What you call | Document |
|---|---|---|
| Any wired file type | `memory.add(path)` | [File ingestion](./file-ingestion.md) |
| CSV / Excel / JSON / YAML / SQL / XML | same | [Structured data](./structured-data.md) |
| Source code | `.py` `.js` `.ts` `.java` `.go` `.rs` `.cpp` | [Codebase memory](./codebase-memory.md) |
| Images, audio, video | extras + optional Whisper | [Images and media](./images-and-media.md) |
| Email, chat, logs | `.eml` `.mbox` chat JSON, WhatsApp `.txt`, `.log` | [Email, chat, logs](./email-chat-logs.md) |
| FHIR, GeoJSON, RDF | sniffed JSON or KG extensions | [Knowledge graphs and FHIR](./knowledge-graphs-and-fhir.md) |
| Proprietary format | `register_extractor` or `extract_file=` | [Custom extractors](./custom-extractors.md) |

## Operations

| Use case | API |
|---|---|
| Inspect what is stored | `list_sources()` |
| Remove one source | `delete(source_path_or_id)` |
| Parse without indexing | `extract(path)` |
| Ingest pre-parsed docs | `add_documents(...)` |
| Tune retrieval | `MemoryConfig.retrieval` |
| Tests without a real model | `FakeEmbeddings(dim=...)` |

## What Memotrix is not

- Not a chatbot. You own the LLM call.
- Not a multi-tenant SaaS. Isolation is your filters and your database.
- Not a REST server in this package.
- Not a crawler. You pass local paths.
