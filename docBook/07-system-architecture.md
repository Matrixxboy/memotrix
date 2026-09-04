---
title: "System Architecture"
category: "Architecture"
status: "Generated"
last_updated: "2026-09-04"
---

# System Architecture

## High-Level Architecture

The Memotrix architecture revolves around the `Memory` class, which acts as the orchestrator.

```mermaid
flowchart TD
    User["User / Agent"] --> Memory["Memory Interface"]
    Memory --> Extractor["File Extractors"]
    Memory --> Embeddings["Embedding Models"]
    Memory --> VectorStore["Vector Store Base"]

    Extractor --> TextChunks["Text Chunks and Metadata"]
    TextChunks --> Embeddings

    Embeddings --> Vectors["Dense Vectors"]
    TextChunks --> VectorStore
    Vectors --> VectorStore

    VectorStore --> InMemory["HNSW + BM25"]
    VectorStore --> Postgres["PostgreSQL + pgvector"]
```

## Component Architecture

1. **`memotrix/api.py` / `memory.py`:** Exposes the public API (`Memory.add`, `Memory.add_text`, `Memory.search`).
2. **`memotrix/filetypes/`:** Contains the **Extractor Plugin Registry** (`registry.py`) and specific extractors for different file formats.
3. **`memotrix/embeddings.py`:** Contains abstract and concrete implementations for generating embeddings.
4. **`memotrix/vectorDB/`:** Contains the indexing logic (`hnsw_index.py`, `sparse_index.py`, `postgres_store.py`).
5. **`memotrix/processes/`:** Manages internal data flows like chunking and retrieval fusion.

## Detailed Flows

### 1. Extractor Plugin Registry
Adding a new file type no longer requires modifying the core SDK. Developers can register new extractors dynamically.

```mermaid
flowchart LR
    File["Input File"] --> Router["extract_file"]
    Router --> Registry{"ExtractorRegistry"}
    Registry -- "Match Extension" --> CustomExt["Custom Extractor"]
    Registry -- "No Match" --> DefaultExt["Default Extractor"]
    CustomExt --> DocData["DocumentData"]
    DefaultExt --> DocData
```

### 2. Ingestion Batching Pipeline
To prevent memory exhaustion and network rate limits, Memotrix chunks large documents and batches them before sending to the Embedding model.

```mermaid
flowchart TD
    DocData["DocumentData"] --> Chunker["Chunking Algorithm"]
    Chunker --> Buffer["Chunk Buffer"]

    Buffer -- "Reaches batch_size 500" --> Embedder["Embeddings.embed_documents"]
    Embedder --> Vectors["Dense Vectors"]

    Vectors --> Store["PostgresStore"]
    Buffer -- "Continues" --> Embedder
```

## Design Philosophy
The system is built on dependency injection. The `Memory` instance doesn't hardcode how vectors are generated or stored; it accepts `embeddings` and `store` objects, making the framework highly extensible.
