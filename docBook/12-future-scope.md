---
title: "Future Scope & Roadmap"
category: "Future"
status: "Generated"
last_updated: "2026-09-04"
---

# Future Scope & Roadmap

## Short Term

### 1. Enhanced Test Coverage
- **Problem:** Ensuring all new file extractors are fully tested against edge cases.
- **Solution:** Expand the `pytest` suite with comprehensive mock documents.
- **Priority:** High

## Medium Term

### 1. Additional Vector Store Backends
- **Problem:** Users may want to use managed vector databases (e.g., Pinecone, Qdrant) instead of managing local Postgres.
- **Solution:** Implement new classes extending `vectorstores.Store`.
- **Priority:** Medium

### 2. Advanced Chunking Strategies
- **Problem:** Simple token-based chunking can split sentences or paragraphs awkwardly, losing context.
- **Solution:** Integrate semantic chunking or structural chunking (e.g., respecting Markdown headers).
- **Priority:** Medium

## Long Term

### 1. Distributed Hybrid Search
- **Problem:** In-memory HNSW + BM25 does not scale horizontally across multiple instances easily.
- **Solution:** Build a distributed indexing layer or deeply integrate with scalable search engines like Elasticsearch/OpenSearch for the sparse component alongside pgvector.
- **Priority:** Low
