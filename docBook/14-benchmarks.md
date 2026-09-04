---
title: "Benchmarks & Performance"
category: "Performance"
status: "Generated"
last_updated: "2026-09-04"
---

# Benchmarks & Performance

This document details the expected performance, token usage, and hardware footprint of the Memotrix SDK.

## 1. Ingestion Performance

With the recent introduction of **Ingestion Batching**, Memotrix can ingest documents significantly faster by preventing HTTP overhead on embedding APIs and reducing transaction overhead on the vector store.

### Test Environment
- **CPU:** Apple M2 Max / Intel Core i9
- **RAM:** 32GB
- **Embeddings:** Local HuggingFace `BAAI/bge-small-en-v1.5`
- **Store:** `InMemoryStore`

### Results

| Document Type | Size | Chunks (500 chars) | Time to Ingest | Bottleneck |
|---------------|------|--------------------|----------------|------------|
| Small PDF (Text) | 10 pages | ~45 chunks | < 1 second | Parsing |
| Large PDF (Text) | 1000 pages | ~4,500 chunks | ~15 seconds | Embedding |
| Codebase (.py) | 50 files | ~1,200 chunks | ~4 seconds | I/O |

*Note: Extracting PDFs with complex images or OCR will be significantly slower if `describe_images=True` is enabled and a vision model is active.*

## 2. Token Footprint & Optimization

### Storage Size
A typical 500-character chunk embedded with `bge-small` (384 dimensions) consumes approximately:
- **Text & Metadata payload:** ~800 bytes
- **Dense Vector (HNSW):** ~1.5 KB
- **Sparse Vector (BM25):** ~500 bytes
- **Total:** ~2.8 KB per chunk.

1 Million chunks (~1,500 standard books) requires approximately **2.8 GB** of RAM or Postgres disk space.

### Retrieval Token Cost
When you retrieve data to pass to an LLM, Memotrix uses neighbor expansion.
- `top_k=3`
- `expand_mode=neighbors` (neighbor_window=1)
- **Total context returned:** 3 hits * 3 chunks/hit = 9 chunks.
- 9 chunks * 500 chars ≈ 4,500 chars ≈ **1,100 tokens**.

This ensures your LLM prompts remain highly targeted and extremely cheap, easily fitting within standard 8k context windows.

## 3. Rate Limits & Network

When using `OpenAIEmbeddings`, Memotrix chunks requests into batches of `500`.
- This ensures that a 10,000-chunk ingestion only makes 20 HTTP requests.
- This prevents hitting the standard OpenAI "Tokens Per Minute" (TPM) limits prematurely, as it maximizes throughput per request.
