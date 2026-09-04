---
title: "System Capabilities"
category: "Capabilities"
status: "Generated"
last_updated: "2026-09-04"
---

# System Capabilities

## Core Capabilities

### 1. Hybrid Search Retrieval
**What it does:** Retrieves the most relevant memory chunks using a combination of dense vector embeddings and sparse BM25 scores.
**How it works:** Queries are processed by both the dense index (HNSW/pgvector) and the sparse index (BM25). The results are fused to provide a unified ranking.

### 2. Multi-format Document Ingestion
**What it does:** Parses and extracts text from various file formats.
**How it works:** Uses dedicated extractors (e.g., PyMuPDF for PDFs, python-docx for Word documents) to convert raw files into standardized text chunks suitable for embedding.

### 3. Agentic Memory Management
**What it does:** Allows tagging memories with specific types (e.g., `episodic`, `procedural`, `semantic`) and `session_id`s.
**How it works:** The `add_text` API accepts metadata filters which are stored alongside the vectors and used for exact-match filtering during retrieval.

### 4. Pluggable Backends
**What it does:** Supports both in-memory and persistent database storage.
**How it works:** Implements an abstract `Store` interface with concrete implementations like `InMemoryStore` and `PostgresStore`.
