---
title: "Core Workflows"
category: "Workflows"
status: "Generated"
last_updated: "2026-09-04"
---

# Core Workflows

## Data Ingestion Workflow

```mermaid
flowchart TD
    A["memory.add / memory.add_text"] --> B{"Is File?"}
    B -- "Yes" --> C["Route to Extractor"]
    C --> D["Extract Text and Metadata"]
    B -- "No" --> D
    D --> E["Chunking"]
    E --> F["Generate Dense Embeddings"]
    F --> G["Store in Vector Store"]
    G --> H["Update Sparse Index"]
```

## Retrieval Workflow

```mermaid
flowchart TD
    A["memory.search query"] --> B["Generate Query Embedding"]
    A --> C["Query Sparse Index - BM25"]
    B --> D["Query Dense Index"]
    C --> E["Normalize and Fuse Scores"]
    D --> E
    E --> F["Apply Metadata Filters"]
    F --> G["Return Top K Hits"]
```

## Explanation
The ingestion workflow normalizes all input (whether raw text or complex files) into a standard chunk format. The retrieval workflow ensures that both keyword matches (sparse) and semantic meaning (dense) contribute to the final ranking, yielding more robust results than vector-search alone.
