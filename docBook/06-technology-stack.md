---
title: "Technology Stack"
category: "Architecture"
status: "Generated"
last_updated: "2026-09-04"
---

# Technology Stack

## Core Language & Frameworks
- **Python (>=3.10):** The core programming language.

## Vector Search & Embeddings
- **hnswlib:** Provides fast, in-memory approximate nearest neighbor (ANN) search. Used as the local dense index.
- **rank-bm25:** Provides the sparse (keyword) search capability for the local index.
- **sentence-transformers:** Used for local, open-weights embedding generation (e.g., HuggingFace models).
- **openai:** Optional integration for OpenAI's embedding models (`OpenAIEmbeddings`).

## Database
- **PostgreSQL + pgvector:** The persistent backend choice for scalable vector similarity search. Managed via `psycopg`.

## Document Extraction
- **pymupdf, python-docx, python-pptx, openpyxl, pandas:** For standard document and spreadsheet extraction.
- **Pillow, pillow-heif:** For image processing.
- **beautifulsoup4:** For HTML parsing.
- **faster-whisper:** For audio transcription.
- **rdflib:** For knowledge graph extraction.

## Why it is used
The stack is designed to be highly modular. By using optional dependencies (e.g., `pip install memotrix[postgres]`), the base package remains lightweight, pulling in heavy dependencies only when the specific functionality is required by the user.
