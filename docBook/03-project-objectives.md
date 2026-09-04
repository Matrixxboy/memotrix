---
title: "Project Objectives"
category: "Overview"
status: "Generated"
last_updated: "2026-09-04"
---

# Project Objectives

## Objectives
1. **Provide a Composable Memory Interface:** Offer a simple `Memory` class that can be instantiated with different embedding models and vector stores.
2. **Support Hybrid Search:** Seamlessly integrate dense (vector) and sparse (keyword) search for optimal retrieval accuracy.
3. **Versatile File Extraction:** Support a vast array of file types (PDF, Office, HTML, images, audio, etc.) out of the box.
4. **Scalability:** Allow developers to start with an in-memory setup and smoothly transition to a PostgreSQL (pgvector) backend.

## Scope
The scope includes document ingestion, chunking, embedding generation, storage, and retrieval. It does not natively include the execution of the LLM generation itself, focusing purely on the memory and retrieval (RAG) aspect of the pipeline.

## Target Users
- AI developers building conversational agents.
- Data engineers setting up RAG pipelines.
- Researchers requiring local, in-memory search capabilities for experiments.
