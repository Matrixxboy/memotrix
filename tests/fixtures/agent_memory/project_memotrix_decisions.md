# Memotrix Architecture Decisions

## Storage

Memotrix stores dense vectors and keyword indexes in PostgreSQL with pgvector. There is no in-memory-only production path. The primary table is memotrix_chunks.

## Retrieval Stack

Hybrid search uses Reciprocal Rank Fusion over dense HNSW and PostgreSQL full-text search. A cross-encoder reranker (ms-marco-MiniLM-L-6-v2) rescores candidates before return.

## Chunking

Default CHUNK_SIZE is 500 characters with 50 character overlap. Chunking prefers whole paragraphs, then sentences, then words — never mid-word splits.

## Embedding Model

Current production embedding model is BAAI/bge-small-en-v1.5 at 384 dimensions. It replaced all-MiniLM-L6-v2 as a drop-in upgrade with stronger retrieval quality.
