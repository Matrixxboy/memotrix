# Memotrix Documentation Hub

Welcome to the Memotrix developer documentation. Memotrix is a composable hybrid memory and RAG library for AI agents.

## Documentation Structure

This documentation is organized into four main sections, following the [Diátaxis framework](https://diataxis.fr/):

### 1. [Tutorials](tutorials/)
Step-by-step guides for learning how to use Memotrix from scratch. Start here if you are new to the framework.
- [Quickstart: 5-minute setup](tutorials/01-quickstart.md)
- [Build a RAG Agent](tutorials/02-build-a-rag-agent.md)

### 2. [How-To Guides](how-to/)
Goal-oriented recipes for solving specific problems with Memotrix.
- [Adding Custom File Extractors](how-to/custom-extractors.md)
- [Migrating to PostgreSQL (pgvector)](how-to/postgres-migration.md)
- [Tuning Hybrid Search](how-to/hybrid-search-tuning.md)

### 3. [Concepts](concepts/)
Theoretical explanations of how Memotrix works under the hood. Read these to understand the architecture deeply.
- [Memory Types (Semantic, Episodic, Procedural)](concepts/memory-types.md)
- [Chunking and Neighbor Expansion](concepts/chunking-and-expansion.md)

### 4. [API Reference](api-reference/)
Detailed technical specifications of the Memotrix classes and methods.
- [The `Memory` Facade](api-reference/memory-facade.md)
- [Vector Stores (`InMemoryStore`, `PostgresStore`)](api-reference/vectorstores.md)

---
> **Note on Architecture Documentation:**
> For internal project history, system capabilities, and high-level UML architecture diagrams, please refer to the [`docBook/`](../docBook/) directory at the root of the repository.
