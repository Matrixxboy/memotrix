---
title: "Key Features"
category: "Features"
status: "Generated"
last_updated: "2026-09-04"
---

# Key Features

## 1. Environment-Based Configuration
**Purpose:** Instantiate the memory system safely in production environments.
**Technical implementation:** `Memory.from_env()` reads from `EMBEDDING_MODEL`, `DATABASE_URL`, and `MEMOTRIX_BACKEND` to automatically configure either local or remote connections without hardcoding secrets.

## 2. Dynamic Extractors
**Purpose:** Automatically route files to the correct parsing algorithm.
**Technical implementation:** The `add(path)` method infers file types and utilizes extractors from the `filetypes/` directory to handle everything from standard text documents to complex formats like FHIR JSON or audio via Whisper.

## 3. Composable Stores
**Purpose:** Flexibility in deployment.
**Technical implementation:** The `vectorstores.py` module defines the base classes, allowing seamless switching between `InMemoryStore` and `PostgresStore`.

## 4. Custom Memory Types
**Purpose:** Allow fine-grained control over what the agent remembers.
**Technical implementation:** Users can specify `memory_type` strings (e.g. "semantic", "procedural") which are stored as metadata. The `search` function accepts a `filters` dictionary to perform exact-match filtering.
