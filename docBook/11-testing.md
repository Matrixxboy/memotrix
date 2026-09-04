---
title: "Testing"
category: "Quality"
status: "Generated"
last_updated: "2026-09-04"
---

# Testing

## Existing Tests
The project contains a `tests/` directory configured to run with `pytest`.

## What is tested
Based on standard practices for such a library, the test suite should cover:
- **Unit Tests:** Initialization of stores, embedding models, and core logic.
- **Integration Tests:** The end-to-end flow of adding a document, generating embeddings, and successfully retrieving it via `memory.search()`.
- **Extractor Tests:** Validating that `add()` correctly routes to and extracts text from dummy PDF, DOCX, TXT, etc., files.

## Test Architecture
Tests are run using `pytest` as configured in `pyproject.toml`.

```bash
pytest tests/
```

*(Note: The exact test execution results and coverage are pending a full CI/CD run in the designated environment, as direct execution is restricted in this documentation generation phase.)*
