---
title: "Exceptions"
---

# Exceptions

```python
from memotrix.utils.exceptions import (
    ConfigurationError,
    DocumentExtractionError,
    UnsupportedDocumentTypeError,
    EmptyDocumentError,
    ExtractionBackendError,
    MissingDependencyError,
)
```

`ConfigurationError` is also exported from `memotrix`.

| Exception | Typical cause |
|---|---|
| `ConfigurationError` | Missing model, DSN, empty `add_text`, bad backend, bad store object |
| `UnsupportedDocumentTypeError` | Unknown extension, generic zip, `.xls` |
| `EmptyDocumentError` | Extractor produced no content |
| `MissingDependencyError` | Optional extra not installed (e.g. HEIC) |
| `ExtractionBackendError` | Parser backend failed |

Also: `FileNotFoundError` / `ValueError` from extractors when the path is missing or not a file.
