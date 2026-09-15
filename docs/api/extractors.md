---
title: "Extractors API"
---

# Extractors API

```python
from memotrix.filetypes import extract_file, supported_extensions, register_extractor, ExtractorRegistry
```

- `extract_file(path, *, describe_images=True, generate_srt=False)` — same router `Memory.add` uses by default.
- `supported_extensions()` — sorted list of wired + registered suffixes.
- `register_extractor(extension, fn_or_instance)` — plugin registry.

`DocumentData` fields: `metadata`, `sections`, `tables`, `images`, `text`, `raw_text`, `language`, `summary`, `transcripts`, `validation`.

Helper: `memotrix.utils.outputSturcture.build_document`.
