---
title: "Codebase memory"
---

# Use case: codebase memory

`ProgrammingFileExtractor` reads source and splits on language-specific class/function patterns, then the ingest pipeline chunks those blocks.

Supported: `.py` `.js` `.ts` `.java` `.cpp` `.cc` `.cxx` `.go` `.rs`.

```python
memory.add("app/main.py")
hits = memory.search("where is hybrid search fused?", memory_type="semantic")
```

Metadata includes `file_type=source_code`, `language`, `block_count`.

**Limitations.** Regex splitting is heuristic, not a full AST. Unsupported languages: use `register_extractor` or ingest as `.txt`.

Demo: `DemoCodeLive/samples/sample_code.py` and `demos/10_file_types.py`.
