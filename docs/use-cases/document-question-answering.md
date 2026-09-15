---
title: "Document question answering"
---

# Use case: document question answering

**Problem.** You have reports, notes, or PDFs. You want answers grounded in those files, not the model’s training data.

**What Memotrix does.** `add(path)` extracts and indexes. `search(query)` returns the top chunks for you to put in a prompt.

```python
memory.add("quarterly_notes.txt")
hits = memory.search("what is the revenue?", top_k=3, max_tokens_returned=800)
context = "\n\n".join(h.get("chunk_text") or "" for h in hits)
# pass `context` to your LLM
```

**Filters.** `memory_type="semantic"` (default on files) or extra metadata if you set it via `add_documents`.

**Sources.** `list_sources()` returns `{source_path, filename, chunks}` for each ingested path.

**Delete.** `delete(resolved_path)` or filename / `source_id` as implemented by `delete_by_source_path` on both indexes.

**Limitations.** Quality depends on the embedding model and chunk size. Neighbor expansion can still miss distant related sections. PDFs need `pip install "memotrix[pdf]"` (or `extractors`). Empty files raise `EmptyDocumentError`.

Demos: `DemoCodeLive/demos/02_ingest_files.py`, `09_rag_loop.py`.
