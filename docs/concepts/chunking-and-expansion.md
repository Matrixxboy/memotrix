---
title: "Chunking and neighbor expansion"
---

# Chunking and neighbor expansion

## Chunking

Ingestion splits `DocumentData` text into windows:

- Default **500** characters
- Default **50** character overlap

Configured via `MemoryConfig.chunking` (`ChunkingConfig`). Overlap exists so a sentence split across a boundary still appears in two chunks.

Programming files are pre-split by class/function regexes in `ProgrammingFileExtractor` before this windowing. CSV files are grouped in batches of 8 rows. Logs are grouped in windows of 40 non-empty lines.

## DocumentStore

On ingest, Memotrix records chunk adjacency (`DocumentStore`). Retrieval can expand a hit to neighboring chunks in the same source.

## Expand modes

Passed on `search(..., expand_mode=...)` or `RetrievalConfig.expand_mode`.

| Mode | Behavior |
|---|---|
| `neighbors` (default) | Matched chunk ± `neighbor_window` (default 1), truncated by `max_tokens_returned` (default 800, ~4 chars/token) |
| `full_file` | Unique sources expanded to full reconstructed text (DocumentStore → all indexed chunks → limited disk re-read for some types) |
| `none` | Matched chunk text only |

Hit dicts include `chunk_text`, `content`, `matched_chunk`, `expansion`, `tokens_estimate`, and may include `truncated`, `neighbor_chunk_indices`.

Use `full_file` only when the source is small enough for the LLM context window.
