---
title: "Agent long-term memory"
---

# Use case: agent long-term memory

**Problem.** An LLM forgets everything when the context window rolls. You need durable facts, session events, and standing instructions.

**What Memotrix does.** `add_text` writes free-form memories into the same hybrid index as files. You tag `memory_type` and optional `session_id`.

```python
from memotrix import Memory
from memotrix.embeddings import FakeEmbeddings  # or HuggingFaceEmbeddings

memory = Memory(embeddings=FakeEmbeddings(dim=16), backend="memory")

memory.add_text(
    "User prefers dark mode.",
    source_id="prefs",
    memory_type="semantic",
)
memory.add_text(
    "User asked to hide the billing tab in this chat.",
    memory_type="episodic",
    session_id="chat-42",
)
memory.add_text(
    "Always cite source_path when answering from memory.",
    memory_type="procedural",
    source_id="cite-rule",
)

facts = memory.search("preferred theme", memory_type="semantic")
session = memory.search("what should be hidden?", memory_type="episodic", session_id="chat-42")
rules = memory.search("how to cite", memory_type="procedural")

memory.close()
```

**Inputs:** non-empty string; optional `source_id` (else `text:<uuid>`); `memory_type`; `session_id`; extra `metadata`.

**Processing:** wrapped as `DocumentData` with `source_path` = `source_id`, then the ingestion pipeline (chunk, embed, index, dedup).

**Output:** same stats dict as `add` (`chunks`, `inserted`, `merged`, …). Search returns hit dicts with `chunk_text` and payload fields.

**Limitations:** filters are exact match only. In-memory store is lost when the process exits. `FakeEmbeddings` will not rank like a real model.

See [Memory types](../concepts/memory-types.md) and demo `DemoCodeLive/demos/03_memory_types.py`.
