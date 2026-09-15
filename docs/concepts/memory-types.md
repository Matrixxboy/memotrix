---
title: "Memory types"
---

# Memory types (semantic, episodic, procedural)

Memotrix does not run three separate databases. `memory_type` is a **string on the chunk payload**. Default for `add()` and `add_text()` is `"semantic"`. Search can exact-match filter on it.

The library does not forbid other strings. The three names below are the convention used in this project and in agent demos.

## Semantic

Facts and world knowledge: manuals, policies, product copy.

```python
memory.add("handbook.pdf")  # metadata memory_type defaults to semantic
memory.add_text("Q3 revenue was 42 million.", memory_type="semantic")
```

## Episodic

Events and history: chat turns, tool results, “what happened in this session”.

```python
memory.add_text(
    "User said: make the button blue.",
    memory_type="episodic",
    session_id="session_123",
)
hits = memory.search(
    "what color did the user want?",
    memory_type="episodic",
    session_id="session_123",
)
```

`session_id` is also an exact-match payload field.

## Procedural

How-to instructions for the agent (deploy steps, citation rules).

```python
memory.add_text("Always cite source_path in answers.", memory_type="procedural")
hits = memory.search("how should answers be cited?", memory_type="procedural")
```

## Filters

`search(..., filters={"department": "DevOps"})` exact-matches extra metadata you passed to `add_text(..., metadata={...})`. Combined with `session_id` / `memory_type` kwargs.

There is no partial-match or range filter in the public API.
