---
title: "Email, chat, and logs"
---

# Use case: email, chat, and logs

## Email

`.eml` and `.mbox` → subject, from, to, date, body (HTML stripped when needed). Natural memory type for these texts is **episodic**, but `add()` still defaults `memory_type` to `semantic` unless you ingest via `add_documents` with your own metadata.

## Chat JSON / JSONL

If JSON looks like a list of messages (`content`/`text` + author/time keys), `ChatExtractor` runs. Nested `messages` / `chats` / `history` are detected.

## WhatsApp export

A `.txt` whose first lines match `[...] Name: ` is treated as chat, not a generic text file.

## Logs

`.log` files are split into windows of 40 non-empty lines, optionally headed with ISO timestamps if lines start with them.

```python
memory.add("app.log")
hits = memory.search("connection refused", expand_mode="none")
```

`expand_mode="none"` is often better for logs: neighbors may be unrelated lines.

Demo samples: `DemoCodeLive/samples/chat.json`, `sample.log`.
