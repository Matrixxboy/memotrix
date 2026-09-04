# Memory Types (Semantic, Episodic, Procedural)

In traditional vector databases, all data is dumped into a single index. In Memotrix, data can be logically categorized into **Memory Types**. This concept borrows from human cognitive psychology to help AI Agents structure their knowledge base.

By default, the `memory_type` is stored as a metadata field (`payload.memory_type`) on every chunk.

## The Three Memory Types

Memotrix doesn't enforce strict rules on these strings, but the following convention is strongly recommended for Agent development:

### 1. Semantic Memory (`semantic`)
This is the default memory type. It represents **factual, world knowledge**. 
- **Examples:** Company policies, Wikipedia articles, product manuals.
- **When to use:** When you `add("report.pdf")`, it defaults to `semantic`.

### 2. Episodic Memory (`episodic`)
This represents **events, experiences, and history**.
- **Examples:** Chat logs between the user and the agent, tool execution logs, user feedback.
- **When to use:** Whenever the agent takes an action or has a conversation. 
- **Best Practice:** Combine `episodic` with a `session_id` to isolate context.
```python
memory.add_text("User said: Make the button blue", memory_type="episodic", session_id="session_123")
```

### 3. Procedural Memory (`procedural`)
This represents **instructions, workflows, and "how-to" knowledge**.
- **Examples:** "How to deploy the app", API rate limit rules, internal coding guidelines.
- **When to use:** To guide the Agent's decision-making process. Procedural memory is often retrieved to inject strict instructions into the LLM's system prompt.

## Filtering by Memory Type

Categorizing your memories allows your Agent to perform highly targeted searches.

```python
# The agent wants to know how to do something:
hits = memory.search("deploy app", memory_type="procedural")

# The agent wants to recall past interactions with this specific user:
hits = memory.search("what color did the user want?", memory_type="episodic", session_id="session_123")
```

This ensures the agent doesn't confuse a company policy (semantic) with a specific instruction the user gave five minutes ago (episodic).
