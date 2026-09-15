---
title: "RAG agent loop"
---

# Use case: RAG agent loop

Memotrix retrieves. **You** generate.

```python
from openai import OpenAI  # pip install "memotrix[openai]" plus openai client
from memotrix import Memory
from memotrix.embeddings import HuggingFaceEmbeddings

memory = Memory(embeddings=HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5"), backend="memory")
memory.add_text("To deploy Alpha, run `alpha deploy --force`. Never deploy on Fridays.", memory_type="procedural")

client = OpenAI()  # needs OPENAI_API_KEY

def ask(question: str) -> str:
    hits = memory.search(question, top_k=3, memory_type="procedural")
    context = "\n\n".join(h.get("chunk_text") or "" for h in hits)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer only from the context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content

# answer = ask("How do I deploy Alpha?")
```

You can use `OpenAIEmbeddings(model="text-embedding-3-small")` instead of HuggingFace if `OPENAI_API_KEY` is set. Dimension is inferred from the first embeddings API response.

The DemoCodeLive RAG script **prints** the assembled prompt and does not call a paid API unless you uncomment it.

See also `docs/tutorials/02-build-a-rag-agent.md`.
