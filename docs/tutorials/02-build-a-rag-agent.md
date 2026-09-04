# Tutorial 2: Build a RAG Agent

In this tutorial, you will build a complete Retrieval-Augmented Generation (RAG) loop using Memotrix as the knowledge base and an LLM (like OpenAI) to generate answers based on the retrieved context.

## 1. Setup

Install the required packages. We will install the `openai` extra for the LLM.

```bash
pip install "memotrix[memory,openai]"
```

Set your OpenAI API key in your terminal:
```bash
export OPENAI_API_KEY="sk-..."
```

## 2. Initialize Memotrix

We will use `OpenAIEmbeddings` for the vector search this time to keep everything within the OpenAI ecosystem.

```python
import os
from memotrix import Memory
from memotrix.embeddings import OpenAIEmbeddings

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize memory
memory = Memory(embeddings=embeddings, backend="memory")
```

## 3. Populate Knowledge

Let's add some specific procedural knowledge that the LLM wouldn't know otherwise.

```python
memory.add_text(
    "To deploy the Alpha engine, you must run `alpha deploy --force`. Never deploy on Fridays.",
    memory_type="procedural",
    metadata={"department": "DevOps"}
)

memory.add_text(
    "The Alpha engine requires Node.js v18.0 or higher.",
    memory_type="semantic",
    metadata={"department": "DevOps"}
)
```

## 4. The Agent Loop

A RAG agent follows three steps: **Retrieve**, **Augment**, and **Generate**.

```python
from openai import OpenAI

client = OpenAI()

def ask_agent(question: str):
    # 1. Retrieve: Search Memotrix for relevant context
    hits = memory.search(
        question, 
        top_k=2, 
        # We can optionally filter by metadata!
        filters={"department": "DevOps"}
    )
    
    # 2. Augment: Format the context for the LLM
    context_blocks = [hit.get("chunk_text") for hit in hits if hit.get("chunk_text")]
    context_str = "\n\n---\n\n".join(context_blocks)
    
    prompt = f"""You are a helpful DevOps assistant. Answer the user's question using ONLY the context provided below.

Context:
{context_str}

User Question: {question}
"""

    # 3. Generate: Call the LLM
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

# Test the agent
answer = ask_agent("How do I deploy the Alpha engine, and are there any restrictions?")
print(answer)
```

**Expected Output:**
> To deploy the Alpha engine, you must run the command `alpha deploy --force`. However, there is a restriction: you must never deploy on Fridays.

## Conclusion

You have just built a fully functional RAG agent. By passing `context` from `memory.search()` into your LLM prompt, you ground the model's responses in your private data and prevent hallucinations.

For more advanced setups, look into:
- [Migrating to PostgreSQL](../how-to/postgres-migration.md) for persistent storage.
- [Chunking and Neighbor Expansion](../concepts/chunking-and-expansion.md) to provide the LLM with surrounding context.
