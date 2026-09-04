# Tutorial 1: Quickstart (5 Minutes)

In this tutorial, you'll learn how to set up Memotrix and add your first document into memory.

## Prerequisites

Memotrix requires Python 3.10+.

## 1. Installation

Memotrix uses granular optional dependencies, so you only install what you need. For this quickstart, we will use the in-memory backend with local HuggingFace embeddings, and we will extract a standard PDF.

```bash
pip install "memotrix[memory,pdf]"
```

*Note: The quotes around the package name are necessary for zsh and some other shells.*

## 2. Initialize Memory

To use Memotrix, you need to instantiate the `Memory` facade. 
By default, this will use an in-memory HNSW index (dense vectors) and an in-memory BM25 index (keyword search).

```python
from memotrix import Memory
from memotrix.embeddings import HuggingFaceEmbeddings

# 1. Choose an embedding model
# We use BGE-small because it's extremely fast for local execution.
embeddings = HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")

# 2. Initialize the Memory system
memory = Memory(embeddings=embeddings, backend="memory")
```

## 3. Ingest a Document

Let's add a document to our memory.

```python
# Create a sample text file
with open("sample_notes.txt", "w") as f:
    f.write("Memotrix is a composable hybrid memory library for agents. It supports dense and sparse indexes.")

# Add the file to Memory
stats = memory.add("sample_notes.txt")
print("Ingestion stats:", stats)
# Output: {'chunks': 1, 'merged': 0, 'inserted': 1}
```

The `add` method automatically:
1. Detects the file type.
2. Extracts the raw text and metadata.
3. Chunks the text (default 500 chars).
4. Embeds the chunks.
5. Inserts them into both the Dense (vector) and Sparse (keyword) indexes.

## 4. Search the Memory

Now, we can query our memory. Memotrix uses **Hybrid Search** by default, meaning it combines semantic meaning (vectors) with exact keyword matching (BM25) and fuses the results using Reciprocal Rank Fusion (RRF).

```python
results = memory.search("What is Memotrix?", top_k=1)

for hit in results:
    print(f"Score: {hit['score']}")
    print(f"Content: {hit['chunk_text']}")
```

## 5. Add Direct Facts

You can also add free-text facts or chat history directly without a file.

```python
memory.add_text(
    "The user's favorite programming language is Python.",
    memory_type="semantic"
)
```

## Next Steps

Congratulations! You've successfully initialized Memotrix, ingested a file, and retrieved information using hybrid search.

Next, check out [Tutorial 2: Build a RAG Agent](02-build-a-rag-agent.md) to learn how to hook this memory up to an LLM.
