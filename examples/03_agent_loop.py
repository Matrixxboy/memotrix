"""Minimal agent loop: retrieve memories, then prompt your own LLM."""

from memotrix import Memory
from memotrix.embeddings import FakeEmbeddings

memory = Memory(embeddings=FakeEmbeddings(dim=8), backend="memory")
memory.add_text("Project Memotrix stores hybrid RAG chunks for agents.", source_id="intro")

query = "What does Memotrix store?"
hits = memory.search(query, top_k=3, max_tokens_returned=400)
context = "\n\n".join(hit.get("chunk_text") or "" for hit in hits)

# Swap this print for your LLM: chat.completions.create(messages=[...])
print("Context for the model:\n", context)
print("\nUser question:", query)
memory.close()
