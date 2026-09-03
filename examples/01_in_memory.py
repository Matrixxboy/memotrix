"""In-process memory: add_text + search (no Postgres, no API key)."""

from memotrix import Memory
from memotrix.embeddings import FakeEmbeddings

memory = Memory(embeddings=FakeEmbeddings(dim=8), backend="memory")
memory.add_text(
    "User prefers dark mode and Python 3.12.",
    source_id="prefs",
    memory_type="semantic",
)
hits = memory.search("What theme does the user prefer?", top_k=3)
for hit in hits:
    print(hit.get("filename"), hit.get("chunk_text", "")[:200])
memory.close()
