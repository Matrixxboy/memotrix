"""13 — HuggingFaceEmbeddings (downloads the model; optional).

Run only when you want a real encoder:

    python demos/13_huggingface.py
"""

from __future__ import annotations

import os

from memotrix import Memory
from memotrix.embeddings import HuggingFaceEmbeddings
from memotrix.vectorstores import InMemoryStore


def main() -> None:
    if os.getenv("MEMOTRIX_SKIP_HF") == "1":
        print("SKIP: MEMOTRIX_SKIP_HF=1")
        return
    model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embeddings = HuggingFaceEmbeddings(model=model)
    memory = Memory(embeddings=embeddings, store=InMemoryStore(embeddings))
    memory.add_text("User prefers dark mode.", source_id="hf-prefs")
    hits = memory.search("theme preference", top_k=1)
    print([(h.get("score"), (h.get("chunk_text") or "")[:80]) for h in hits])
    memory.close()


if __name__ == "__main__":
    main()
