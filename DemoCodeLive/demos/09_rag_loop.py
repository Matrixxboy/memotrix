"""09 — RAG loop: format search hits into a prompt (no API call)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import make_memory


def main() -> None:
    memory = make_memory()
    memory.add_text(
        "To deploy Alpha, run `alpha deploy --force`. Never deploy on Fridays.",
        memory_type="procedural",
        source_id="alpha-deploy",
    )
    query = "How do I deploy Alpha?"
    hits = memory.search(query, top_k=3, memory_type="procedural")
    context = "\n\n".join(h.get("chunk_text") or "" for h in hits)
    prompt = (
        "Answer only from the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
    )
    print(prompt)
    print("(Pass this prompt to your own LLM. DemoCodeLive does not call OpenAI.)")
    memory.close()


if __name__ == "__main__":
    main()
