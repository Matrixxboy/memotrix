"""03 — Semantic, episodic, and procedural memories."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import make_memory


def main() -> None:
    memory = make_memory()
    memory.add_text("Q3 revenue was 42 million.", memory_type="semantic", source_id="fact-q3")
    memory.add_text(
        "User asked to hide the billing tab.",
        memory_type="episodic",
        session_id="chat-42",
        source_id="ep-1",
    )
    memory.add_text(
        "Always cite source_path when answering from memory.",
        memory_type="procedural",
        source_id="proc-cite",
    )

    print("semantic:", memory.search("revenue", memory_type="semantic")[0].get("chunk_text", "")[:80])
    print(
        "episodic:",
        memory.search("billing", memory_type="episodic", session_id="chat-42")[0].get("chunk_text", "")[:80],
    )
    print("procedural:", memory.search("cite", memory_type="procedural")[0].get("chunk_text", "")[:80])
    memory.close()


if __name__ == "__main__":
    main()
