"""01 — Hello memory: add_text + search."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import make_memory


def main() -> None:
    memory = make_memory()
    stats = memory.add_text(
        "User prefers dark mode and Python 3.12.",
        source_id="prefs",
        memory_type="semantic",
    )
    print("add_text:", stats)
    hits = memory.search("What theme does the user prefer?", top_k=3)
    for hit in hits:
        print(f"score={hit.get('score')} text={hit.get('chunk_text', '')[:180]!r}")
    memory.close()


if __name__ == "__main__":
    main()
