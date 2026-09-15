"""02 — Ingest a local file with memory.add."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import make_memory

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def main() -> None:
    memory = make_memory()
    path = SAMPLES / "notes.txt"
    print("add:", memory.add(path))
    hits = memory.search("what is the revenue?", top_k=3)
    for hit in hits:
        print(hit.get("filename"), hit.get("chunk_text", "")[:200])
    memory.close()


if __name__ == "__main__":
    main()
