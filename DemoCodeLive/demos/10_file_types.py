"""10 — Multiple wired file types that do not need optional extras."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import make_memory

SAMPLES = Path(__file__).resolve().parent.parent / "samples"

FILES = [
    "notes.txt",
    "facts.md",
    "meta.json",
    "data.csv",
    "sample_code.py",
    "sample.log",
    "chat.json",
]


def main() -> None:
    memory = make_memory(dim=24)
    for name in FILES:
        path = SAMPLES / name
        stats = memory.add(path)
        print(name, "chunks=", stats.get("chunks"), "inserted=", stats.get("inserted"))
    print("sources:", [s["filename"] for s in memory.list_sources()])
    print("search log error:", memory.search("connection refused", expand_mode="none")[:1])
    memory.close()


if __name__ == "__main__":
    main()
