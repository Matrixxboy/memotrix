"""06 — extract() without ingesting."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import make_memory

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def main() -> None:
    memory = make_memory()
    doc = memory.extract(SAMPLES / "notes.txt")
    print("filename metadata:", doc.metadata.get("filename"))
    print("text preview:", (doc.text or "")[:160])
    print("sources after extract-only:", memory.list_sources())
    memory.close()


if __name__ == "__main__":
    main()
