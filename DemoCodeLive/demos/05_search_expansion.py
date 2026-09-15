"""05 — expand_mode: neighbors, none, full_file."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import make_memory

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def main() -> None:
    memory = make_memory()
    memory.add(SAMPLES / "notes.txt")
    for mode in ("neighbors", "none", "full_file"):
        hits = memory.search("revenue", top_k=1, expand_mode=mode)
        hit = hits[0] if hits else {}
        print(mode, "expansion=", hit.get("expansion"), "chars=", len(hit.get("chunk_text") or ""))
    memory.close()


if __name__ == "__main__":
    main()
