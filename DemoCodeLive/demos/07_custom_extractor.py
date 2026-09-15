"""07 — register_extractor for a custom extension."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import make_memory

from memotrix.filetypes import register_extractor
from memotrix.utils.outputSturcture import build_document

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def extract_memo(path: Path, **kwargs):
    text = path.read_text(encoding="utf-8")
    return build_document(path, f"MEMO FILE\n{text}", extra_metadata={"file_type": "memo"})


def main() -> None:
    register_extractor(".memo", extract_memo)
    custom = SAMPLES / "note.memo"
    custom.write_text("Ship hybrid search on Friday.", encoding="utf-8")
    memory = make_memory()
    print(memory.add(custom))
    hits = memory.search("hybrid search", top_k=1)
    print(hits[0].get("chunk_text", "")[:200] if hits else "no hits")
    memory.close()


if __name__ == "__main__":
    main()
