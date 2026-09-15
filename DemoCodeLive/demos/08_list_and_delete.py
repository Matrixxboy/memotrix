"""08 — list_sources and delete."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import make_memory


def main() -> None:
    memory = make_memory()
    memory.add_text("Temporary fact.", source_id="temp-fact")
    print("before:", memory.list_sources())
    deleted = memory.delete("temp-fact")
    print("deleted:", deleted, "after:", memory.list_sources())
    memory.close()


if __name__ == "__main__":
    main()
