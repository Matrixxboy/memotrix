"""04 — session_id and extra metadata filters."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import make_memory


def main() -> None:
    memory = make_memory()
    memory.add_text(
        "Deploy Alpha with `alpha deploy --force`.",
        memory_type="procedural",
        metadata={"department": "DevOps"},
        source_id="deploy-alpha",
    )
    memory.add_text(
        "Marketing tagline is Remember everything.",
        memory_type="semantic",
        metadata={"department": "Marketing"},
        source_id="tagline",
    )

    devops = memory.search("deploy", filters={"department": "DevOps"})
    marketing = memory.search("tagline", filters={"department": "Marketing"})
    print("devops hits:", len(devops), devops[0].get("chunk_text", "")[:60] if devops else None)
    print("marketing hits:", len(marketing), marketing[0].get("chunk_text", "")[:60] if marketing else None)
    memory.close()


if __name__ == "__main__":
    main()
